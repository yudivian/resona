import os
import uuid
import logging
import shutil
import gc
from typing import Optional, List, Dict, Any
from src.models import AppConfig, VoiceProfile, SourceType, TrackType
from src.backend.engine import TTSModelProvider, InferenceEngine, VoiceBlender
from src.backend.embedding import EmbeddingModelProvider, EmbeddingEngine
from src.backend.store import VoiceStore

logger = logging.getLogger(__name__)

class SessionManager:
    """
    Core orchestrator linking UI interactions to Backend Logic and Persistence.

    The SessionManager acts as the central controller for the application. It manages
    the lifecycle of the dual inference engines (Track A and Track B), handles
    temporary file registries to prevent disk clutter, and orchestrates the complex
    process of saving voices by coordinating between the InferenceEngine, the
    EmbeddingEngine (for semantic search), and the VoiceStore (for persistence).
    """

    def __init__(self, config: AppConfig):
        """
        Initializes the SessionManager with all necessary backend providers.

        This constructor sets up the entire backend pipeline:
        1. TTSModelProvider: Loads the base TTS model weights.
        2. InferenceEngine: Instantiates two separate engines for mixing capabilities.
        3. VoiceStore: Connects to the BeaverDB persistence layer.
        4. EmbeddingEngine: Prepares the semantic vectorization model.

        Args:
            config (AppConfig): The global application configuration object containing
                                paths, hardware settings, and model parameters.
        """
        self.config = config
        self.tts_provider = TTSModelProvider(config)
        self.engine_a = InferenceEngine(config, self.tts_provider)
        self.engine_b = InferenceEngine(config, self.tts_provider)
        
        self.store = VoiceStore(config)
        self.embed_provider = EmbeddingModelProvider(config)
        self.embed_engine = EmbeddingEngine(config, self.embed_provider)
        
        self._cached_blend_engine: Optional[InferenceEngine] = None
        self._last_blend_alpha: float = -1.0
        self.track_a_type: TrackType = TrackType.DESIGN
        self.track_b_type: TrackType = TrackType.DESIGN
        self._temp_files_registry: List[str] = []

    def set_language(self, lang_code: str):
        """
        Updates the target synthesis language for all active inference engines.

        This method ensures consistency across the application by setting the
        language code for both Track A and Track B. It also invalidates any
        cached blended engine, forcing a re-synthesis on the next request to
        ensure the new language is applied.

        Args:
            lang_code (str): The ISO language code (e.g., 'en', 'es', 'fr').
        """
        if self.engine_a.lang != lang_code:
            self.engine_a.lang = lang_code
            self.engine_b.lang = lang_code
            self._cached_blend_engine = None

    def register_temp_file(self, path: str):
        """
        Registers a temporary file path for automatic session cleanup.

        The session manager maintains a registry of all files generated during
        user interaction (clones, previews, anchors). These files are tracked
        so they can be safely deleted when the session resets or ends.

        Args:
            path (str): The absolute filesystem path to the temporary file.
        """
        if path and path not in self._temp_files_registry:
            self._temp_files_registry.append(path)

    def design_voice(self, track_id: str, prompt: str, seed: Optional[int] = None):
        """
        Triggers the Voice Design process for a specific track.

        This method delegates the generation request to the appropriate InferenceEngine.
        It accepts an optional seed to allow for deterministic reproduction of voices
        (Reroll functionality). Once generated, the anchor audio file is registered
        for cleanup.

        Args:
            track_id (str): The identifier of the track to modify ('A' or 'B').
            prompt (str): The text description of the desired voice characteristics.
            seed (Optional[int]): An integer seed to control the random generation process.
                                  If None, a random seed is used by the engine.
        """
        engine = self.engine_a if track_id == "A" else self.engine_b
        engine.design_identity(prompt, seed)
        if engine.last_anchor_path:
            self.register_temp_file(engine.last_anchor_path)
        self._cached_blend_engine = None

    def clone_voice(self, track_id: str, audio_path: str, transcript: str):
        """
        Triggers the Voice Cloning process using a reference audio file.

        This method instructs the InferenceEngine to extract the speaker embedding
        from the provided audio file. It correctly interfaces with the `extract_identity`
        method of the backend engine.

        Args:
            track_id (str): The identifier of the track to modify ('A' or 'B').
            audio_path (str): The absolute path to the reference WAV/MP3 file.
            transcript (str): The verbatim text content of the reference audio,
                              used to align the phonemes during extraction.
        """
        engine = self.engine_a if track_id == "A" else self.engine_b
        engine.extract_identity(audio_path, transcript)
        if engine.last_anchor_path:
            self.register_temp_file(engine.last_anchor_path)
        self._cached_blend_engine = None

    def list_library_voices(self) -> Dict[str, str]:
        """
        Retrieves a mapping of available library voices for UI selection.

        This method queries the VoiceStore for all persisted profiles and creates
        a simplified dictionary mapping voice names to their UUIDs. This is used
        to populate dropdown menus in the frontend.

        Returns:
            Dict[str, str]: A dictionary where keys are the human-readable voice names
                            and values are their corresponding unique ID strings.
        """
        profiles = self.store.get_all()
        return {p.name: p.id for p in profiles}

    def load_voice_from_library(self, track_id: str, voice_id: str):
        """
        Loads a persisted voice profile from the store into the specified track engine.

        This method retrieves the profile, rehydrates the inference engine with the 
        stored identity vector, and updates the session's track metadata to reflect 
        that the track is now using a pre-existing asset.

        Args:
            track_id (str): The target track identifier ("A" or "B").
            voice_id (str): The unique identifier of the profile in the store.

        Raises:
            ValueError: If the voice_id does not exist in the database.
        """
        profile = self.store.get_profile(voice_id)
        if not profile:
            raise ValueError(f"Voice profile {voice_id} not found.")

        engine = self.engine_a if track_id == "A" else self.engine_b
        
        # We use the engine's internal method to transform the stored List[float] 
        # back into a VoiceClonePromptItem, passing all required metadata.
        engine.load_identity_from_state(
            vector=profile.identity_embedding,
            seed=profile.seed,
            anchor_path=profile.anchor_audio_path
        )
        
        # IMPORTANT: Maintaining your original logic for TrackType tracking
        if track_id == "A":
            self.track_a_type = TrackType.PREEXISTING
        else:
            self.track_b_type = TrackType.PREEXISTING
            
        self._cached_blend_engine = None
        logger.info(f"Voice '{profile.name}' loaded into Track {track_id} (Type: PREEXISTING)")

    def preview_voice(self, text: str, blend_alpha: Optional[float] = None) -> str:
        """
        Synthesizes a preview audio clip using the current engine states.

        This method handles both single-track synthesis and complex blending.
        If a blend_alpha is provided (between 0.0 and 1.0 exclusive), it invokes
        the VoiceBlender to create an interpolated identity before synthesis.
        It uses lazy evaluation to avoid re-calculating the blend if the alpha
        hasn't changed.

        Args:
            text (str): The text content to synthesize into speech.
            blend_alpha (Optional[float]): The mixing ratio between Track A (0.0) and
                                           Track B (1.0). If None, defaults to Track A.

        Returns:
            str: The absolute path to the generated preview WAV file.

        Raises:
            ValueError: If the target engine (or blended engine) has no active identity.
        """
        target_engine = None
        if blend_alpha is None or blend_alpha == 0.0:
            target_engine = self.engine_a
        elif blend_alpha == 1.0:
            target_engine = self.engine_b
        else:
            # Just-In-Time Blending Logic
            if self._cached_blend_engine is None or abs(self._last_blend_alpha - blend_alpha) > 0.001:
                self._cached_blend_engine = VoiceBlender.blend(self.engine_a, self.engine_b, blend_alpha)
                self._last_blend_alpha = blend_alpha
                if self._cached_blend_engine.last_anchor_path:
                    self.register_temp_file(self._cached_blend_engine.last_anchor_path)
            target_engine = self._cached_blend_engine
        
        if not target_engine.active_identity:
             raise ValueError("Inference failed: Engine has no active identity to synthesize.")

        out_path = target_engine.render(text)
        self.register_temp_file(out_path)
        return out_path

    def save_session_voice(self, name: str, description: str, tags: List[str], metadata: Dict[str, Any]):
        """
        Materializes the current session state into a permanent VoiceProfile.

        This orchestration method gathers all necessary data to persist a voice:
        1. It resolves the final identity vector (handling JIT blending if necessary).
        2. It generates a semantic embedding using the EmbeddingEngine for future searchability.
        3. It constructs the VoiceProfile object with correct metadata.
        4. It delegates the physical storage to the VoiceStore.

        Args:
            name (str): The display name for the new voice profile.
            description (str): A human-readable description for UI visibility.
            tags (List[str]): A list of string tags for classification.
            metadata (Dict[str, Any]): Additional context, specifically containing the
                                       'semantic_index' text and 'source_type'.

        Raises:
            ValueError: If the engine being saved has no active identity.
        """
        alpha = metadata.get("blend")
        engine_to_save = None
        
        # Resolve the specific engine instance to save (A, B, or a Blend)
        if alpha is not None and 0.0 < float(alpha) < 1.0:
            current_alpha = float(alpha)
            if self._cached_blend_engine is None or abs(self._last_blend_alpha - current_alpha) > 0.001:
                self._cached_blend_engine = VoiceBlender.blend(self.engine_a, self.engine_b, current_alpha)
                self._last_blend_alpha = current_alpha
            engine_to_save = self._cached_blend_engine
        else:
            engine_to_save = self.engine_b if (alpha is not None and float(alpha) == 1.0) else self.engine_a

        if not engine_to_save or not engine_to_save.active_identity:
            raise ValueError("Save failed: The target engine has no active identity to save.")

        # Generate the semantic vector for search capability
        semantic_text = metadata.get("semantic_index", f"{name}. {description}")
        semantic_vector = self.embed_engine.generate_embedding(semantic_text)

        # Construct the persistence model
        profile = VoiceProfile(
            id=str(uuid.uuid4()),
            name=name,
            description=description, 
            identity_embedding=engine_to_save.get_identity_vector(),
            semantic_embedding=semantic_vector,
            seed=engine_to_save.active_seed,
            language=engine_to_save.lang,
            source_type=metadata.get("source_type", SourceType.DESIGN),
            tags=tags
        )
        
        self.store.add_profile(profile, anchor_source_path=engine_to_save.last_anchor_path)
        self.reset_workflow()

    def reset_workflow(self):
        """
        Resets the session to a clean state.

        This method performs a soft reset by:
        1. Deleting all registered temporary files from the disk.
        2. Clearing the file registry.
        3. Nullifying the active identities in the inference engines.
        4. Invoking garbage collection to free up memory.
        """
        gc.collect()
        for path in self._temp_files_registry:
            try:
                if os.path.exists(path): os.remove(path)
            except Exception: pass
        self._temp_files_registry.clear()
        self._reset_engine_states()

    def purge_temp_storage(self):
        """
        Performs a hard reset of the temporary storage system.

        Unlike reset_workflow, this method aggressively deletes everything inside
        the configured temporary directory, regardless of whether it is registered
        in the current session. Use with caution.
        """
        temp_dir = os.path.abspath(self.config.paths.temp_dir)
        if os.path.exists(temp_dir):
            for item in os.listdir(temp_dir):
                item_path = os.path.join(temp_dir, item)
                try:
                    if os.path.isfile(item_path): os.unlink(item_path)
                    elif os.path.isdir(item_path): shutil.rmtree(item_path)
                except Exception: pass
        self.reset_workflow()

    def _reset_engine_states(self):
        """
        Internal helper to clear the identity states of the inference engines.
        """
        self._cached_blend_engine = None
        self._last_blend_alpha = -1.0
        self.engine_a.active_identity = None
        self.engine_b.active_identity = None
        
    def export_voice(self, voice_id: str) -> bytes:
        """
        Retrieves a voice profile and its associated assets to create a Resona Bundle.
        
        It fetches the profile from the store, locates the physical anchor audio file,
        and uses the VoiceBundleIO service to generate a portable binary package.
        """
        from src.backend.io import VoiceBundleIO
        
        profile = self.store.get_profile(voice_id)
        if not profile:
            raise ValueError(f"Voice profile {voice_id} not found.")

        # Resolve the absolute path of the anchor audio
        anchor_path = os.path.join(self.config.paths.assets_dir, profile.anchor_audio_path)
        if not os.path.exists(anchor_path):
            raise FileNotFoundError(f"Anchor audio file missing at {anchor_path}")

        with open(anchor_path, "rb") as f:
            anchor_bytes = f.read()

        return VoiceBundleIO.pack_bundle(
            profile=profile,
            identity_vector=profile.identity_embedding,
            anchor_audio_bytes=anchor_bytes
        )

    def import_voice(self, bundle_bytes: bytes):
        """
        Unpacks a Resona Bundle and integrates it into the local store.
        
        It extracts the metadata and audio, persists the audio file into the 
        assets directory, and registers the profile in BeaverDB.
        """
        from src.backend.io import VoiceBundleIO
        
        profile_dict, identity_vector, anchor_bytes = VoiceBundleIO.unpack_bundle(bundle_bytes)
        
        # Reconstruct the profile to ensure validation
        profile = VoiceProfile(**profile_dict)
        profile.identity_embedding = identity_vector
        
        # Save the audio file to the assets directory
        target_path = os.path.join(self.config.paths.assets_dir, profile.anchor_audio_path)
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        
        with open(target_path, "wb") as f:
            f.write(anchor_bytes)
            
        # Persist the profile in the store (BeaverDB)
        self.store.add_profile(profile)
        logger.info(f"Imported voice bundle: {profile.name} ({profile.id})")

    def delete_voice(self, voice_id: str):
        """
        Removes a voice profile from the system and synchronizes engine states.
        
        If the deleted voice is currently loaded in any inference engine, 
        the engine state is cleared to maintain system integrity.
        """
        # 1. Physical and DB deletion
        self.store.delete_profile(voice_id)
        
        # 2. State synchronization: Clear engines if they were using this voice
        # We perform a full engine reset if there's a risk of stale identities
        self._reset_engine_states()
        logger.info(f"Deleted voice {voice_id} and synchronized engine states.")
        
    def update_voice_metadata(self, voice_id: str, new_name: str, new_desc: str, new_tags: List[str]):
        """
        Updates the metadata of an existing voice and regenerates its semantic index.
        
        This method performs a critical synchronization:
        1. It updates the descriptive fields (Name, Description, Tags).
        2. It RE-CALCULATES the semantic embedding vector to ensure the voice 
           remains discoverable via the new terms.
        3. It persists the changes using an upsert strategy via the store.

        Args:
            voice_id (str): The unique UUID of the profile to update.
            new_name (str): The new display name.
            new_desc (str): The new description text.
            new_tags (List[str]): The new list of tags.

        Raises:
            ValueError: If the voice_id does not exist.
        """
        # 1. Fetch existing profile
        profile = self.store.get_profile(voice_id)
        if not profile:
            raise ValueError(f"Cannot update: Voice {voice_id} not found.")

        # 2. Update Scalar Fields
        # We modify the object in memory directly.
        profile.name = new_name
        profile.description = new_desc
        profile.tags = new_tags
        
        # 3. Regenerate Semantic Vector (Critical Step)
        # We construct a rich context string to improve search relevance.
        # Format: "Name: {name}. Desc: {desc}. Tags: {tags}"
        semantic_context = f"Name: {new_name}. Desc: {new_desc}. Tags: {', '.join(new_tags)}"
        
        # We use the embedding engine already initialized in the session
        new_semantic_vector = self.embed_engine.generate_embedding(semantic_context)
        profile.semantic_embedding = new_semantic_vector

        # 4. Persist (Upsert)
        # We reuse add_profile. By passing anchor_source_path=None, we tell the store
        # to ONLY update the metadata/vectors and leave the physical audio file untouched.
        self.store.add_profile(profile, anchor_source_path=None)
            
        logger.info(f"Metadata and semantic index updated for voice: {voice_id}")