import os
import uuid
import logging
import shutil
import gc
from typing import Optional, List, Dict, Any
from src.models import AppConfig, VoiceProfile, SourceType, TrackType
from src.backend.engine import TTSModelProvider, InferenceEngine, VoiceBlender, CALIBRATION_TEXTS
from src.backend.embedding import EmbeddingModelProvider, EmbeddingEngine
from src.backend.store import VoiceStore
from src.backend.io import VoiceBundleIO


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
        
        engine.load_identity_from_state(
            vector=profile.identity_embedding,
            seed=profile.seed,
            anchor_path=profile.anchor_audio_path
        )
        
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
        Persists the current voice state as a permanent VoiceProfile.

        This method orchestrates the transition from a temporary session state to a 
        stored library item. It handles the logic for both single-track and blended 
        identities. Crucially, it extracts the original design prompts from the 
        metadata payload and stores them in the `source_prompts` field of the 
        profile, ensuring the creative DNA of the voice is preserved. It then 
        constructs a rich semantic index using these prompts, the description, and 
        tags to enable advanced natural language search.

        Args:
            name (str): The display name for the new library entry.
            description (str): A detailed user-provided description.
            tags (List[str]): Categorization labels.
            metadata (Dict[str, Any]): A dictionary containing session context, 
                including blend parameters and raw design prompts.

        Raises:
            ValueError: If the target engine lacks a valid active identity to save.
        """
        alpha = metadata.get("blend")
        engine_to_save = None
        
        if alpha is not None and 0.0 < float(alpha) < 1.0:
            current_alpha = float(alpha)
            if self._cached_blend_engine is None or abs(self._last_blend_alpha - current_alpha) > 0.001:
                self._cached_blend_engine = VoiceBlender.blend(self.engine_a, self.engine_b, current_alpha)
                self._last_blend_alpha = current_alpha
            engine_to_save = self._cached_blend_engine
        else:
            engine_to_save = self.engine_b if (alpha is not None and float(alpha) == 1.0) else self.engine_a

        if not engine_to_save or not engine_to_save.active_identity:
            raise ValueError("Save failed: No active identity detected in the target engine.")

        source_prompts = {}
        if "design_prompt_A" in metadata:
            source_prompts["A"] = metadata["design_prompt_A"]
        if "design_prompt_B" in metadata:
            source_prompts["B"] = metadata["design_prompt_B"]
            
        prompt_context_list = [f"Track {k}: {v}" for k, v in source_prompts.items()]
        combined_prompt_text = " | ".join(prompt_context_list)

        semantic_text = self._build_semantic_context(
            name=name,
            lang=engine_to_save.lang,
            tags=tags,
            desc=description,
            prompt_context=combined_prompt_text
        )
        semantic_vector = self.embed_engine.generate_embedding(semantic_text)
        
        profile = VoiceProfile(
            id=str(uuid.uuid4()),
            name=name,
            description=description, 
            identity_embedding=engine_to_save.get_identity_vector(),
            semantic_embedding=semantic_vector,
            seed=engine_to_save.active_seed,
            language=engine_to_save.lang,
            source_type=metadata.get("source_type", SourceType.DESIGN),
            tags=tags,
            source_prompts=source_prompts, 
            refinement_prompt=combined_prompt_text,
            anchor_text = CALIBRATION_TEXTS.get(engine_to_save.lang, CALIBRATION_TEXTS["en"])
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
        profile = self.store.get_profile(voice_id)
        if not profile:
            raise ValueError(f"Voice profile {voice_id} not found.")

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
        Imports a serialized voice bundle (.rnb) and integrates it into the local ecosystem.

        This method implements a "Hybrid Trust" strategy to ensure both fidelity to the 
        original voice and compatibility with the local search environment.

        Architecture Decisions:
        1.  **Acoustic Trust (Identity):** We blindly accept the `identity_embedding` 
            contained in the bundle. This is the "DNA" of the voice. Re-calculating it 
            locally is avoided because:
            a) It preserves the exact acoustic signature defined by the creator.
            b) It avoids potential floating-point deviations or model version mismatches.
            c) It saves significant computational resources (no heavy inference).

        2.  **Semantic Regeneration (Search):** We strictly regenerate the `semantic_embedding`.
            The vector space for text search depends entirely on the local embedding model 
            loaded in the current session. Using a vector from a different machine/model 
            would make the voice "invisible" or incorrectly ranked in text searches.

        3.  **Deterministic Storage (Idempotency):** We ignore the original filename from 
            the bundle and enforce a naming convention based on the Voice UUID 
            (`{name}_{uuid}_anchor.wav`). 
            - If the user imports the same voice twice, it overwrites the existing file 
              cleanly rather than creating duplicate files with timestamp suffixes.
            - This ensures 1:1 mapping between a logical voice profile and its physical asset.

        Args:
            bundle_bytes (bytes): The raw binary content of the .rnb file (zip archive).
        """
        from src.backend.io import VoiceBundleIO
        
        # 1. Unpack the bundle components
        # We extract the metadata dictionary, the pre-calculated identity vector, and the raw audio bytes.
        profile_dict, identity_vector, anchor_bytes = VoiceBundleIO.unpack_bundle(bundle_bytes)
        
        # 2. Rehydrate the VoiceProfile
        # We instantiate the model and inject the original acoustic identity immediately.
        profile = VoiceProfile(**profile_dict)
        profile.identity_embedding = identity_vector
        
        # 3. Deterministic Asset Storage
        # We construct a filename using the UUID to guarantee uniqueness and idempotency.
        safe_name = profile.name.replace(' ', '_').lower()
        new_filename = f"{safe_name}_{profile.id}_anchor.wav"
        
        target_path = os.path.join(self.config.paths.assets_dir, new_filename)
        
        # Ensure the assets directory exists
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        
        # Write the audio file directly to disk.
        # By using 'wb', we overwrite any existing file with the same ID, preventing garbage accumulation.
        with open(target_path, "wb") as f:
            f.write(anchor_bytes)
            
        # Update the profile's reference to point to this new local file.
        profile.anchor_audio_path = new_filename
        
        # 4. Regenerate Semantic Search Vector
        # We rebuild the text context (name + description + prompts) and feed it to the 
        # LOCAL embedding engine to ensure this voice appears correctly in local text searches.
        prompt_context_list = [f"Track {k}: {v}" for k, v in profile.source_prompts.items()]
        combined_prompt = " | ".join(prompt_context_list)
        
        semantic_text = self._build_semantic_context(
            name=profile.name,
            lang=profile.language,
            tags=profile.tags,
            desc=profile.description,
            prompt_context=combined_prompt
        )
        
        # Generate the vector if the engine is available (lightweight CPU/GPU operation)
        if self.embed_engine:
            profile.semantic_embedding = self.embed_engine.generate_embedding(semantic_text)
        
        # 5. Persist to Database
        # We call add_profile without passing 'anchor_source_path', as we have already 
        # manually handled the file storage in step 3. The store will only handle indexing.
        self.store.add_profile(profile)
        
        logger.info(f"Imported voice '{profile.name}' ({profile.id}). Audio preserved, Semantic index regenerated.")


    def delete_voice(self, voice_id: str):
        """
        Removes a voice profile from the system and synchronizes engine states.
        
        If the deleted voice is currently loaded in any inference engine, 
        the engine state is cleared to maintain system integrity.
        """

        self.store.delete_profile(voice_id)
        
        self._reset_engine_states()
        logger.info(f"Deleted voice {voice_id} and synchronized engine states.")
        
    def update_voice_metadata(self, voice_id: str, new_name: str, new_desc: str, new_tags: List[str]):
        """
        Modifies the descriptive metadata of an existing voice and refreshes its semantic index.

        This method updates the mutable fields of a VoiceProfile (name, description, tags). 
        Crucially, it recalculates the semantic embedding to reflect these changes while 
        preserving the immutable acoustic context. It retrieves the original design prompts 
        and language from the persisted profile to ensure the new vector maintains the 
        full creative history of the voice, allowing it to remain searchable by its 
        origin story even after being renamed.

        Args:
            voice_id (str): The unique identifier of the profile to update.
            new_name (str): The updated display name.
            new_desc (str): The updated description text.
            new_tags (List[str]): The updated collection of tags.

        Raises:
            ValueError: If the specified voice_id is not found in the storage layer.
        """
        profile = self.store.get_profile(voice_id)
        if not profile:
            raise ValueError(f"Update failed: Voice profile {voice_id} not found.")

        profile.name = new_name
        profile.description = new_desc
        profile.tags = new_tags

        prompt_context_list = [f"Track {k}: {v}" for k, v in profile.source_prompts.items()]
        combined_prompt_text = " | ".join(prompt_context_list)

        semantic_text = self._build_semantic_context(
            name=new_name,
            lang=profile.language,
            tags=new_tags,
            desc=new_desc,
            prompt_context=combined_prompt_text
        )

        new_semantic_vector = self.embed_engine.generate_embedding(semantic_text)
        profile.semantic_embedding = new_semantic_vector

        self.store.add_profile(profile, anchor_source_path=None)
        
    def _build_semantic_context(self, name: str, lang: str, tags: List[str], desc: str, prompt_context: str) -> str:
        """
        Constructs a comprehensive textual representation for semantic indexing.

        This internal utility aggregates all descriptive metadata and design 
        prompts into a single string. This "semantic document" is used by the 
        embedding engine to create a vector that represents the voice's 
        identity and creative origin, enabling high-quality semantic retrieval.

        Args:
            name (str): The display name assigned to the voice profile.
            lang (str): The primary language code associated with the voice.
            tags (List[str]): User-defined labels for categorization.
            desc (str): A detailed description of the voice's characteristics.
            prompt_context (str): The raw design prompts used during creation.

        Returns:
            str: A formatted string containing the full context for vectorization.
        """
        safe_desc = desc or ""
        safe_prompt = prompt_context or ""
        tags_str = ", ".join(tags) if tags else ""

        components = [
            f"Voice Name: {name}",
            f"Language: {lang}",
            f"Tags: {tags_str}",
            f"Description: {safe_desc}"
        ]
        
        if safe_prompt:
            components.append(f"Origin Context: {safe_prompt}")
            
        return ". ".join(components)
    
    def search_by_text(self, query: str, limit: int = 20) -> List[VoiceProfile]:
        """
        Executes a semantic search against the voice library using a natural language query.

        This method converts the user's text description into a semantic vector
        using the embedding engine and queries the storage layer for conceptually
        similar voice profiles.

        Args:
            query (str): The search text describing the desired voice.
            limit (int): The maximum number of results to return.

        Returns:
            List[VoiceProfile]: A list of matching profiles sorted by relevance.
        """
        if not query or not query.strip():
            return []

        vector = self.embed_engine.generate_embedding(query)
        return self.store.search_semantic(vector, limit)
    
    