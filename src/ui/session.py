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
    Main controller linking UI interactions to Backend Logic and Persistence.
    
    Manages the lifecycle of dual inference engines, temporary file registries, 
    and orchestrates the saving process by integrating semantic embedding 
    generation and the VoiceStore persistence layer.
    """

    def __init__(self, config: AppConfig):
        """
        Initializes engines and persistence layers.

        Args:
            config (AppConfig): Global application configuration.
        """
        self.config = config
        self.tts_provider = TTSModelProvider(config)
        self.engine_a = InferenceEngine(config, self.tts_provider)
        self.engine_b = InferenceEngine(config, self.tts_provider)
        
        # Core Persistence and Semantic Components
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
        Updates the target language for all active inference engines.

        Args:
            lang_code (str): The ISO language code (e.g., 'es', 'en').
        """
        if self.engine_a.lang != lang_code:
            self.engine_a.lang = lang_code
            self.engine_b.lang = lang_code
            self._cached_blend_engine = None

    def register_temp_file(self, path: str):
        """
        Registers a temporary file for future cleanup.

        Args:
            path (str): Absolute path to the file.
        """
        if path and path not in self._temp_files_registry:
            self._temp_files_registry.append(path)

    def design_voice(self, track_id: str, prompt: str, seed: Optional[int] = None):
        """
        Triggers Voice Design on the specific track.

        Args:
            track_id (str): 'A' or 'B'.
            prompt (str): Text description of the desired voice.
            seed (Optional[int]): Generation seed for determinism.
        """
        engine = self.engine_a if track_id == "A" else self.engine_b
        engine.design_identity(prompt, seed)
        if engine.last_anchor_path:
            self.register_temp_file(engine.last_anchor_path)
        self._cached_blend_engine = None

    def clone_voice(self, track_id: str, audio_path: str, transcript: str):
        """
        Extracts a voice identity from a reference audio file.

        Args:
            track_id (str): 'A' or 'B'.
            audio_path (str): Path to the reference WAV file.
            transcript (str): Text spoken in the reference audio.
        """
        engine = self.engine_a if track_id == "A" else self.engine_b
        engine.clone_identity(audio_path, transcript)
        if engine.last_anchor_path:
            self.register_temp_file(engine.last_anchor_path)
        self._cached_blend_engine = None

    def preview_voice(self, text: str, blend_alpha: Optional[float] = None) -> str:
        """
        Generates audio for a preview phrase using lazy evaluation for blending.

        Args:
            text (str): Phrase to synthesize.
            blend_alpha (Optional[float]): The mix ratio (0.0 to 1.0).

        Returns:
            str: Absolute path to the generated WAV file.
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
            raise ValueError("Inference failed: Engine has no active identity.")
        
        output_path = target_engine.render(text)
        self.register_temp_file(output_path)
        return output_path

    def save_session_voice(self, name: str, description: str, tags: List[str], metadata: Dict[str, Any]):
        """
        Materializes the voice profile by ensuring identity generation and
        persisting via VoiceStore with semantic indexing.

        Args:
            name (str): Voice profile name.
            description (str): Visible description for the user.
            tags (List[str]): List of classification tags.
            metadata (Dict[str, Any]): Metadata including the semantic_index text.
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
            raise ValueError("Save failed: Target engine lacks a valid identity.")

        # JIT Semantic Indexing
        semantic_text = metadata.get("semantic_index", f"{name}. {description}")
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
            tags=tags
        )
        
        self.store.add_profile(profile, anchor_source_path=engine_to_save.last_anchor_path)
        self.reset_workflow()

    def reset_workflow(self):
        """Soft reset of the current session: cleans files and identities."""
        gc.collect()
        for path in self._temp_files_registry:
            try:
                if os.path.exists(path): os.remove(path)
            except Exception: pass
        self._temp_files_registry.clear()
        self._reset_engine_states()

    def purge_temp_storage(self):
        """Hard reset: Wipes the entire temporary directory."""
        temp_dir = os.path.abspath(self.config.paths.temp_dir)
        if os.path.exists(temp_dir):
            for item in os.listdir(temp_dir):
                item_path = os.path.join(temp_dir, item)
                try:
                    if os.path.isfile(item_path): os.unlink(item_path)
                    elif os.path.isdir(item_path): shutil.rmtree(item_path)
                except Exception: pass
        self.reset_workflow()