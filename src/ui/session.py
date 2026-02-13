import os
import uuid
import time
import logging
import shutil
from typing import Optional, List, Dict, Any
from src.models import AppConfig, VoiceProfile, SourceType, TrackType
from src.backend.engine import TTSModelProvider, InferenceEngine, VoiceBlender

logger = logging.getLogger(__name__)

class SessionManager:
    """
    Main controller linking UI interactions to Backend Logic.
    """

    def __init__(self, config: AppConfig):
        """
        Initializes the session, provider, and engine tracks.

        Args:
            config (AppConfig): Application configuration.
        """
        self.config = config
        self.tts_provider = TTSModelProvider(config)
        self.engine_a = InferenceEngine(config, self.tts_provider)
        self.engine_b = InferenceEngine(config, self.tts_provider)
        self._cached_blend_engine: Optional[InferenceEngine] = None
        self._last_blend_alpha: float = -1.0
        self.track_a_type: TrackType = TrackType.DESIGN
        self.track_b_type: TrackType = TrackType.DESIGN

    def set_language(self, lang_code: str):
        """
        Updates language context for both engines.

        Args:
            lang_code (str): The ISO 639-1 language code.
        """
        if self.engine_a.lang != lang_code:
            self.engine_a.lang = lang_code
            self.engine_b.lang = lang_code
            self._cached_blend_engine = None

    def _get_target_engine(self, track_id: str) -> InferenceEngine:
        if track_id == "A": return self.engine_a
        elif track_id == "B": return self.engine_b
        else: raise ValueError(f"Invalid track_id: {track_id}")

    def design_voice(self, track_id: str, prompt: str, seed: Optional[int] = None):
        """
        Triggers Voice Design on the specific track.

        Args:
            track_id (str): 'A' or 'B'.
            prompt (str): Design description.
            seed (Optional[int]): Seed for reproducibility.
        """
        engine = self._get_target_engine(track_id)
        engine.design_identity(prompt, seed)
        self._cached_blend_engine = None
        if track_id == "A":
            self.track_a_type = TrackType.DESIGN
        else:
            self.track_b_type = TrackType.DESIGN

    def clone_voice(self, track_id: str, audio_path: str, transcript: str):
        """
        Triggers Voice Cloning on the specific track.

        Args:
            track_id (str): 'A' or 'B'.
            audio_path (str): Path to source audio.
            transcript (str): Reference text for alignment.
        """
        engine = self._get_target_engine(track_id)
        engine.extract_identity(audio_path, transcript)
        self._cached_blend_engine = None
        if track_id == "A":
            self.track_a_type = TrackType.CLONE
        else:
            self.track_b_type = TrackType.CLONE

    def load_voice_from_library(self, track_id: str, voice_id: str):
        """
        Loads a voice profile from the database into the track.

        Args:
            track_id (str): 'A' or 'B'.
            voice_id (str): Unique ID of the voice in DB.
        """
        target_engine = self._get_target_engine(track_id)
        # Mocking DB load for now as per instructions
        # Real implementation would fetch VoiceProfile and call:
        # target_engine.load_identity_from_state(...)
        if track_id == "A":
            self.track_a_type = TrackType.PREEXISTING
        else:
            self.track_b_type = TrackType.PREEXISTING

    def preview_voice(self, text: str, blend_alpha: Optional[float] = None) -> str:
        """
        Generates audio based on current state.

        Args:
            text (str): Text to synthesize.
            blend_alpha (Optional[float]): Mixing factor.

        Returns:
            str: Path to the generated audio file.
        """
        target_engine = None

        if blend_alpha is None or blend_alpha == 0.0:
            target_engine = self.engine_a
        elif blend_alpha == 1.0:
            target_engine = self.engine_b
        else:
            if self._cached_blend_engine is None or self._last_blend_alpha != blend_alpha:
                self._cached_blend_engine = VoiceBlender.blend(
                    self.engine_a, self.engine_b, blend_alpha
                )
                self._last_blend_alpha = blend_alpha
            target_engine = self._cached_blend_engine

        if not target_engine.active_identity:
            raise ValueError("Target track has no active identity.")

        return target_engine.render(text)

    def get_current_seed(self, blend_alpha: Optional[float] = None) -> int:
        """
        Retrieves the active seed for metadata storage.

        Args:
            blend_alpha (Optional[float]): Mixing factor.

        Returns:
            int: The active seed.
        """
        if blend_alpha is not None and 0.0 < blend_alpha < 1.0:
             if self._cached_blend_engine:
                 return self._cached_blend_engine.active_seed
             return 0
        
        if blend_alpha == 1.0:
            return self.engine_b.active_seed or 0
        
        return self.engine_a.active_seed or 0

    def list_library_voices(self) -> Dict[str, str]:
        """
        Returns a dict of {id: name} for the UI dropdown.

        Returns:
            Dict[str, str]: Dictionary of voices.
        """
        return {"demo_01": "Demo Voice A", "demo_02": "Demo Voice B"}

    def save_session_voice(self, name: str, description: str, tags: List[str], metadata: Dict[str, Any]):
        """
        Saves the current voice configuration to the library.

        Args:
            name (str): Name of the voice.
            description (str): Description text.
            tags (List[str]): List of tags.
            metadata (Dict): Contains seed, blend settings, etc.
        """
        alpha = metadata.get("blend")
        engine = None
        
        if alpha is not None and 0.0 < float(alpha) < 1.0:
            engine = self._cached_blend_engine
        elif alpha is not None and float(alpha) == 1.0:
            engine = self.engine_b
        else:
            engine = self.engine_a

        if not engine or not engine.active_identity:
            raise ValueError("No active identity to save.")

        if hasattr(engine, 'get_identity_vector'):
            vector = engine.get_identity_vector()
        else:
            vector = engine.active_identity.ref_spk_embedding.squeeze().cpu().tolist()
        
        saved_anchor_filename = None
        if engine.last_anchor_path and os.path.exists(engine.last_anchor_path):
            unique_id = str(uuid.uuid4())
            safe_name = "".join([c for c in name if c.isalnum() or c in (' ', '-', '_')]).strip().replace(' ', '_')
            saved_anchor_filename = f"{safe_name}_{unique_id}.wav"
            dest_path = os.path.join(self.config.paths.assets_dir, saved_anchor_filename)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            shutil.copy(engine.last_anchor_path, dest_path)

        profile = VoiceProfile(
            name=name,
            description=description,
            identity_embedding=vector,
            seed=metadata.get("seed", 0),
            language=metadata.get("language", "en"),
            source_type=metadata.get("source_type", SourceType.DESIGN),
            track_a_type=self.track_a_type,
            track_b_type=self.track_b_type,
            tags=tags,
            anchor_audio_path=saved_anchor_filename
        )
        
        # Real persistence would happen here calling self.store.save(profile)
        logger.info(f"VoiceProfile created: {profile.name}")