import uuid
import time
import os
from typing import Optional, List, Dict, Any
from src.models import AppConfig, VoiceProfile, SourceType, TrackType
from src.backend.engine import TTSModelProvider, InferenceEngine, VoiceBlender
from src.backend.embedding import EmbeddingModelProvider, EmbeddingEngine
from src.backend.store import VoiceStore

class SessionManager:
    """
    Orchestrates the design session state by synchronizing inference engines, 
    embedding generation, and the persistence layer.
    """
    def __init__(self, config: AppConfig):
        """
        Initializes the session with required providers and engines.

        Args:
            config (AppConfig): Global application configuration.
        """
        self.config = config
        
        self.tts_provider = TTSModelProvider(config)
        self.embed_provider = EmbeddingModelProvider(config)
        
        self.engine_a = InferenceEngine(config, self.tts_provider, lang="en")
        self.engine_b = InferenceEngine(config, self.tts_provider, lang="en")
        self.embed_engine = EmbeddingEngine(config, self.embed_provider)
        
        self.store = VoiceStore(config)
        
        self.active_mix: Optional[List[float]] = None
        self.active_seed: int = 42

    def load_voice_to_track(self, track_label: str, profile_id: str):
        """
        Loads a persisted profile from the store into the targeted track engine.

        Args:
            track_label (str): Identifier for the track ('A' or 'B').
            profile_id (str): Unique identifier of the voice profile.
        """
        profile = self.store.get(profile_id)
        if not profile:
            raise ValueError(f"Profile {profile_id} not found")
        
        target_engine = self.engine_a if track_label.upper() == "A" else self.engine_b
        target_engine.lang = profile.language
        
        if profile.anchor_audio_path:
            full_anchor_path = os.path.join(self.config.paths.assets_dir, profile.anchor_audio_path)
            if os.path.exists(full_anchor_path):
                target_engine.extract_identity(full_anchor_path)
                return

        target_engine.load_identity_from_vector(profile.identity_embedding)

    def blend_tracks(self, alpha: float):
        """
        Computes a blended identity vector between track A and track B.

        Args:
            alpha (float): Interpolation ratio between 0.0 and 1.0.
        """
        blended_engine = VoiceBlender.blend(self.engine_a, self.engine_b, alpha)
        self.active_mix = blended_engine.get_identity_vector()
        self.active_seed = 42

    def save_session_voice(self, name: str, description: str, tags: List[str], metadata: Dict[str, Any]):
        """
        Finalizes the current session by generating semantic embeddings, 
        rendering the anchor audio, and persisting the profile.

        Args:
            name (str): Display name for the new voice.
            description (str): Textual description for semantic search.
            tags (List[str]): Classification tags.
            metadata (Dict[str, Any]): Additional parameters and source info.
        """
        if self.active_mix is None:
            raise ValueError("No active identity mix to save")

        semantic_vector = self.embed_engine.generate_embedding(description)
        
        safe_name = "".join([c for c in name if c.isalnum() or c in (' ', '-', '_')]).strip().replace(' ', '_')
        
        temp_anchor_path = self.engine_a.render_anchor(asset_name=safe_name)
        
        profile = VoiceProfile(
            id=str(uuid.uuid4()),
            name=name,
            identity_embedding=self.active_mix,
            semantic_embedding=semantic_vector,
            description=description,
            tags=tags,
            seed=self.active_seed,
            language=self.engine_a.lang,
            source_type=metadata.get("source_type", SourceType.BLEND),
            track_a_type=metadata.get("track_a_type"),
            track_b_type=metadata.get("track_b_type"),
            is_refined=metadata.get("is_refined", False),
            refinement_prompt=metadata.get("refinement_prompt"),
            parameters=metadata.get("parameters", {}),
            created_at=time.time()
        )
        
        self.store.add_profile(profile, anchor_source_path=temp_anchor_path)