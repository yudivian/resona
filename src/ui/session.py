import uuid
from typing import Optional, List, Dict
from src.models import AppConfig, VoiceProfile, SourceType, TrackType
from src.backend.engine import TTSModelProvider, InferenceEngine, VoiceBlender
from src.backend.embedding import EmbeddingModelProvider, EmbeddingEngine
from src.backend.store import VoiceStore

class SessionManager:
    """
    Manages the design session state, keeping inference engines and persistence layer in sync with the UI.
    """
    def __init__(self, config: AppConfig):
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
        Loads a persisted profile from the store into the specified track engine.
        """
        profile = self.store.get(profile_id)
        if not profile:
            raise ValueError(f"Profile {profile_id} not found")
        
        engine = self.engine_a if track_label.upper() == "A" else self.engine_b
        
        engine.lang = profile.language
        engine.set_identity(profile.identity_embedding, seed=profile.seed)

    def create_design_on_track(self, track_label: str, prompt: str, language: str):
        """
        Generates a new identity from a text prompt on the specified track.
        """
        engine = self.engine_a if track_label.upper() == "A" else self.engine_b
        engine.lang = language
        engine.design_identity(prompt)

    def blend_tracks(self, alpha: float) -> str:
        """
        Interprets the current state of both tracks, blends them, and generates a preview.
        """
        if self.engine_a.active_identity is None or self.engine_b.active_identity is None:
            raise ValueError("Tracks A and B must be initialized before blending")
            
        self.active_mix = VoiceBlender.blend(self.engine_a, self.engine_b, alpha)
        
        self.active_seed = self.engine_a.seed 
        
        self.engine_a.set_identity(self.active_mix, seed=self.active_seed)
        return self.engine_a.generate_preview(text="Previewing the blended voice.")

    def refine_current_mix(self, instruction: str, preview_text: str) -> str:
        """
        Applies a semantic refinement instruction to the currently active mix.
        """
        if self.active_mix is None:
            raise ValueError("No active mix available for refinement")
            
        return self.engine_a.generate_preview(text=preview_text, refinement=instruction)

    def save_session_result(self, name: str, description: str, tags: List[str], metadata: Dict) -> VoiceProfile:
        """
        Finalizes the session by generating semantic embeddings, rendering the anchor audio, and persisting the profile.
        """
        if self.active_mix is None:
            raise ValueError("Cannot save an empty session")

        semantic_vector = self.embed_engine.generate_embedding(description)
        
        safe_name = "".join([c for c in name if c.isalnum() or c in (' ', '-', '_')]).strip().replace(' ', '_')
        anchor_path = self.engine_a.render_anchor(
            calibration_script="The quick brown fox jumps over the lazy dog.", 
            asset_name=safe_name
        )
        
        profile = VoiceProfile(
            name=name,
            identity_embedding=self.active_mix,
            seed=self.active_seed,
            language=self.engine_a.lang,
            
            source_type=metadata.get("source_type", SourceType.BLEND),
            track_a_type=metadata.get("track_a_type"),
            track_b_type=metadata.get("track_b_type"),
            
            is_refined=metadata.get("is_refined", False),
            refinement_prompt=metadata.get("refinement_prompt"),
            
            description=description,
            semantic_embedding=semantic_vector,
            tags=tags,
            
            anchor_audio_path=anchor_path
        )
        
        self.store.save(profile)
        return profile