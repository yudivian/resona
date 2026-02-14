import os
import uuid
import time
import logging
import shutil
import gc
from typing import Optional, List, Dict, Any
from src.models import AppConfig, VoiceProfile, SourceType, TrackType
from src.backend.engine import TTSModelProvider, InferenceEngine, VoiceBlender

logger = logging.getLogger(__name__)

class SessionManager:
    # ... (init and registration methods remain unchanged)

    def save_session_voice(self, name: str, description: str, tags: List[str], metadata: Dict[str, Any]):
        """
        Orchestrates persistence and ensures the profile is saved to the database.
        """
        alpha = metadata.get("blend")
        engine_to_save = None
        
        # Determine Target Engine
        if alpha is not None and 0.0 < float(alpha) < 1.0:
            current_alpha = float(alpha)
            if self._cached_blend_engine is None or abs(self._last_blend_alpha - current_alpha) > 0.001:
                self._cached_blend_engine = VoiceBlender.blend(self.engine_a, self.engine_b, current_alpha)
                self._last_blend_alpha = current_alpha
            engine_to_save = self._cached_blend_engine
        else:
            engine_to_save = self.engine_b if (alpha is not None and float(alpha) == 1.0) else self.engine_a

        if not engine_to_save or not engine_to_save.active_identity:
            raise ValueError("No active identity found for saving.")

        # Persist Audio Anchor
        profile_id = str(uuid.uuid4())
        saved_anchor_filename = f"{profile_id}.wav"
        dest_path = os.path.join(self.config.paths.assets_dir, saved_anchor_filename)
        
        if engine_to_save.last_anchor_path and os.path.exists(engine_to_save.last_anchor_path):
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            shutil.copy(engine_to_save.last_anchor_path, dest_path)
            if engine_to_save.last_anchor_path in self._temp_files_registry:
                self._temp_files_registry.remove(engine_to_save.last_anchor_path)

        # Create Profile Object
        profile = VoiceProfile(
            id=profile_id,
            name=name,
            description=description, 
            identity_embedding=engine_to_save.get_identity_vector(),
            seed=engine_to_save.active_seed,
            language=engine_to_save.lang,
            source_type=metadata.get("source_type", SourceType.DESIGN),
            track_a_type=self.track_a_type,
            track_b_type=self.track_b_type,
            tags=tags,
            anchor_audio_path=saved_anchor_filename
        )
        
        # PERSIST TO DATABASE
        # Assuming the tts_provider or a dedicated db_manager handles persistence
        # This is the step that was previously missing.
        self.tts_provider.db.save_profile(profile, semantic_index=metadata.get("semantic_index"))
        
        logger.info(f"VoiceProfile '{name}' persisted to database with ID: {profile_id}")
        
        # Clean session state
        self.reset_workflow()