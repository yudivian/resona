import os
import shutil
import time
from typing import List, Optional
from beaver import BeaverDB, Document
from src.models import VoiceProfile, AppConfig

class VoiceStore:
    """
    Unified persistence layer using BeaverDB with native Vector Support and physical asset management.
    """
    def __init__(self, config: AppConfig):
        """
        Initializes the store and ensures all required directories exist.
        """
        self.config = config
        
        db_dir = os.path.dirname(config.paths.db_file)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
            
        os.makedirs(config.paths.assets_dir, exist_ok=True)
        os.makedirs(config.paths.temp_dir, exist_ok=True)

        self.db = BeaverDB(config.paths.db_file)
        self.master_data = self.db.dict("voices_data")
        self.identity_index = self.db.collection("idx_identity")
        self.semantic_index = self.db.collection("idx_semantic")

    def add_profile(self, profile: VoiceProfile, anchor_source_path: Optional[str] = None) -> str:
        """
        Persists the voice profile and copies the physical anchor audio to the assets directory.
        """
        if anchor_source_path and os.path.exists(anchor_source_path):
            safe_name = profile.name.replace(' ', '_').lower()
            timestamp = int(time.time())
            filename = f"{safe_name}_{timestamp}_anchor.wav"
            
            destination = os.path.join(self.config.paths.assets_dir, filename)
            shutil.copy2(anchor_source_path, destination)
            
            profile.anchor_audio_path = filename

        self.master_data[profile.id] = profile.model_dump()
        
        doc_identity = Document(
            id=profile.id,
            embedding=profile.identity_embedding,
            content=profile.name,
            metadata={"type": "identity"}
        )
        self.identity_index.add(doc_identity)
        
        if profile.semantic_embedding:
            doc_semantic = Document(
                id=profile.id,
                embedding=profile.semantic_embedding,
                content=profile.description or "",
                metadata={"type": "semantic"}
            )
            self.semantic_index.add(doc_semantic)
            
        return profile.id

    def get(self, voice_id: str) -> Optional[VoiceProfile]:
        """
        Retrieves a voice profile by its ID and instantiates the Pydantic model.
        """
        try:
            data = self.master_data.get(voice_id)
            if data:
                return VoiceProfile(**data)
            return None
        except KeyError:
            return None

    def get_all(self) -> List[VoiceProfile]:
        """
        Retrieves all stored voice profiles from the master data dictionary.
        """
        return [VoiceProfile(**data) for data in self.master_data.values()]