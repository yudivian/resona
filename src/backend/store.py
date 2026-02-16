import os
import shutil
import time
import logging
from typing import List, Optional
from beaver import BeaverDB, Document
from src.models import VoiceProfile, AppConfig

logger = logging.getLogger(__name__)

class VoiceStore:
    """
    Unified persistence layer using BeaverDB with native Vector Support.
    
    This class manages the master metadata dictionary and specialized 
    vector collections for identity and semantic search, complying with 
    BeaverDB's indexing API.
    """

    def __init__(self, config: AppConfig):
        """
        Initializes the store and ensures directory integrity.

        Args:
            config (AppConfig): Global configuration for paths and DB settings.
        """
        self.config = config
        # Ensure database directory exists
        os.makedirs(os.path.dirname(config.paths.db_file), exist_ok=True)
        # Ensure assets directory exists for audio anchors
        os.makedirs(config.paths.assets_dir, exist_ok=True)

        self.db = BeaverDB(config.paths.db_file)
        # Master dictionary for full JSON profile data
        self.master_data = self.db.dict("voices_data")
        # Vector collections
        self.identity_index = self.db.collection("idx_identity")
        self.semantic_index = self.db.collection("idx_semantic")

    def add_profile(self, profile: VoiceProfile, anchor_source_path: Optional[str] = None) -> str:
        """
        Persists the voice profile and indexes its vectors using BeaverDB's index API.

        Args:
            profile (VoiceProfile): The Pydantic model containing voice data.
            anchor_source_path (Optional[str]): Path to the source WAV file to be archived.

        Returns:
            str: The unique ID of the persisted profile.
        """
        # 1. Physical Asset Management
        if anchor_source_path and os.path.exists(anchor_source_path):
            safe_name = profile.name.replace(' ', '_').lower()
            filename = f"{safe_name}_{int(time.time())}_anchor.wav"
            destination = os.path.join(self.config.paths.assets_dir, filename)
            shutil.copy2(anchor_source_path, destination)
            profile.anchor_audio_path = filename

        # 2. Metadata Persistence
        self.master_data[profile.id] = profile.model_dump()
        
        # 3. Identity Vector Indexing
        # FIX: Using .index() instead of .add() and 'text' instead of 'content'
        doc_identity = Document(
            id=profile.id,
            embedding=profile.identity_embedding,
            text=profile.name,
            metadata={"type": "identity"}
        )
        self.identity_index.index(doc_identity)
        
        # 4. Semantic Vector Indexing
        if profile.semantic_embedding:
            doc_semantic = Document(
                id=profile.id,
                embedding=profile.semantic_embedding,
                text=profile.description or "No description",
                metadata={"type": "semantic"}
            )
            self.semantic_index.index(doc_semantic)
            
        logger.info(f"Profile {profile.id} persisted and indexed successfully.")
        return profile.id

    def get_all(self) -> List[VoiceProfile]:
        """
        Retrieves all stored voice profiles from the master data dictionary.

        Returns:
            List[VoiceProfile]: A list of instantiated VoiceProfile objects.
        """
        return [VoiceProfile(**data) for data in self.master_data.values()]