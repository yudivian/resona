import os
from typing import List, Optional
from beaver import BeaverDB, Document
from src.models import VoiceProfile, AppConfig

class VoiceStore:
    """
    Unified persistence layer using a relational-style pattern with BeaverDB.
    
    Architecture:
    - Master Store: 'voices_data' (Dict) - The source of truth for metadata.
    - Identity Index: 'idx_identity' (Docs) - Vector index for voice DNA.
    - Semantic Index: 'idx_semantic' (Docs) - Vector index for design concepts.
    """
    def __init__(self, config: AppConfig):
        self.config = config
        
        db_dir = os.path.dirname(config.paths.db_file)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
            
        os.makedirs(config.paths.assets_dir, exist_ok=True)
        os.makedirs(config.paths.temp_dir, exist_ok=True)

        self.db = BeaverDB(config.paths.db_file)
        
        self.master_data = self.db.dict("voices_data", model=VoiceProfile)
        self.identity_index = self.db.docs("idx_identity")
        self.semantic_index = self.db.docs("idx_semantic")

    def save(self, profile: VoiceProfile) -> str:
        """
        Saves the profile to the master dict and updates both vector indices.
        """
        self.master_data.set(profile.id, profile)
        
        identity_doc = Document(
            id=profile.id,
            embedding=profile.identity_embedding
        )
        self.identity_index.index(identity_doc)
        
        if profile.semantic_embedding:
            semantic_doc = Document(
                id=profile.id,
                embedding=profile.semantic_embedding
            )
            self.semantic_index.index(semantic_doc)
            
        return profile.id

    def get(self, voice_id: str) -> Optional[VoiceProfile]:
        """
        Retrieves the profile from the master dictionary.
        """
        try:
            return self.master_data.get(voice_id)
        except KeyError:
            return None

    def search_identity(self, query_vector: List[float], limit: int = 5) -> List[VoiceProfile]:
        """
        Performs vector search on the identity index and hydrates results from master.
        """
        results = self.identity_index.search(vector=query_vector, top_k=limit)
        
        profiles = []
        for doc, _ in results:
            profile = self.get(doc.id)
            if profile:
                profiles.append(profile)
        return profiles

    def search_semantic(self, query_vector: List[float], limit: int = 5) -> List[VoiceProfile]:
        """
        Performs vector search on the semantic index and hydrates results from master.
        """
        results = self.semantic_index.search(vector=query_vector, top_k=limit)
        
        profiles = []
        for doc, _ in results:
            profile = self.get(doc.id)
            if profile:
                profiles.append(profile)
        return profiles

    def get_all(self) -> List[VoiceProfile]:
        """
        Returns all profiles from the master dictionary.
        """
        return list(self.master_data.values())

    def delete(self, voice_id: str) -> bool:
        """
        Removes the profile from the master store and all associated indices.
        """
        try:
            self.master_data.delete(voice_id)
            self.identity_index.drop(voice_id)
            self.semantic_index.drop(voice_id)
            return True
        except (KeyError, Exception):
            return False

    def count(self) -> int:
        """
        Returns the count from the master store.
        """
        return self.master_data.count()