import os
from typing import List, Optional
from beaver import BeaverDB, Document
from src.models import VoiceProfile, AppConfig

class VoiceStore:
    """
    Unified persistence layer using BeaverDB with native Vector Support.
    """
    def __init__(self, config: AppConfig):
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

    def save(self, profile: VoiceProfile) -> str:
        """
        Saves the profile metadata and updates vector indices.
        """
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
        Retrieves a voice profile by its ID.
        """
        try:
            data = self.master_data.get(voice_id)
            if data:
                return VoiceProfile(**data)
            return None
        except KeyError:
            return None

    def search_identity(self, query_vector: List[float], limit: int = 5) -> List[VoiceProfile]:
        """
        Performs vector search on the identity index.
        """
        results = self.identity_index.search(query_vector, k=limit)
        
        profiles = []
        for match in results:
            profile = self.get(match.item.id)
            if profile:
                profiles.append(profile)
        return profiles

    def search_semantic(self, query_vector: List[float], limit: int = 5) -> List[VoiceProfile]:
        """
        Performs vector search on the semantic index.
        """
        results = self.semantic_index.search(query_vector, k=limit)
        
        profiles = []
        for match in results:
            profile = self.get(match.item.id)
            if profile:
                profiles.append(profile)
        return profiles

    def get_all(self) -> List[VoiceProfile]:
        """
        Retrieves all stored voice profiles.
        """
        return [VoiceProfile(**data) for data in self.master_data.values()]

    def delete(self, voice_id: str) -> bool:
        """
        Deletes a profile from storage.
        """
        try:
            if voice_id in self.master_data:
                del self.master_data[voice_id]
            return True
        except Exception:
            return False

    def count(self) -> int:
        """
        Returns the total number of profiles.
        """
        return len(self.master_data)