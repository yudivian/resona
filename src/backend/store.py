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

    This class serves as the single source of truth for voice profile storage.
    It manages two distinct data structures:
    1. A master dictionary storing the full JSON serialization of VoiceProfile objects.
    2. Specialized vector collections ('idx_identity' and 'idx_semantic') to facilitate
       similarity search and retrieval.

    It adheres strictly to the BeaverDB API, using the `.index()` method for both
    insertion and updates (upsert), and ensures that all physical assets (WAV files)
    are safely archived in the configured assets directory.
    """

    def __init__(self, config: AppConfig):
        """
        Initializes the VoiceStore and ensures the integrity of the storage directories.

        This constructor verifies that the database file path and the assets directory
        exist. If they do not, it creates the necessary folder structures to prevent
        runtime IO errors during save operations. It also initializes the BeaverDB
        connection and loads references to the master dictionary and vector collections.

        Args:
            config (AppConfig): The global application configuration object containing
                                paths for the database file and assets directory.
        """
        self.config = config
        os.makedirs(os.path.dirname(config.paths.db_file), exist_ok=True)
        os.makedirs(config.paths.assets_dir, exist_ok=True)

        self.db = BeaverDB(config.paths.db_file)
        self.master_data = self.db.dict("voices_data")
        self.identity_index = self.db.collection("idx_identity")
        self.semantic_index = self.db.collection("idx_semantic")

    def add_profile(self, profile: VoiceProfile, anchor_source_path: Optional[str] = None) -> str:
        """
        Persists a complete VoiceProfile to the database and indexes its vector representations.

        This method performs a multi-step save operation:
        1. Asset Archival: If a source audio file is provided, it is physically copied
           to the secure assets directory with a timestamped filename.
        2. Metadata Storage: The full profile model is serialized and stored in the
           master key-value dictionary.
        3. Identity Indexing: The speaker embedding is wrapped in a Document and
           indexed in the 'idx_identity' collection for potential speaker recognition features.
        4. Semantic Indexing: If a semantic description vector exists, it is indexed
           in the 'idx_semantic' collection to enable text-to-voice search capabilities.

        Args:
            profile (VoiceProfile): The fully constructed Pydantic model containing all
                                    voice metadata and embeddings.
            anchor_source_path (Optional[str]): The absolute filesystem path to the temporary
                                                WAV file that generated this identity. If provided,
                                                it is archived permanently.

        Returns:
            str: The unique UUID string of the persisted profile.
        """
        if anchor_source_path and os.path.exists(anchor_source_path):
            safe_name = profile.name.replace(' ', '_').lower()
            filename = f"{safe_name}_{int(time.time())}_anchor.wav"
            destination = os.path.join(self.config.paths.assets_dir, filename)
            shutil.copy2(anchor_source_path, destination)
            profile.anchor_audio_path = filename

        self.master_data[profile.id] = profile.model_dump()
        
        doc_identity = Document(
            id=profile.id,
            embedding=profile.identity_embedding,
            text=profile.name,
            metadata={"type": "identity"}
        )
        self.identity_index.index(doc_identity)
        
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

    def delete_profile(self, profile_id: str):
        """
        Permanently deletes a voice profile from the database, its associated 
        vector indices, and the physical anchor audio file.

        This method strictly follows the BeaverDB API:
        1. Deletes metadata from the master dictionary using pop/del.
        2. Removes vector entries from collections using drop().
        3. Physically deletes the reference WAV file from the assets directory.

        Args:
            profile_id (str): The unique UUID of the profile to remove.
        """
        # 1. Retrieve profile to find the physical asset path
        data = self.master_data.get(profile_id)
        if not data:
            logger.warning(f"Delete failed: Profile {profile_id} not found.")
            return

        profile = VoiceProfile(**data)

        # 2. Remove physical audio asset from disk
        if profile.anchor_audio_path:
            file_path = os.path.join(self.config.paths.assets_dir, profile.anchor_audio_path)
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    logger.info(f"Deleted physical asset: {file_path}")
                except Exception as e:
                    logger.error(f"Failed to delete physical asset {file_path}: {e}")

        # 3. Delete metadata from BeaverDB dictionary
        # pop() is safer as it handles the removal and allows verification
        self.master_data.pop(profile_id, None)

        # 4. Delete vectors from collections using BeaverDB drop()
        # BeaverDB drop expects a document identifier or the document itself
        try:
            self.identity_index.drop(profile_id)
            if profile.semantic_embedding:
                self.semantic_index.drop(profile_id)
        except Exception as e:
            logger.error(f"Error dropping vectors for {profile_id} in BeaverDB: {e}")

        logger.info(f"Profile {profile_id} successfully removed from store and indices.")
    
    def get_profile(self, profile_id: str) -> Optional[VoiceProfile]:
        """
        Retrieves a single VoiceProfile object by its unique identifier.

        This method looks up the profile data in the master dictionary. If the ID exists,
        it deserializes the data back into a Pydantic VoiceProfile model. This is critical
        for loading pre-existing voices into the inference engine without re-generation.

        Args:
            profile_id (str): The UUID string of the profile to retrieve.

        Returns:
            Optional[VoiceProfile]: The instantiated VoiceProfile object if the ID exists,
                                    otherwise None.
        """
        data = self.master_data.get(profile_id)
        if data:
            return VoiceProfile(**data)
        return None

    def get_all(self) -> List[VoiceProfile]:
        """
        Retrieves all stored voice profiles available in the system.

        This method iterates through the entire master dictionary and reconstructs
        VoiceProfile objects for every entry. It is primarily used to populate
        UI selection lists or for batch processing.

        Returns:
            List[VoiceProfile]: A list of all instantiated VoiceProfile objects currently
                                stored in the database.
        """
        return [VoiceProfile(**data) for data in self.master_data.values()]