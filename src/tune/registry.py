import time
import logging
from typing import List, Optional, Any, Dict
from beaver import BeaverDB

from src.models import TuneRecord, TuneStatus, AppConfig

logger = logging.getLogger(__name__)

class TuneRegistry:
    """
    Manages the persistence of tune records using BeaverDB's dictionary structure.
    """

    def __init__(self, config: AppConfig):
        """
        Initializes the TuneRegistry and the persistent dictionary.

        Args:
            config (AppConfig): Global application configuration.
        """
        self.db = BeaverDB(config.paths.db_file)
        self.tunes_data = self.db.dict("tunes_data")

    def create_tune(self, record: TuneRecord) -> TuneRecord:
        """
        Persists a new tune record in the database.

        Args:
            record: The TuneRecord instance to persist.

        Returns:
            The persisted TuneRecord instance.
        """
        self.tunes_data[record.id] = record.model_dump(mode="json")
        return record

    def get_tune(self, tune_id: str) -> Optional[TuneRecord]:
        """
        Retrieves a specific tune by its ID.

        Args:
            tune_id: The unique identifier of the tune.

        Returns:
            Optional[TuneRecord]: The tune record if found, otherwise None.
        """
        data = self.tunes_data.get(tune_id)
        if data:
            return TuneRecord(**data)
        return None

    def list_tunes(self) -> List[TuneRecord]:
        """
        Returns all tunes stored in the database.

        Returns:
            List[TuneRecord]: Sorted by creation time descending.
        """
        records = [TuneRecord(**data) for data in self.tunes_data.values()]
        return sorted(records, key=lambda x: x.created_at, reverse=True)

    def update_status(self, tune_id: str, status: TuneStatus, error_message: Optional[str] = None) -> Optional[TuneRecord]:
        """
        Updates the status and timestamp of a tune.

        Args:
            tune_id: The unique identifier.
            status: New status to apply.
            error_message: Optional error description.

        Returns:
            Optional[TuneRecord]: The updated record.
        """
        record = self.get_tune(tune_id)
        if not record:
            return None

        record.status = status
        record.updated_at = time.time()
        if error_message is not None:
            record.error_message = error_message

        return self.create_tune(record)

    def update_tune(self, tune_id: str, updates: Dict[str, Any]) -> Optional[TuneRecord]:
        """
        Performs a partial update on a tune record.

        Args:
            tune_id: The unique identifier.
            updates: Dictionary of fields to update.

        Returns:
            Optional[TuneRecord]: The updated record.
        """
        record = self.get_tune(tune_id)
        if not record:
            return None

        record_data = record.model_dump()
        record_data.update(updates)
        record_data["updated_at"] = time.time()

        updated_record = TuneRecord(**record_data)
        return self.create_tune(updated_record)

    def delete_tune(self, tune_id: str) -> bool:
        """
        Removes a tune from the database.

        Args:
            tune_id: Unique identifier.

        Returns:
            bool: True if deleted.
        """
        if tune_id in self.tunes_data:
            self.tunes_data.pop(tune_id, None)
            return True
        return False