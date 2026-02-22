import os
import json
import shutil
import uuid
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

from src.models import TuneRecord
from src.tune.registry import TuneRegistry
from config import settings

logger = logging.getLogger(__name__)

class TuningOrchestrator:
    """
    Orchestrates the physical workspace and logical registration of tuning jobs.
    """

    def __init__(self, registry: TuneRegistry):
        """
        Initializes the TuningOrchestrator.

        Args:
            registry (TuneRegistry): The registry instance to persist tune metadata.
        """
        self.registry = registry
        self.base_dir = Path(settings.paths.tunespace_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def setup_workspace(self, tune_id: str) -> Path:
        """
        Creates the required directory structure for a new tune job.

        Args:
            tune_id (str): The unique identifier of the tune.

        Returns:
            Path: The path to the root of the tune's workspace.
        """
        tune_dir = self.base_dir / tune_id
        
        (tune_dir / "raw" / "dataset").mkdir(parents=True, exist_ok=True)
        (tune_dir / "processed").mkdir(parents=True, exist_ok=True)
        (tune_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (tune_dir / "logs").mkdir(parents=True, exist_ok=True)
        
        return tune_dir

    def initialize_tune(
        self, 
        name: str, 
        dataset_samples: List[Dict[str, str]], 
        reference_audio_source: str, 
        training_params: Dict[str, Any],
        language: str = "Auto",
        description: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> TuneRecord:
        """
        Sets up a new tune by creating its workspace, copying raw assets, 
        generating the training manifest, and registering it in the database.

        Args:
            name (str): The display name for the tune.
            dataset_samples (List[Dict[str, str]]): A list of dictionaries containing 
                                                    'audio_path' and 'text' keys.
            reference_audio_source (str): The path to the reference audio file.
            training_params (Dict[str, Any]): Hyperparameters for the fine-tuning process.
            language (str, optional): The language of the tune. Defaults to "Auto".
            description (Optional[str], optional): A brief description. Defaults to None.
            tags (Optional[List[str]], optional): Tags for categorization. Defaults to None.

        Returns:
            TuneRecord: The fully constructed and persisted tune record.
        
        Raises:
            FileNotFoundError: If the reference audio file does not exist.
        """
        tune_id = str(uuid.uuid4())
        workspace_path = self.setup_workspace(tune_id)
        
        raw_dir = workspace_path / "raw"
        dataset_dir = raw_dir / "dataset"
        
        target_reference_path = raw_dir / "reference.wav"
        manifest_path = raw_dir / "train_raw.jsonl"
        
        if not os.path.exists(reference_audio_source):
            raise FileNotFoundError(f"Reference audio not found: {reference_audio_source}")
            
        shutil.copy2(reference_audio_source, target_reference_path)

        manifest_entries = []
        
        for idx, sample in enumerate(dataset_samples):
            source_audio = sample.get("audio_path")
            transcript = sample.get("text")
            
            if not source_audio or not os.path.exists(source_audio):
                continue
                
            _, ext = os.path.splitext(source_audio)
            target_audio = dataset_dir / f"sample_{idx}{ext}"
            
            shutil.copy2(source_audio, target_audio)
            
            manifest_entries.append({
                "audio": str(target_audio),
                "text": transcript,
                "ref_audio": str(target_reference_path)
            })

        with open(manifest_path, "w", encoding="utf-8") as f:
            for entry in manifest_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        record = TuneRecord(
            id=tune_id,
            name=name,
            workspace_path=str(workspace_path),
            dataset_manifest_path=str(manifest_path),
            reference_audio_path=str(target_reference_path),
            training_params=training_params,
            language=language,
            description=description,
            tags=tags or []
        )

        return self.registry.create_tune(record)

    def cleanup_workspace(self, tune_id: str) -> bool:
        """
        Removes temporary heavy directories after a tune is completed 
        or failed, keeping only the checkpoints and logs to save disk space.

        Args:
            tune_id (str): The unique identifier of the tune.

        Returns:
            bool: True if cleanup was successful, False otherwise.
        """
        tune_dir = self.base_dir / tune_id
        
        if not tune_dir.exists():
            return False
            
        try:
            raw_dir = tune_dir / "raw"
            processed_dir = tune_dir / "processed"
            
            if raw_dir.exists():
                shutil.rmtree(raw_dir)
            if processed_dir.exists():
                shutil.rmtree(processed_dir)
                
            return True
        except OSError as e:
            logger.error(f"Failed to clean workspace for tune {tune_id}: {e}")
            return False

    def delete_tune(self, tune_id: str) -> bool:
        """
        Deletes the physical workspace directory of a tune completely and 
        removes its corresponding record from the database registry.

        Args:
            tune_id (str): The unique identifier of the tune.

        Returns:
            bool: True if both the physical directory and the registry record 
                  were successfully deleted. False if any step failed.
        """
        tune_dir = self.base_dir / tune_id
        
        if tune_dir.exists() and tune_dir.is_dir():
            try:
                shutil.rmtree(tune_dir)
            except OSError as e:
                logger.error(f"Failed to delete workspace for tune {tune_id}: {e}")
                return False
                
        return self.registry.delete_tune(tune_id)