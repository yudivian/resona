import sys
import logging
from pathlib import Path

from src.models import TuneRecord, TuneStatus
from src.tune.registry import TuneRegistry
from config import settings

logger = logging.getLogger(__name__)

class DataHandler:
    """
    Handles the data preparation pipeline by natively importing the external 
    prepare_data.py script and dynamically injecting command-line arguments.
    """

    def __init__(self, registry: TuneRegistry):
        """
        Initializes the DataHandler and ensures the external scripts are importable.

        Args:
            registry (TuneRegistry): The registry to update tune statuses.
        """
        self.registry = registry
        self.device = settings.system.compute.tune_device
        self.tokenizer_model_path = settings.models.tts.base_repo_id

        self.qwen_scripts_dir = Path(settings.paths.temp_dir).parent / "qwen3-tts-finetune"
        if str(self.qwen_scripts_dir) not in sys.path:
            sys.path.append(str(self.qwen_scripts_dir))

    def _process_dataset(self, raw_manifest_path: str, processed_manifest_path: str) -> bool:
        """
        Imports the prepare_data module, mocks sys.argv to bypass argparse limitations, 
        executes the main logic, and restores the original sys.argv.

        Args:
            raw_manifest_path (str): Path to the input JSONL manifest.
            processed_manifest_path (str): Path to the output JSONL manifest.
            
        Returns:
            bool: True if execution was successful, False otherwise.
        """
        try:
            import prepare_data

            original_argv = sys.argv.copy()
            
            sys.argv = [
                "prepare_data.py",
                "--device", self.device,
                "--tokenizer_model_path", self.tokenizer_model_path,
                "--input_jsonl", raw_manifest_path,
                "--output_jsonl", processed_manifest_path
            ]

            prepare_data.main()

            sys.argv = original_argv
            
            return True

        except ImportError as e:
            logger.error(f"Could not import prepare_data.py: {e}")
            return False
        except SystemExit as e:
            logger.error(f"prepare_data.py executed a SystemExit (likely argparse error): {e}")
            sys.argv = original_argv
            return False
        except Exception as e:
            logger.error(f"Dataset processing failed: {e}")
            sys.argv = original_argv
            return False

    def prepare_tune_data(self, tune_id: str) -> bool:
        """
        Executes the full data preparation pipeline for a specific tune job.

        Args:
            tune_id (str): The unique identifier of the tune.

        Returns:
            bool: True if the preparation was completely successful, False otherwise.
        """
        record = self.registry.get_tune(tune_id)
        if not record:
            logger.error(f"Tune {tune_id} not found in registry.")
            return False

        self.registry.update_status(tune_id, TuneStatus.PROCESSING_DATA)

        workspace_path = Path(record.workspace_path)
        raw_manifest = record.dataset_manifest_path
        processed_manifest = workspace_path / "processed" / "train_with_codes.jsonl"

        success = self._process_dataset(raw_manifest, str(processed_manifest))
        
        if not success:
            self.registry.update_status(tune_id, TuneStatus.FAILED, error_message="Failed processing dataset codes via prepare_data.py.")
            return False

        self.registry.update_tune(tune_id, {
            "dataset_manifest_path": str(processed_manifest)
        })
        
        return True