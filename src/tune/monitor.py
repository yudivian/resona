import time
import logging
from typing import Dict

from src.models import TuneStatus, AppConfig, TuneProgress
from src.tune.registry import TuneRegistry

logger = logging.getLogger(__name__)

class TuneMonitor:
    """
    Manages in-memory telemetry for fine-tuning jobs, buffering metrics to avoid 
    database I/O bottlenecks and checking for manual cancellation signals.
    """

    def __init__(self, registry: TuneRegistry, config: AppConfig):
        """
        Initializes the TuneMonitor.

        Args:
            registry (TuneRegistry): The database registry to sync progress.
            config (AppConfig): Global application configuration.
        """
        self.registry = registry
        self.sync_interval = config.models.finetune.sync_interval_seconds
        self.in_memory_progress: Dict[str, TuneProgress] = {}
        self.last_sync_time: Dict[str, float] = {}

    def _initialize_memory_if_needed(self, tune_id: str) -> None:
        """
        Prepares the local RAM buffer for a specific tune job if it doesn't exist.

        Args:
            tune_id (str): The identifier of the tune job.
        """
        if tune_id not in self.in_memory_progress:
            record = self.registry.get_tune(tune_id)
            if record and record.progress:
                self.in_memory_progress[tune_id] = record.progress
            else:
                self.in_memory_progress[tune_id] = TuneProgress()
            self.last_sync_time[tune_id] = time.time()

    def _sync_to_disk(self, tune_id: str) -> None:
        """
        Flushes the current in-memory progress metrics to the persistent database.

        Args:
            tune_id (str): The identifier of the tune job.
        """
        if tune_id in self.in_memory_progress:
            progress_data = self.in_memory_progress[tune_id].model_dump()
            self.registry.update_tune(tune_id, {"progress": progress_data})
            self.last_sync_time[tune_id] = time.time()

    def log_training_step(
        self, 
        tune_id: str, 
        epoch: int, 
        total_epochs: int, 
        step: int, 
        total_steps: int, 
        loss: float
    ) -> None:
        """
        Records a single training step iteration in RAM, calculates overall progress, 
        and triggers a database sync if the configured interval has elapsed.

        Args:
            tune_id (str): The identifier of the tune job.
            epoch (int): The current epoch index.
            total_epochs (int): The total number of planned epochs.
            step (int): The current batch step within the epoch.
            total_steps (int): The total number of batches per epoch.
            loss (float): The calculated loss value for the current step.
        """
        self._initialize_memory_if_needed(tune_id)
        
        progress = self.in_memory_progress[tune_id]
        progress.current_epoch = epoch
        progress.total_epochs = total_epochs
        progress.current_step = step
        progress.total_steps = total_steps
        progress.current_loss = loss
        progress.loss_history.append(loss)
        
        total_global_steps = total_epochs * total_steps
        current_global_step = (epoch * total_steps) + step
        progress.percent = (current_global_step / total_global_steps) * 100.0 if total_global_steps > 0 else 0.0

        current_time = time.time()
        if current_time - self.last_sync_time[tune_id] >= self.sync_interval:
            self._sync_to_disk(tune_id)

    def is_cancelled(self, tune_id: str) -> bool:
        """
        Checks the persistent database to determine if the user has requested 
        to abort the ongoing training job.

        Args:
            tune_id (str): The identifier of the tune job.

        Returns:
            bool: True if the job status is marked as CANCELED, False otherwise.
        """
        record = self.registry.get_tune(tune_id)
        if not record:
            return False
        return record.status == TuneStatus.CANCELED

    def force_sync(self, tune_id: str) -> None:
        """
        Manually forces a write of the in-memory metrics to the database, 
        bypassing the timer. Useful for final wrap-up operations.

        Args:
            tune_id (str): The identifier of the tune job.
        """
        self._sync_to_disk(tune_id)