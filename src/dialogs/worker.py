import sys
import os
import time
import logging
import traceback
import torch
from pathlib import Path
from typing import Optional, Dict

from beaver import BeaverDB

from src.config import settings
from src.models import DialogProject, ProjectStatus, LineStatus
from src.backend.store import VoiceStore
from src.backend.engine import TTSModelProvider, InferenceEngine
from src.emotions.manager import EmotionManager
from src.dialogs.resolver import DialogResolver
from src.dialogs.cluster import DialogClusterer

log_path = os.path.abspath("worker_trace.log")
logging.basicConfig(
    filename=log_path,
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] - %(message)s',
    force=True
)

class DialogWorker:
    """
    Independent worker responsible for the end-to-end synthesis pipeline.
    
    It manages GPU resource allocation, project state persistence with retry 
    logic, and executes batched inference through the InferenceEngine.
    """

    def __init__(self, project_id: str) -> None:
        """
        Initializes the worker with database and resource manager references.
        """
        self.project_id = project_id
        self.db = BeaverDB(settings.paths.db_file)
        self.projects_dict = self.db.dict("dialog_projects")
        
        self.voice_store = VoiceStore(settings)
        self.emotion_manager = EmotionManager(settings)
        
        self.resolver = DialogResolver(self.voice_store, self.emotion_manager)
        self.clusterer = DialogClusterer()

    def _update_db(self, project: DialogProject, retries: int = 5) -> None:
        """
        Persists the current project state into BeaverDB with exponential backoff.
        
        This prevents worker crashes during concurrent database access from the UI.
        """
        for i in range(retries):
            try:
                self.projects_dict[self.project_id] = project.model_dump()
                return
            except Exception as e:
                if "locked" in str(e).lower() and i < retries - 1:
                    time.sleep(0.1 * (2 ** i))
                    continue
                logging.error(f"Database sync failed after {retries} attempts: {e}")
                raise

    def _get_emotion_params(self, emotion_id: str, intensity_id: Optional[str]) -> Dict[str, float]:
        """
        Resolves canonical emotion IDs into numerical inference hyperparameters.
        """
        catalog = self.emotion_manager.catalog
        emotion_data = catalog.get("emotions", {}).get(emotion_id, {})
        params = emotion_data.get("parameters", {"temp": 0.5, "top_p": 0.85, "penalty": 1.0})
        
        if intensity_id:
            intensity_data = catalog.get("intensifiers", {}).get(intensity_id, {})
            mods = intensity_data.get("modifiers", {"temp": 1.0, "top_p": 1.0, "penalty": 1.0})
            params = {
                "temp": params.get("temp", 0.5) * mods.get("temp", 1.0),
                "top_p": params.get("top_p", 0.85) * mods.get("top_p", 1.0),
                "penalty": params.get("penalty", 1.0) * mods.get("penalty", 1.0)
            }
        return params
    
    def _get_project(self) -> Optional[DialogProject]:
        """
        Retrieves the most recent project state from the persistent database.

        This method acts as the primary data ingestion point for the worker,
        ensuring it operates on the latest architectural definition and state
        transitions. It fetches the raw dictionary representation from BeaverDB
        and deserializes it into a strictly validated DialogProject model.

        Returns:
            Optional[DialogProject]: The validated project model instance, or
                                     None if the project identifier is absent.
        """
        data = self.projects_dict.get(self.project_id)
        if data:
            return DialogProject(**data)
        return None

    def _get_emotion_params(self, emotion_id: str, intensity_id: Optional[str]) -> Dict[str, float]:
        """
        Resolves canonical emotion identifiers into numerical inference hyperparameters.

        This internal helper interacts with the EmotionManager to retrieve the base
        temperature, top_p, and repetition penalty vectors associated with a specific
        emotion. It optionally applies multiplicative modifiers based on the provided
        intensity level, yielding a discrete configuration dictionary ready for the TTS engine.

        Args:
            emotion_id (str): The primary identifier for the target emotion.
            intensity_id (Optional[str]): The identifier for the desired emotional magnitude.

        Returns:
            Dict[str, float]: The computed hyperparameters formatted for the InferenceEngine.
        """
        catalog = self.emotion_manager.catalog
        emotion_data = catalog.get("emotions", {}).get(emotion_id, {})
        params = emotion_data.get("parameters", {"temp": 0.5, "top_p": 0.85, "penalty": 1.0})
        
        if intensity_id:
            intensity_data = catalog.get("intensifiers", {}).get(intensity_id, {})
            mods = intensity_data.get("modifiers", {"temp": 1.0, "top_p": 1.0, "penalty": 1.0})
            params = {
                "temp": params.get("temp", 0.5) * mods.get("temp", 1.0),
                "top_p": params.get("top_p", 0.85) * mods.get("top_p", 1.0),
                "penalty": params.get("penalty", 1.0) * mods.get("penalty", 1.0)
            }
            
        return params

    def run(self) -> None:
        """
        Executes the main synthesis loop for the assigned dialog project.

        This method retrieves the project definition from the database and passes it
        to the DialogResolver, which mutates the project in-place to validate acoustic
        and emotional dependencies. It subsequently employs the DialogClusterer to
        aggregate homogenous lines into optimized generation batches. An asynchronous
        state-polling mechanism is executed prior to every cluster generation to
        detect external lifecycle signals (such as pause or cancel directives) issued
        by the orchestrator. For each valid cluster, a dedicated InferenceEngine is
        instantiated based on the required language profile. Rendered audio tracks
        are safely persisted to disk, and internal project states are atomically
        committed back to the database.

        Raises:
            Exception: Captures and logs any catastrophic failure during inference,
                       subsequently updating the project's global state to failed
                       and storing the specific traceback in the database record.
        """
        try:
            project = self._get_project()
            if not project:
                logging.error("Project not found in DB.")
                return

            project.status = ProjectStatus.GENERATING
            project.pid = os.getpid()
            self._update_db(project)

            self.resolver.resolve_project(project)
            self._update_db(project)

            clusters = self.clusterer.build_clusters(project)

            tts_provider = TTSModelProvider(settings)

            for cluster in clusters:
                project_data = self.projects_dict.get(self.project_id, {})
                db_state = project_data.get('status')
                
                if db_state in [
                    ProjectStatus.PAUSED.value, 
                    ProjectStatus.CANCELLED.value, 
                    "paused", 
                    "cancelled"
                ]:
                    logging.info(f"Execution halted by orchestrator signal. Status: {db_state}")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    return

                audio_dir = Path(project.project_path) / "audio"
                audio_dir.mkdir(parents=True, exist_ok=True)

                voice_profile = self.voice_store.get_profile(cluster.voice_id)
                
                engine = InferenceEngine(
                    config=settings, 
                    tts_provider=tts_provider, 
                    lang=cluster.language
                )

                engine.load_identity_from_state(
                    vector=voice_profile.identity_embedding,
                    seed=voice_profile.seed
                )

                output_paths = [str(audio_dir / f"{sid}.wav") for sid in cluster.state_ids]

                if cluster.emotion_id:
                    params = self._get_emotion_params(cluster.emotion_id, cluster.intensity_id)
                    engine.render_batch_with_emotion(cluster.texts, params, output_paths)
                else:
                    engine.render_batch(cluster.texts, output_paths)

                project = self._get_project()

                for sid in cluster.state_ids:
                    for s in project.states:
                        if s.id == sid:
                            s.status = LineStatus.COMPLETED
                            s.audio_path = f"audio/{sid}.wav"
                
                project.updated_at = time.time()
                self._update_db(project)
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            project.status = ProjectStatus.COMPLETED
            self._update_db(project)
            logging.info("Project completed successfully.")

        except Exception as e:
            logging.error(f"Fatal error: {traceback.format_exc()}")
            project = self._get_project()
            if project:
                project.status = ProjectStatus.FAILED
                project.error = str(e)
                self._update_db(project)



if __name__ == "__main__":
    if len(sys.argv) > 1:
        DialogWorker(sys.argv[1]).run()