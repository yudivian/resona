import sys
import os
import time
import logging
import traceback
import torch
from pathlib import Path
from typing import Optional, Dict, Any

from beaver import BeaverDB

from src.config import settings
from src.models import DialogProject, ProjectStatus, LineStatus
from src.backend.store import VoiceStore
from src.backend.engine import TTSModelProvider, InferenceEngine
from src.emotions.manager import EmotionManager
from src.dialogs.resolver import DialogResolver
from src.dialogs.cluster import DialogClusterer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [WORKER] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.abspath("worker_trace.log"))
    ],
    force=True
)

class DialogWorker:
    """
    Independent background worker responsible for the end-to-end synthesis pipeline.

    This service ensures maximum throughput by instantiating the heavy neural network 
    weights strictly once. It delegates subsequent language and contextual variations 
    to lightweight, dynamic inference parameters during the active generation loop. 
    It incorporates rigorous database retry mechanics to prevent process termination 
    caused by concurrent SQLite locks acquired by external UI polling.
    """

    def __init__(self, project_id: str) -> None:
        """
        Initializes the background worker and establishes references to localized 
        resource managers.

        Args:
            project_id (str): The unique canonical identifier of the active project.
        """
        self.project_id = project_id
        self.voice_store = VoiceStore(settings)
        self.emotion_manager = EmotionManager(settings)
        self.resolver = DialogResolver(self.voice_store, self.emotion_manager)
        self.clusterer = DialogClusterer()

    def _update_db(self, project: DialogProject, retries: int = 15) -> None:
        """
        Persists the updated project state utilizing a volatile database connection 
        and an exponential backoff strategy to ensure concurrent safety.

        Args:
            project (DialogProject): The rigorously validated data model mapping.
            retries (int, optional): The maximum threshold of write attempts. Defaults to 15.

        Raises:
            Exception: If the underlying file system remains locked after exhaustion of the retry limit.
        """
        for i in range(retries):
            try:
                db_instance = BeaverDB(settings.paths.db_file)
                db_instance.dict("dialog_projects")[self.project_id] = project.model_dump()
                return
            except Exception as e:
                if i < retries - 1:
                    time.sleep(0.5)
                    continue
                logging.error(f"Database sync failed after {retries} attempts: {e}")
                raise

    def _get_project_data(self, retries: int = 15) -> Optional[Dict[str, Any]]:
        """
        Retrieves raw dictionary metadata from the persistence layer utilizing 
        a volatile connection loop to evade read contention.

        Args:
            retries (int, optional): The upper limit for discrete read attempts. Defaults to 15.

        Returns:
            Optional[Dict[str, Any]]: The parsed dictionary correlating to the project, or None.

        Raises:
            Exception: If SQLite throws a sequential lock timeout error exceeding the loop allowance.
        """
        for i in range(retries):
            try:
                db_instance = BeaverDB(settings.paths.db_file)
                return db_instance.dict("dialog_projects").get(self.project_id)
            except Exception as e:
                if i < retries - 1:
                    time.sleep(0.5)
                    continue
                logging.error(f"DB read failed after {retries} attempts: {e}")
                raise
        return None

    def _write_raw_data_safe(self, data: Dict[str, Any], retries: int = 10) -> None:
        """
        Executes a raw structural override on the database storage, explicitly bypassing 
        Pydantic validation layers for emergency error logging.

        Args:
            data (Dict[str, Any]): The unvalidated error state payload.
            retries (int, optional): The cap for forced write iterations. Defaults to 10.

        Raises:
            Exception: If critical file contention blocks the emergency payload serialization.
        """
        for i in range(retries):
            try:
                db_instance = BeaverDB(settings.paths.db_file)
                db_instance.dict("dialog_projects")[self.project_id] = data
                return
            except Exception as e:
                if i < retries - 1:
                    time.sleep(0.5)
                    continue
                logging.error(f"Raw DB write failed after {retries} attempts: {e}")
                raise

    def _get_project(self) -> Optional[DialogProject]:
        """
        Deserializes the target project metadata from structural dictionaries into 
        a strongly-typed Pydantic instance.

        Returns:
            Optional[DialogProject]: The structured model definition, or None upon absence.
        """
        data = self._get_project_data()
        if data:
            return DialogProject(**data)
        return None

    def _get_emotion_params(self, emotion_id: str, intensity_id: Optional[str]) -> Dict[str, float]:
        """
        Derives discrete numerical inference boundaries directly mapped to the 
        canonical semantic emotion identifiers.

        Args:
            emotion_id (str): The primary emotion categorization key.
            intensity_id (Optional[str]): The supplementary magnitude multiplier mapping.

        Returns:
            Dict[str, float]: The computed dictionary housing thermal parameters and penalties.
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
        Executes the primary lifecycle loop for audio synthesis.

        It resolves project dependencies and clusters sequential lines. It enforces 
        high-efficiency hardware usage by loading the primary synthesis weights exclusively 
        once prior to the execution loop. Individual parameters are hot-swapped per 
        inference pass, ensuring minimal execution latency.

        Raises:
            Forces an interception of all unhandled stack traces, injecting them securely 
            into the project state for cross-process visibility before termination.
        """
        try:
            logging.info(f"Worker initialized for project: {self.project_id}")
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
            
            logging.info("Initializing global TTS model provider (Single Load)...")
            tts_provider = TTSModelProvider(settings)

            for cluster in clusters:
                project_data = self._get_project_data() or {}
                db_state = str(project_data.get('status', '')).lower()
                
                if db_state in [ProjectStatus.PAUSED.value, ProjectStatus.CANCELLED.value, "paused", "cancelled"]:
                    logging.info(f"Execution halted by orchestrator signal. Status: {db_state}")
                    if torch.cuda.is_available(): 
                        torch.cuda.empty_cache()
                    return

                logging.info(f"Processing cluster: {len(cluster.texts)} lines for language: {cluster.language}")
                audio_dir = Path(project.project_path) / "audio"
                audio_dir.mkdir(parents=True, exist_ok=True)
                voice_profile = self.voice_store.get_profile(cluster.voice_id)
                
                engine = InferenceEngine(config=settings, tts_provider=tts_provider, lang=cluster.language)
                engine.load_identity_from_state(vector=voice_profile.identity_embedding, seed=voice_profile.seed)
                output_paths = [str(audio_dir / f"{sid}.wav") for sid in cluster.state_ids]

                if cluster.emotion_id:
                    params = self._get_emotion_params(cluster.emotion_id, cluster.intensity_id)
                    engine.render_batch_with_emotion(cluster.texts, params, output_paths)
                else:
                    engine.render_batch(cluster.texts, output_paths)

                project = self._get_project()
                if not project: 
                    logging.error("Project lost from DB during generation.")
                    break

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
            logging.error(f"Fatal error encountered: {traceback.format_exc()}")
            try:
                data = self._get_project_data()
                if data:
                    data['status'] = ProjectStatus.FAILED.value
                    data['error'] = str(e)
                    self._write_raw_data_safe(data)
            except Exception as inner_e:
                logging.error(f"Catastrophic failure saving state: {inner_e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        try:
            worker = DialogWorker(sys.argv[1])
            worker.run()
        finally:
            os._exit(0)