import subprocess
import os
import sys
import psutil
import shutil
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from beaver import BeaverDB

from src.config import settings
from src.models import DialogProject, ProjectStatus, LineStatus, ProjectSource
from src.backend.audio_engine import AudioEngine, AudioSegmentConfig


logger = logging.getLogger("Orchestrator")
logger.setLevel(logging.DEBUG)

class DialogOrchestrator:
    """
    Service responsible for the lifecycle management and supervision of detached 
    background synthesis processes.

    The orchestrator functions entirely through transient queries and implements 
    absolute process detachment. It severs all OS-level file descriptors, streams, 
    and session groups between the parent web framework and the child AI worker, 
    routing standard output to physical files to ensure zero cross-process contention.
    """

    def __init__(self) -> None:
        """
        Initializes the DialogOrchestrator architecture utilizing stateless operations.
        """
        pass

    def get_project_data_safe(self, project_id: str, retries: int = 10) -> Optional[Dict[str, Any]]:
        """
        Query implementation designed to extract project payloads utilizing transient 
        SQLite bindings protected by iteration fallbacks.

        Args:
            project_id (str): The target identifier mapped to the payload.
            retries (int, optional): Evaluation cap for access retries. Defaults to 10.

        Returns:
            Optional[Dict[str, Any]]: The parsed mapping dict, or None if extraction fails.
        """
        for i in range(retries):
            try:
                db = BeaverDB(settings.paths.db_file)
                data = db.dict("dialog_projects").get(project_id)
                return data
            except Exception:
                if i < retries - 1:
                    time.sleep(0.25)
                    continue
        return None

    def _write_project_data_safe(self, project_id: str, data: Dict[str, Any], retries: int = 10) -> bool:
        """
        Commits structural mutations to the database without retaining file hooks.

        Args:
            project_id (str): The persistent UUID identifier array key.
            data (Dict[str, Any]): The structured metadata corresponding to the target project.
            retries (int, optional): The fallback evaluation threshold. Defaults to 10.

        Returns:
            bool: True if the atomic operation succeeds cleanly, False upon absolute iteration failure.
        """
        for i in range(retries):
            try:
                db = BeaverDB(settings.paths.db_file)
                db.dict("dialog_projects")[project_id] = data
                return True
            except Exception:
                if i < retries - 1:
                    time.sleep(0.25)
                    continue
        return False

    def _delete_project_data_safe(self, project_id: str, retries: int = 10) -> bool:
        """
        Purges precise object branches mapped in the storage backend.

        Args:
            project_id (str): The primary identifier marking the node for deletion.
            retries (int, optional): The upper iteration limit. Defaults to 10.

        Returns:
            bool: True mapping a successful deletion confirmation, False otherwise.
        """
        for i in range(retries):
            try:
                db = BeaverDB(settings.paths.db_file)
                projects = db.dict("dialog_projects")
                if project_id in projects:
                    del projects[project_id]
                return True
            except Exception:
                if i < retries - 1:
                    time.sleep(0.25)
                    continue
        return False

    def start_generation(self, project_id: str) -> bool:
        """
        Orchestrates the deployment of the background worker module utilizing absolute 
        process detachment.

        Forces complete closure of inherited file descriptors and establishes a new 
        OS session group. I/O streams are strictly routed to physical disk allocations 
        to guarantee the web framework cannot throttle the inference engine logs.

        Args:
            project_id (str): The designated UUID string corresponding to the synthesis request.

        Returns:
            bool: True validating immediate subprocess initiation, False if targeting a non-existent state.
        """
        data = self.get_project_data_safe(project_id)
        if not data:
            return False
            
        project = DialogProject(**data)
        current_status = str(project.status).lower()

        if current_status in [ProjectStatus.STARTING.value, ProjectStatus.GENERATING.value, "starting", "generating"]:
            if project.pid and self._is_pid_alive(project.pid):
                return True

        worker_script = os.path.abspath(os.path.join("src", "dialogs", "worker.py"))
        python_exe = sys.executable
        
        env = os.environ.copy()
        env["PYTHONPATH"] = os.getcwd()

        kwargs: Dict[str, Any] = {"env": env}
        if os.name == 'nt':
            kwargs['creationflags'] = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs['start_new_session'] = True

        log_path = os.path.abspath("worker_sys.log")
        log_file = open(log_path, "a")

        process = subprocess.Popen(
            [python_exe, worker_script, project_id],
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            close_fds=True,
            **kwargs
        )

        project.pid = process.pid
        project.status = ProjectStatus.STARTING
        self._write_project_data_safe(project_id, project.model_dump())
        
        return True
    
    def sync_status(self, project_id: str) -> Optional[DialogProject]:
        logger.debug(f"[ORCHESTRATOR] Syncing status for project: {project_id}")
        data = self.get_project_data_safe(project_id)
        if not data:
            logger.warning(f"[ORCHESTRATOR] Project data not found for {project_id}")
            return None
        
        project = DialogProject(**data)
        current_status = str(project.status).lower()
        
        if current_status in [ProjectStatus.STARTING.value, ProjectStatus.GENERATING.value, "starting", "generating"]:
            is_alive = self._is_pid_alive(project.pid)
            logger.debug(f"[ORCHESTRATOR] Worker PID {project.pid} is alive: {is_alive}")
            
            if not is_alive:
                logger.error(f"[ORCHESTRATOR] 🚨 FATAL: PID {project.pid} died unexpectedly while {current_status.upper()}!")
                project.status = ProjectStatus.FAILED
                data['status'] = ProjectStatus.FAILED.value
                data['error'] = "Worker process died unexpectedly without saving state (Possible GPU OOM)."
                self._write_project_data_safe(project_id, data)
        
        return project

    def _is_pid_alive(self, pid: Optional[int]) -> bool:
        """
        Determines execution viability interacting directly with kernel resource assignments.

        Args:
            pid (Optional[int]): The assigned kernel identifier pointing to the active execution cluster.

        Returns:
            bool: True ensuring system availability and continuous processing capacity, False otherwise.
        """
        if pid is None:
            return False
            
        try:
            process = psutil.Process(pid)
            return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            return False
        
    def pause_generation(self, project_id: str) -> bool:
        """
        Enacts graceful execution cessation by modifying persistent pipeline directives.

        Args:
            project_id (str): Targets the precise project map evaluating execution halt directives.

        Returns:
            bool: True detailing valid metadata injection of pause states, False otherwise.
        """
        data = self.get_project_data_safe(project_id)
        if not data:
            return False

        current_status = str(data.get('status', '')).lower()
        if current_status in [ProjectStatus.STARTING.value, ProjectStatus.GENERATING.value, "starting", "generating"]:
            data['status'] = ProjectStatus.PAUSED.value
            return self._write_project_data_safe(project_id, data)
            
        return False

    def cancel_generation(self, project_id: str) -> bool:
        """
        Forces ungraceful termination upon active process threads mapping designated project limits, 
        additionally forcing filesystem deletions for derived asset arrays.

        Args:
            project_id (str): Identifier targeting both logical mapping nodes and OS file allocations.

        Returns:
            bool: True upon full termination sequences clearing validation, False otherwise.
        """
        data = self.get_project_data_safe(project_id)
        if not data:
            return False

        data['status'] = ProjectStatus.CANCELLED.value
        self._write_project_data_safe(project_id, data)

        pid = data.get('pid')
        if pid and self._is_pid_alive(pid):
            timeout = 10
            start_wait = time.time()
            
            while self._is_pid_alive(pid) and (time.time() - start_wait < timeout):
                time.sleep(0.5)
            
            if self._is_pid_alive(pid):
                try:
                    process = psutil.Process(pid)
                    process.terminate()
                    process.wait(timeout=3)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

        data = self.get_project_data_safe(project_id)
        if not data:
            return False

        for state in data.get('states', []):
            state['status'] = LineStatus.PENDING.value
            state['audio_path'] = None
            state['error'] = None
        
        data['pid'] = None
        data['status'] = ProjectStatus.IDLE.value 
        self._write_project_data_safe(project_id, data)

        audio_dir = Path(data.get('project_path', '')) / "audio"
        if audio_dir.exists():
            shutil.rmtree(audio_dir, ignore_errors=True)
            audio_dir.mkdir(parents=True, exist_ok=True)

        return True

    def restart_generation(self, project_id: str) -> bool:
        """
        Triggers aggregate cleanup pipelines mapping cancel parameters prior to enacting 
        native launch procedures.

        Args:
            project_id (str): Core canonical sequence mapping.

        Returns:
            bool: True detailing sequential operational confirmation, False otherwise.
        """
        if self.cancel_generation(project_id):
            return self.start_generation(project_id)
        return False

    def delete_project(self, project_id: str) -> bool:
        """
        Initiates final destruction algorithms terminating relevant subprocesses before 
        annihilating both directory bindings and database logical maps.

        Args:
            project_id (str): Global identifier pointing to the data scope boundaries.

        Returns:
            bool: True detailing system-wide project map erasure, False otherwise.
        """
        data = self.get_project_data_safe(project_id)
        if not data:
            return False
            
        pid = data.get("pid")
        if pid and self._is_pid_alive(pid):
            try:
                psutil.Process(pid).terminate()
            except Exception:
                pass
                
        project_path = Path(data.get("project_path", ""))
        if project_path.exists():
            shutil.rmtree(project_path, ignore_errors=True)
            
        return self._delete_project_data_safe(project_id)
    
    
    def purge_project_assets(self, project_id: str) -> bool:
        """
        Annuls designated operational processes and clears filesystem derivations without 
        altering persistent schema attributes related to the root project map.

        Args:
            project_id (str): Core target mapping.

        Returns:
            bool: True highlighting successful directory purges, False corresponding to errors.
        """
        data = self.get_project_data_safe(project_id)
        if not data:
            return False
            
        pid = data.get("pid")
        if pid and self._is_pid_alive(pid):
            try:
                psutil.Process(pid).terminate()
            except Exception:
                pass
                
        project_path = Path(data.get("project_path", ""))
        audio_dir = project_path / "audio"
        
        if audio_dir.exists():
            shutil.rmtree(audio_dir, ignore_errors=True)
        
        db = BeaverDB(settings.paths.db_file)
        projects_dict = db.dict("dialog_projects")
        data["merged_audio_path"] = None
        projects_dict[project_id] = data
            
        return True
    
    
    def merge_project_audio(self, project_id: str) -> Optional[str]:
        """
        Orchestrates the transition from high-level project metadata to a 
        low-level audio timeline. 

        It maps DialogLine definitions into AudioSegmentConfig instances, 
        invokes the generic AudioEngine for tensor-based merging, and 
        persists the resulting master file path to the database.

        Args:
            project_id (str): The unique identifier of the project to merge.

        Returns:
            Optional[str]: Relative path to the merged master WAV, or None if failed.
        """
        data = self.get_project_data_safe(project_id)
        if not data:
            return None
            
        project = DialogProject(**data)
        project_root = Path(project.project_path)
        
        segments = []
        state_map = {s.line_id: s for s in project.states}
        
        for line in project.definition.script:
            state = state_map.get(line.id)
            if state and state.status == LineStatus.COMPLETED and state.audio_path:
                audio_full_path = project_root / state.audio_path
                if audio_full_path.exists():
                    segments.append(AudioSegmentConfig(
                        path=audio_full_path,
                        fade_in_ms=line.fade_in_ms,
                        fade_out_ms=line.fade_out_ms,
                        post_delay_ms=line.post_delay_ms,
                        room_tone_level=line.room_tone_level
                    ))

        if not segments:
            return None

        engine = AudioEngine(target_sample_rate=24000)
        output_dir = project_root / "audio"
        output_path = output_dir / "master_dialog.wav"
        
        success = engine.merge_segments(segments, output_path)
        
        if success:
            relative_master_path = str(output_path.relative_to(project_root))
            relative_mp3_path = str(output_path.with_suffix('.mp3').relative_to(project_root))
            
            db = BeaverDB(settings.paths.db_file)
            projects_dict = db.dict("dialog_projects")
            
            data["merged_audio_path"] = relative_master_path
            data["merged_mp3_path"] = relative_mp3_path
            projects_dict[project_id] = data
            
            return relative_master_path
            
        return None
    
    def add_project(self, project: DialogProject) -> str:
        """
        Registers a newly instantiated dialog project into the persistence layer.

        Delegates the physical database transaction to the internal safe-write 
        mechanism, ensuring atomic operations and transient failure recoveries.

        Args:
            project (DialogProject): The validated project entity.

        Returns:
            str: The canonical identifier assigned to the new project.
        """
        self._write_project_data_safe(project.id, project.model_dump())
        return project.id

    def update_project(self, project: DialogProject) -> bool:
        """
        Executes a destructive upsert operation on an existing dialog project.

        Leverages the internal asset purging mechanisms to strictly guarantee 
        the termination of any active OS-level processes and the complete 
        annihilation of legacy audio directories before committing the modified 
        script state to the database.

        Args:
            project (DialogProject): The mutated project entity containing the updated script.

        Returns:
            bool: True mapping a successful atomic commit, False otherwise.
        """
        self.purge_project_assets(project.id)
        return self._write_project_data_safe(project.id, project.model_dump())
    
    def get_all_projects(self, source: Optional[ProjectSource] = None, retries: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieves the complete project registry from the persistence layer with optional filtering.

        This method performs a thread-safe read of the global projects dictionary. 
        If a source discriminator is provided, it filters the result set to return 
        only projects matching the specified origin (UI or API), ensuring 
        consistency through transient retry logic.

        Args:
            source (Optional[ProjectSource]): The origin discriminator or None to retrieve all.
            retries (int): Evaluation threshold for concurrent access fallbacks.

        Returns:
            List[Dict[str, Any]]: A collection of raw project metadata payloads.
        """
        for i in range(retries):
            try:
                db = BeaverDB(settings.paths.db_file)
                projects_dict = db.dict("dialog_projects")
                
                all_data = [p_data for p_data in projects_dict.values() if p_data]
                
                if source:
                    return [p for p in all_data if p.get("source") == source.value]
                
                return all_data
            except Exception:
                if i < retries - 1:
                    time.sleep(0.25)
                    continue
        return []