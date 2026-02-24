import subprocess
import os
import sys
import psutil
from typing import Optional
from beaver import BeaverDB

from src.config import settings
from src.models import DialogProject, ProjectStatus

class DialogOrchestrator:
    """
    Service responsible for the lifecycle management and supervision of detached 
    background synthesis processes.

    The orchestrator acts as the authoritative bridge between the user interface 
    and the operating system's process scheduler. It ensures that synthesis workloads 
    are launched in complete isolation (preventing API blocking) while maintaining 
    deterministic tracking of their execution health via process IDs (PIDs) and 
    the shared BeaverDB state.
    """

    def __init__(self) -> None:
        """
        Initializes the DialogOrchestrator and establishes a connection to the 
        persistent state database.
        """
        self.db: BeaverDB = BeaverDB(settings.paths.db_file)
        self.projects_dict: dict = self.db.dict("dialog_projects")

    def start_generation(self, project_id: str) -> bool:
        """
        Deploys a new detached background worker to process the specified dialog project.

        This method configures an isolated execution environment, ensuring the Python 
        path is correctly inherited. It launches the worker script as a fully detached 
        process group, capturing its PID immediately upon creation. If a healthy process 
        is already actively generating the project, the launch is safely bypassed.

        Args:
            project_id (str): The unique identifier of the target project to synthesize.

        Returns:
            bool: True if a worker was successfully deployed or is already running natively.
                  False if the project payload could not be located in the database.
        """
        data: Optional[dict] = self.projects_dict.get(project_id)
        if not data:
            return False
            
        project: DialogProject = DialogProject(**data)

        if project.status in [ProjectStatus.STARTING, ProjectStatus.GENERATING]:
            if project.pid and self._is_pid_alive(project.pid):
                return True

        worker_script: str = os.path.abspath(os.path.join("src", "dialogs", "worker.py"))
        python_exe: str = sys.executable
        
        env = os.environ.copy()
        env["PYTHONPATH"] = os.getcwd()

        kwargs: dict = {"env": env}
        if os.name == 'nt':
            kwargs['creationflags'] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
        else:
            kwargs['start_new_session'] = True

        process = subprocess.Popen(
            [python_exe, worker_script, project_id],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
            **kwargs
        )

        project.pid = process.pid
        project.status = ProjectStatus.STARTING
        self.projects_dict[project_id] = project.model_dump()
        
        return True

    def sync_status(self, project_id: str) -> Optional[DialogProject]:
        """
        Synchronizes the logical database state of a project with the physical reality 
        of the operating system's process table.

        If the database asserts that a project is actively generating, but the underlying 
        process PID is dead, missing, or in a zombie state, this method performs auto-healing 
        by permanently marking the project as FAILED, thus preventing infinite UI loading states.

        Args:
            project_id (str): The unique identifier of the project to synchronize.

        Returns:
            Optional[DialogProject]: The updated project instance, or None if it does not exist.
        """
        data: Optional[dict] = self.projects_dict.get(project_id)
        if not data:
            return None
        
        project: DialogProject = DialogProject(**data)
        
        if project.status in [ProjectStatus.STARTING, ProjectStatus.GENERATING]:
            if not self._is_pid_alive(project.pid):
                project.status = ProjectStatus.FAILED
                data['status'] = ProjectStatus.FAILED
                self.projects_dict[project_id] = data
        
        return project

    def _is_pid_alive(self, pid: Optional[int]) -> bool:
        """
        Interrogates the operating system kernel to determine the true health of a process.

        Unlike naive existence checks, this method strictly validates that the process 
        is not only present in the process table but is actively running and has not 
        degraded into a Zombie state following an unreported crash or termination.

        Args:
            pid (Optional[int]): The process identifier to investigate.

        Returns:
            bool: True if the process is physically alive and healthy, False otherwise.
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
        Suspends the active dialog generation process gracefully.

        This method signals the background worker to halt execution by transitioning
        the project's global state to a paused status. It relies on the worker's
        internal state-checking mechanism between cluster inference cycles to release
        GPU resources and terminate its process safely, ensuring no data corruption
        occurs in partially generated audio files.

        Args:
            project_id (str): The unique identifier of the dialog project.

        Returns:
            bool: True if the state was successfully transitioned to paused,
                  False if the project was not found or was not in an active state.
        """
        data = self.projects_dict.get(project_id)
        if not data:
            return False

        current_status = data.get('status')
        if current_status in [ProjectStatus.STARTING.value, ProjectStatus.GENERATING.value, "starting", "generating"]:
            data['status'] = ProjectStatus.PAUSED.value
            self.projects_dict[project_id] = data
            return True
            
        return False

    def cancel_generation(self, project_id: str) -> bool:
        """
        Terminates the active generation process and performs a complete project reset.

        This method enforces a hard stop on the background worker by setting the
        project status to cancelled. It implements an active wait mechanism to allow
        the worker to perform graceful resource deallocation (e.g., VRAM flushing).
        If the worker fails to terminate within the designated timeout window, it
        escalates to an operating system-level process termination to prevent zombie
        processes. Subsequently, it resets all internal line states, clears audio
        path references, and physically purges the associated audio directory to
        restore the project to its pristine initial state.

        Args:
            project_id (str): The unique identifier of the dialog project.

        Returns:
            bool: True if the project was successfully cancelled and purged,
                  False if the project identifier could not be located.
        """
        data = self.projects_dict.get(project_id)
        if not data:
            return False

        data['status'] = ProjectStatus.CANCELLED.value
        self.projects_dict[project_id] = data

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

        data = self.projects_dict.get(project_id)

        for state in data.get('states', []):
            state['status'] = LineStatus.PENDING.value
            state['audio_path'] = None
            state['error'] = None
        
        data['pid'] = None
        data['status'] = ProjectStatus.PENDING.value
        self.projects_dict[project_id] = data

        audio_dir = Path(data.get('project_path')) / "audio"
        if audio_dir.exists():
            shutil.rmtree(audio_dir, ignore_errors=True)
            audio_dir.mkdir(parents=True, exist_ok=True)

        return True

    def restart_generation(self, project_id: str) -> bool:
        """
        Executes a complete lifecycle reset and subsequently restarts generation.

        This composite method strictly chains the cancellation and starting protocols.
        It guarantees that any residual artifacts, compromised states, or active
        subprocesses are entirely eradicated before spawning a fresh background
        worker, providing a clean execution environment for the project.

        Args:
            project_id (str): The unique identifier of the dialog project.

        Returns:
            bool: True if the project was successfully reset and restarted,
                  False if the initialization sequence failed at any stage.
        """
        if self.cancel_generation(project_id):
            return self.start_generation(project_id)
        return False

    def delete_project(self, project_id: str) -> bool:
        """
        Eradicates the project entirely from both the storage backend and filesystem.

        This method serves as the final destructor for a dialog project. It intercepts
        and terminates any active background workers associated with the project PID
        to ensure no file handles remain locked. It then recursively purges the entire
        project directory from the local filesystem and deletes the project's state
        record from the persistent database.

        Args:
            project_id (str): The unique identifier of the dialog project.

        Returns:
            bool: True if the project was successfully deleted, False if the
                  project did not exist within the database.
        """
        data = self.projects_dict.get(project_id)
        if not data:
            return False
            
        pid = data.get("pid")
        if pid and self._is_pid_alive(pid):
            try:
                psutil.Process(pid).terminate()
            except Exception:
                pass
                
        project_path = Path(data.get("project_path"))
        if project_path.exists():
            shutil.rmtree(project_path, ignore_errors=True)
            
        del self.projects_dict[project_id]
        return True