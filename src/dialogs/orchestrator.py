"""
Dialog Orchestrator Module.

Manages the lifecycle of detached synthesis processes. 
Ensures deterministic PID tracking and physical log file generation 
for standard errors (stderr) to prevent silent failures.
"""

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
    Service for spawning and supervising background synthesis workers.
    """

    def __init__(self):
        """Initializes the orchestrator with the shared database dictionary."""
        self.db: BeaverDB = BeaverDB(settings.paths.db_file)
        self.projects_dict: dict = self.db.dict("dialog_projects")

    def start_generation(self, project_id: str) -> bool:
        """
        Launches a detached worker and forces OS-level error logging.

        Args:
            project_id (str): The target project UUID.

        Returns:
            bool: True if the process was launched successfully.
        """
        data: Optional[dict] = self.projects_dict.get(project_id)
        if not data:
            return False
            
        project: DialogProject = DialogProject(**data)

        # Check for existing healthy processes
        if project.status in [ProjectStatus.STARTING, ProjectStatus.GENERATING]:
            if project.pid and self._is_pid_alive(project.pid):
                return True

        worker_script: str = os.path.abspath(os.path.join("src", "dialogs", "worker.py"))
        python_exe: str = sys.executable
        
        # Environment inheritance for proper relative imports
        env = os.environ.copy()
        env["PYTHONPATH"] = os.getcwd()

        kwargs: dict = {"env": env}
        if os.name == 'nt':
            kwargs['creationflags'] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
        else:
            kwargs['start_new_session'] = True

        # OS-LEVEL TRACEABILITY: Capture raw stdout/stderr
        stderr_log = open("worker_stderr.log", "w")

        process = subprocess.Popen(
            [python_exe, worker_script, project_id],
            stdout=stderr_log,
            stderr=subprocess.STDOUT,
            close_fds=True,
            **kwargs
        )

        # DETERMINISTIC REGISTRATION
        project.pid = process.pid
        project.status = ProjectStatus.STARTING
        self.projects_dict[project_id] = project.model_dump()
        return True

    def sync_status(self, project_id: str) -> Optional[DialogProject]:
        """
        Validates the actual OS-level health of the registered process.
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
        Checks if a process is active and not in a ZOMBIE state.
        """
        if pid is None:
            return False
        try:
            p = psutil.Process(pid)
            return p.is_running() and p.status() != psutil.STATUS_ZOMBIE
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return False