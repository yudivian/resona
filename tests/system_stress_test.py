"""
Standalone stress testing module for the dialog generation orchestrator.

This script simulates a highly aggressive production workload by forcing the 
background worker to handle context switching between different voice profiles, 
languages, and emotional modifiers across sequential clusters, aiming to expose 
potential VRAM leaks or context-switching deadlocks.
"""

import sys
import os
import time
import uuid
import logging
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))

from src.config import settings
from src.models import DialogProject, DialogScript, DialogLine, LineState, ProjectStatus, LineStatus
from src.backend.store import VoiceStore
from src.dialogs.orchestrator import DialogOrchestrator
from beaver import BeaverDB

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] - %(message)s')

def main() -> None:
    """
    Executes the multi-context stress test for the synthesis pipeline.

    Retrieves available voices, constructs a multi-cluster dialog project with 
    alternating languages and voices, dispatches it via the orchestrator, and 
    monitors the execution state until completion or failure.

    Returns:
        None
    """
    logging.info("--- INITIATING AGGRESSIVE CONTEXT-SWITCH STRESS TEST ---")
    
    db = BeaverDB(settings.paths.db_file)
    projects_dict = db.dict("dialog_projects")
    
    store = VoiceStore(settings)
    profiles = store.get_all()
    
    if not profiles:
        logging.error("CRITICAL: No voices found in DB. Test aborted.")
        return
        
    voice_1 = profiles[0].id
    voice_2 = profiles[1].id if len(profiles) > 1 else profiles[0].id

    if voice_1 == voice_2:
        logging.warning("Only 1 voice found. Test will switch languages but use the same voice.")
    else:
        logging.info(f"Context Switch Mode ACTIVE: Voice A ({voice_1}) -> Voice B ({voice_2})")
    
    project_id = f"stress_test_ctx_{uuid.uuid4().hex[:6]}"
    project_path = ROOT_DIR / settings.paths.dialogspace_dir / project_id
    log_file = Path("worker_trace.log")

    if log_file.exists(): 
        log_file.unlink()
    
    lines = []
    
    for i in range(5):
        lines.append(DialogLine(
            id=f"L{i}", index=i, voice_id=voice_1, 
            text=f"Línea {i} del primer bloque en Español.", language="es"
        ))
        
    for i in range(5, 8):
        lines.append(DialogLine(
            id=f"L{i}", index=i, voice_id=voice_2, 
            text=f"Line {i} of the second block in English.", language="en"
        ))
        
    for i in range(8, 10):
        lines.append(DialogLine(
            id=f"L{i}", index=i, voice_id=voice_1, 
            text=f"Línea {i} del tercer bloque, con emoción.", language="es", emotion="happy"
        ))

    project = DialogProject(
        id=project_id,
        definition=DialogScript(name="Context Switch Stress Test", script=lines),
        project_path=str(project_path),
        states=[
            LineState(id=f"S{l.index}", line_id=l.id, index=l.index, status=LineStatus.PENDING)
            for l in lines
        ]
    )

    projects_dict[project_id] = project.model_dump()
    
    orchestrator = DialogOrchestrator()
    logging.info(f"Deploying worker for project {project_id}...")
    
    if not orchestrator.start_generation(project_id):
        logging.error("Orchestrator failed to spawn worker process.")
        return

    start_time = time.time()
    timeout = 180 
    
    while time.time() - start_time < timeout:
        p = orchestrator.sync_status(project_id)
        if not p: 
            break

        status_str = str(p.status).lower()
        completed = sum(1 for s in p.states if "completed" in str(s.status).lower())
        
        logging.info(f" [PROGRESS] {status_str} | Completed: {completed}/10")

        if "completed" in status_str:
            logging.info("SUCCESS! The system survived all context switches.")
            break
            
        if "failed" in status_str:
            logging.error("CRITICAL FAILURE DETECTED. Worker crashed during context switch.")
            if log_file.exists():
                print(log_file.read_text())
            break
            
        time.sleep(3)

if __name__ == "__main__":
    main()