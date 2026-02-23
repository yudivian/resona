"""
Dialog Worker Module (Extreme Traceability).

This script executes the AI inference pipeline as a detached process.
It implements a global try/except block to catch and log any fatal 
initialization or runtime errors, ensuring the Orchestrator is always notified.
"""

import sys
import os
import time
import logging
import traceback

# 1. IMMEDIATE TELEMETRY: Setup logging before importing heavy libraries
log_path = os.path.abspath("worker_trace.log")
logging.basicConfig(
    filename=log_path,
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] - %(message)s',
    force=True # Force handler overwrite if already set
)

logging.info(f"====== STARTING WORKER (PID: {os.getpid()}) ======")
logging.info(f"Current PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not Set')}")

def main() -> None:
    try:
        # 2. CONTROLLED IMPORTS: Catch missing modules
        logging.info("Importing project dependencies...")
        from beaver import BeaverDB
        from src.config import settings
        from src.models import DialogProject, ProjectStatus, LineStatus
        
        project_id = sys.argv[1]
        logging.info(f"Connecting to BeaverDB for project: {project_id}")
        db = BeaverDB(settings.paths.db_file)
        projects_dict = db.dict("dialog_projects")
        
        data = projects_dict.get(project_id)
        if not data:
            logging.error(f"Project {project_id} not found in BeaverDB.")
            return
            
        project = DialogProject(**data)
        logging.info("Project loaded successfully. Initializing heavy dependencies...")

        # 3. HEAVY LOAD INITIALIZATION
        from src.backend.store import VoiceStore
        from src.emotions.manager import EmotionManager
        from src.dialogs.resolver import DialogResolver
        from src.dialogs.cluster import DialogClusterer

        logging.info("Instantiating VoiceStore and EmotionManager...")
        # Aligned with store.py: Requires the full 'settings' AppConfig object
        voice_store = VoiceStore(settings)
        emotion_manager = EmotionManager(settings.paths.emotionspace_dir)
        
        resolver = DialogResolver(voice_store, emotion_manager)
        clusterer = DialogClusterer()

        logging.info("Resolving script dependencies...")
        resolver.resolve_project(project)
        clusters = clusterer.build_clusters(project)
        
        logging.info(f"Generated {len(clusters)} clusters. Transitioning to GENERATING status.")
        project.status = ProjectStatus.GENERATING
        projects_dict[project_id] = project.model_dump()

        # 4. INFERENCE LOOP
        for i, cluster in enumerate(clusters):
            logging.info(f"Processing Cluster {i+1}/{len(clusters)} with {len(cluster.state_ids)} audio lines.")
            
            # Polling for external interrupts
            fresh = projects_dict.get(project_id, {})
            if fresh.get('status') == ProjectStatus.PAUSED:
                logging.warning("PAUSE signal received. Aborting cleanly to release GPU.")
                return 

            # GPU Inference Simulation
            time.sleep(1.0) 
            
            # State mapping via UUID
            for state_id in cluster.state_ids:
                for s in project.states:
                    if s.id == state_id:
                        s.status = LineStatus.COMPLETED
                        s.audio_path = f"audio/{state_id}.wav"
                        logging.debug(f"Audio generated successfully: {s.audio_path}")
            
            # Heartbeat update
            project.updated_at = time.time()
            projects_dict[project_id] = project.model_dump()

        logging.info("====== PROJECT COMPLETED SUCCESSFULLY ======")
        project.status = ProjectStatus.COMPLETED
        projects_dict[project_id] = project.model_dump()

    except Exception as e:
        # 5. FATAL CATCH: Log the full traceback and update the database
        error_msg = traceback.format_exc()
        logging.error(f"FATAL ERROR IN WORKER:\n{error_msg}")
        
        try:
            # Attempt to signal the orchestrator via the database
            project.status = ProjectStatus.FAILED
            project.error = str(e)
            projects_dict[project_id] = project.model_dump()
            logging.info("FAILED status safely saved to DB.")
        except Exception as db_err:
            logging.error(f"Failed to save FAILED status to DB: {db_err}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        logging.error("No project_id provided as CLI argument.")
        sys.exit(1)
    
    # Wrap the main call to prevent silent interpreter exits
    try:
        main()
    except Exception as fatal:
        logging.critical(f"INTERPRETER CRASH: {traceback.format_exc()}")