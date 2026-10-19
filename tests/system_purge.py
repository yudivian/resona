import sys
import shutil
import logging
from pathlib import Path
from beaver import BeaverDB

sys.path.append(str(Path(__file__).parent.parent))
from src.config import settings

logging.basicConfig(level=logging.INFO, format='%(message)s')

def main() -> None:
    db = BeaverDB(settings.paths.db_file)
    projects_dict = db.dict("dialog_projects")
    
    # Purgar todos los proyectos de test
    targets = [pid for pid in projects_dict.keys() if pid.startswith("stress_test_")]
    
    if not targets:
        logging.info("No test artifacts to purge.")
        return

    for pid in targets:
        logging.info(f"Purging {pid}...")
        del projects_dict[pid]
        path = Path(settings.paths.dialogspace_dir) / pid
        if path.exists():
            shutil.rmtree(path)
            
    # También borramos el log de traza para la siguiente sesión
    log_file = Path("worker_trace.log")
    if log_file.exists():
        log_file.unlink()

    logging.info("System purge complete.")

if __name__ == "__main__":
    main()