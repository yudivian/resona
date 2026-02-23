import argparse
import sys
import subprocess
import logging
import uvicorn
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [CLI] - %(message)s")
logger = logging.getLogger(__name__)

def run_studio(args):
    """
    Launches the Streamlit User Interface (Resona Studio).
    
    This function locates the 'app.py' entry point relative to this script
    and executes it using the Streamlit CLI via a subprocess. This isolation
    ensures that Streamlit's internal loop does not conflict with the CLI wrapper.
    """
    current_dir = Path(__file__).parent
    app_path = current_dir / "app.py"
    
    if not app_path.exists():
        logger.error(f"Could not find Studio entry point at: {app_path}")
        sys.exit(1)
        
    logger.info(f"🎨 Launching Resona Studio from: {app_path}")
    
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path)]
    
    if args.debug:
        logger.setLevel(logging.DEBUG)
        
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        logger.info("🛑 Studio stopped by user.")
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Studio crashed with error code {e.returncode}")
        sys.exit(e.returncode)

def run_emotions(args):
    """
    Launches the Streamlit User Interface for Emotion CAD (app_emotions.py).
    
    This command isolates the Emotion CAD tool execution.
    """
    current_dir = Path(__file__).parent
    app_path = current_dir / "app_emotions.py"
    
    if not app_path.exists():
        logger.error(f"Could not find Emotion CAD entry point at: {app_path}")
        sys.exit(1)
        
    logger.info(f"🎭 Launching Resona Emotion CAD from: {app_path}")
    
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path)]
    
    if args.debug:
        logger.setLevel(logging.DEBUG)
        
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        logger.info("🛑 Emotion CAD stopped by user.")
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Emotion CAD crashed with error code {e.returncode}")
        sys.exit(e.returncode)

def run_voice_api(args):
    """
    Starts the Core Voice Discovery & Asset API.
    
    This command initializes the FastAPI application defined in 'src.api.api'.
    It uses Uvicorn as the ASGI server production runner.
    """
    logger.info(f"🎙️ Starting Resona Voice API on {args.host}:{args.port}")
    if args.reload:
        logger.warning("⚠️ Hot-Reload enabled. Do not use in production.")

    uvicorn.run(
        "src.api.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info" if not args.debug else "debug"
    )

def main():
    """
    Main entry point for the Resona CLI.
    Parses arguments and dispatches control to the appropriate sub-command handler.
    """
    parser = argparse.ArgumentParser(
        description="Resona Infrastructure CLI - Manage Studio, Emotion CAD, and API services.",
        prog="resona"
    )
    
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.required = True

    studio_parser = subparsers.add_parser(
        "studio", 
        help="Launch the visual design interface (Streamlit)"
    )
    studio_parser.set_defaults(func=run_studio)

    emotions_parser = subparsers.add_parser(
        "emotions", 
        help="Launch the Emotion CAD visual interface (Streamlit)"
    )
    emotions_parser.set_defaults(func=run_emotions)

    voice_api_parser = subparsers.add_parser(
        "voice-api", 
        help="Start the Core Voice Discovery & Asset API (FastAPI)"
    )
    voice_api_parser.add_argument(
        "--host", 
        type=str, 
        default="0.0.0.0", 
        help="Bind socket to this host (default: 0.0.0.0)"
    )
    voice_api_parser.add_argument(
        "--port", 
        type=int, 
        default=8000, 
        help="Bind socket to this port (default: 8000)"
    )
    voice_api_parser.add_argument(
        "--reload", 
        action="store_true", 
        help="Enable auto-reload for development"
    )
    voice_api_parser.set_defaults(func=run_voice_api)

    args = parser.parse_args()

    try:
        args.func(args)
    except KeyboardInterrupt:
        logger.info("\n👋 Resona CLI terminated.")
        sys.exit(0)

if __name__ == "__main__":
    main()