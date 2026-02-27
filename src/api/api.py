import logging
from fastapi import FastAPI
from contextlib import asynccontextmanager

from src.config import settings
from src.api.routes import router
from apscheduler.schedulers.background import BackgroundScheduler
from src.dialogs.orchestrator import DialogOrchestrator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger("resona.api")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the application lifecycle and background maintenance tasks.
    """
    logger.info(f"🚀 Resona API v{settings.system.version} is starting up...")
    
    scheduler = BackgroundScheduler()
    orchestrator = DialogOrchestrator()
    
    interval = settings.system.cleanup_interval_minutes
    
    scheduler.add_job(
        orchestrator.cleanup_expired_api_projects, 
        "interval", 
        minutes=interval,
        id="api_cleanup_task"
    )
    
    scheduler.start()
    logger.info("🧹 API Project Cleanup Scheduler initiated (30m interval).")
    
    yield
    
    scheduler.shutdown()
    logger.info("🛑 Resona API is shutting down.")

app = FastAPI(
    title="Resona Voice API",
    description="Voice Asset Management and Discovery Interface",
    version=settings.system.version,
    lifespan=lifespan
)

app.include_router(router, prefix="/v1", tags=["Voices"])

@app.get("/health")
def health_check():
    """
    Simple health check endpoint for monitoring uptime.
    
    Returns:
        dict: The operational status and current version of the system.
    """
    return {
        "status": "online", 
        "version": settings.system.version
    }