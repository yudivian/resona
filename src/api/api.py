import logging
from fastapi import FastAPI
from contextlib import asynccontextmanager

from src.config import settings
from src.api.routes import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger("resona.api")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the application lifecycle events (Startup and Shutdown).
    
    This context manager is the modern FastAPI way to handle initialization logic.
    It logs the system state before the API starts accepting requests and performs
    cleanup (if any) after the API stops.
    """
    logger.info(f"🚀 Resona API v{settings.system.version} is starting up...")
    logger.info(f"🔧 Compute Device Configured: {settings.system.compute.tts_device}")
    
    yield
    
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