import logging
from functools import lru_cache
from typing import Optional, Any

from src.config import settings, AppConfig
from src.backend.store import VoiceStore
from src.emotions.manager import EmotionManager

from beaver import BeaverDB
from src.dialogs.orchestrator import DialogOrchestrator

logger = logging.getLogger(__name__)

@lru_cache()
def get_config() -> AppConfig:
    """
    Retrieves the global application configuration as a singleton.

    This function uses lru_cache to ensure the configuration object is loaded 
    and parsed only once during the application lifecycle, preventing redundant 
    I/O operations.

    Returns:
        AppConfig: The validated application settings object.
    """
    return settings

@lru_cache()
def get_store() -> VoiceStore:
    """
    Provides a singleton instance of the VoiceStore (Database connection).

    The VoiceStore is initialized with the global settings. Since BeaverDB 
    handles connections efficiently, this initialization is lightweight 
    and safe to perform during the API startup phase.

    Returns:
        VoiceStore: The initialized persistence layer controller.
    """
    logger.debug("Initializing VoiceStore connection for API request.")
    return VoiceStore(settings)

@lru_cache()
def get_embed_engine() -> Optional[Any]:
    """
    Lazily initializes the Embedding Engine (Heavy Machine Learning Model).

    This function implements a 'Lazy Loading' strategy. The heavy ML libraries 
    (such as PyTorch, ONNX, or FastEmbed) and the model weights are NOT loaded 
    when the API starts. They are only loaded the first time an endpoint 
    specifically requests this dependency (e.g., /search).

    This approach drastically reduces the memory footprint and startup time 
    for users who only need to list or retrieve voices without performing 
    semantic searches.

    Returns:
        Optional[Any]: The initialized EmbeddingEngine instance if successful, 
        or None if the initialization fails due to missing libraries or 
        hardware constraints. The return type is 'Any' to avoid top-level 
        imports of the engine class.
    """
    try:
        logger.info("❄️ Cold Start: Loading Embedding Engine (ONNX/Torch)...")
        
        from src.backend.embedding import EmbeddingEngine, EmbeddingModelProvider
        
        provider = EmbeddingModelProvider(settings)
        engine = EmbeddingEngine(settings, provider)
        
        logger.info("✅ Embedding Engine ready.")
        return engine
        
    except ImportError as e:
        logger.critical(f"Failed to import embedding backend libraries. Ensure dependencies are installed: {e}")
        return None
    except Exception as e:
        logger.critical(f"Failed to initialize Embedding Model during cold start: {e}")
        return None
    
@lru_cache()
def get_emotion_manager() -> EmotionManager:
    """
    Provides a singleton instance of the EmotionManager.
    
    Returns:
        EmotionManager: The initialized emotion catalog controller.
    """
    logger.debug("Initializing EmotionManager for API request.")
    return EmotionManager(settings)

@lru_cache()
def get_tts_provider() -> Optional[Any]:
    """
    Lazily initializes the heavy TTS Model Provider.
    Loaded only on the first synthesis request to save memory during API startup.
    
    Returns:
        Optional[Any]: The initialized TTSModelProvider instance or None if it fails.
    """
    try:
        logger.info("❄️ Cold Start: Loading TTS Provider...")
        
        from src.backend.engine import TTSModelProvider
        
        provider = TTSModelProvider(settings)
        
        logger.info("✅ TTS Provider ready.")
        return provider
        
    except ImportError as e:
        logger.critical(f"Failed to import TTS backend libraries: {e}")
        return None
    except Exception as e:
        logger.critical(f"Failed to initialize TTS Model during cold start: {e}")
        return None
    
@lru_cache()
def get_orchestrator() -> DialogOrchestrator:
    """
    Provides a stateless singleton instance of the DialogOrchestrator.

    The orchestrator manages absolute process detachment and OS-level subprocess 
    lifecycle for asynchronous audio synthesis tasks. Caching this instance ensures 
    that the same validation and process-hunting logical boundaries are applied 
    uniformly across all execution control endpoints.

    Returns:
        DialogOrchestrator: The service responsible for background worker delegation.
    """
    logger.debug("Initializing DialogOrchestrator for API request.")
    return DialogOrchestrator()