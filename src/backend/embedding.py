import os
from typing import List
from fastembed import TextEmbedding
from src.models import AppConfig

class EmbeddingModelProvider:
    """
    Singleton resource manager for the FastEmbed text embedding model.
    
    This provider ensures the heavy ONNX model is loaded only once per application 
    lifetime. It enforces a stable local cache directory to prevent ONNX Runtime 
    path validation errors, which occur when model weights (.onnx_data) are stored 
    in volatile or restricted system paths like /tmp/.
    """
    _model = None

    def __init__(self, config: AppConfig):
        """
        Initializes the provider with the global configuration.

        Args:
            config (AppConfig): Global application configuration object.
        """
        self.config = config

    def get_model(self) -> TextEmbedding:
        """
        Retrieves or initializes the TextEmbedding model instance.
        
        Configures an explicit local cache directory. ONNX models with external 
        data tensors require strict relative path integrity; a local project path 
        ensures the engine can always locate its weight data without escaping 
        security boundaries.

        Returns:
            TextEmbedding: The instantiated and ready-to-use FastEmbed model.
        """
        if EmbeddingModelProvider._model is None:
            model_id = self.config.models.semantic.repo_id
            use_cuda = "cuda" in self.config.system.compute.embed_device
            
            # CRITICAL FIX: Absolute local path to maintain ONNX data integrity.
            # Fixes "External data path ... escapes model directory" error.
            cache_path = os.path.abspath(os.path.join(os.getcwd(), "data", "models_cache"))
            os.makedirs(cache_path, exist_ok=True)
            
            EmbeddingModelProvider._model = TextEmbedding(
                model_name=model_id,
                cuda=use_cuda,
                cache_dir=cache_path
            )
        return EmbeddingModelProvider._model

class EmbeddingEngine:
    """
    Engine responsible for transforming raw text into normalized vector representations.
    """
    def __init__(self, config: AppConfig, provider: EmbeddingModelProvider):
        """
        Initializes the engine with its corresponding singleton provider.

        Args:
            config (AppConfig): Global application configuration.
            provider (EmbeddingModelProvider): The shared model provider instance.
        """
        self.config = config
        self.provider = provider

    def generate_embedding(self, text: str) -> List[float]:
        """
        Transforms an input string into a dense semantic vector.

        Args:
            text (str): The input text to be processed.

        Returns:
            List[float]: A list of floats representing the semantic embedding.

        Raises:
            ValueError: If the input text is empty or None.
        """
        if not text:
            raise ValueError("Inference failed: Input text for embedding cannot be empty.")
        
        model = self.provider.get_model()
        # embed() returns a generator of numpy arrays. We take the first one.
        generator = model.embed([text])
        vector = next(generator)
        return vector.tolist()