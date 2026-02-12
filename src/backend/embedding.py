from typing import List
from fastembed import TextEmbedding
from src.models import AppConfig

class EmbeddingModelProvider:
    """
    Singleton resource manager for the FastEmbed model.
    """
    _model = None

    def __init__(self, config: AppConfig):
        self.config = config

    def get_model(self) -> TextEmbedding:
        """
        Initializes the TextEmbedding model using configuration-driven parameters.
        Uses dynamic model registration to silence pooling warnings without hardcoding.
        """
        if EmbeddingModelProvider._model is None:
            model_id = self.config.models.semantic.repo_id
            use_cuda = "cuda" in self.config.system.compute.embed_device
            
            # Obtenemos los metadatos que la librería ya conoce para este repo_id
            supported_models = TextEmbedding.list_supported_models()
            model_info = next((m for m in supported_models if m["model"] == model_id), None)
            
            # Si el modelo es reconocido, lo registramos formalmente para evitar el warning de pooling.
            # No se añade ningún parámetro que no provenga de la propia definición de la librería.
            if model_info:
                TextEmbedding.add_custom_model(**model_info)

            EmbeddingModelProvider._model = TextEmbedding(
                model_name=model_id,
                cuda=use_cuda
            )
        return EmbeddingModelProvider._model

class EmbeddingEngine:
    """
    Worker engine that transforms text into normalized vector representations.
    """
    def __init__(self, config: AppConfig, provider: EmbeddingModelProvider):
        self.config = config
        self.model = provider.get_model()

    def generate_embedding(self, text: str) -> List[float]:
        """
        Converts a text description into a dense vector using the loaded model.
        """
        if not text:
            raise ValueError("Input text cannot be empty")
        
        generator = self.model.embed([text])
        vector = list(generator)[0]
        
        return vector.tolist()