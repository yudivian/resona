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
        Initializes the TextEmbedding model using ONNX Runtime.
        """
        if EmbeddingModelProvider._model is None:
            use_cuda = "cuda" in self.config.system.compute.embed_device
            
            EmbeddingModelProvider._model = TextEmbedding(
                model_name=self.config.models.semantic.repo_id,
                threads=4,
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