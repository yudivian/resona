from typing import List
import numpy as np
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
        Initializes the TextEmbedding model using configuration parameters.
        No custom registration is used to avoid signature conflicts with official models.
        """
        if EmbeddingModelProvider._model is None:
            # Extraemos el repo_id de config.yaml (ej: "sentence-transformers/...")
            model_id = self.config.models.semantic.repo_id
            # Determinamos si usar CUDA basado en tu configuración de sistema
            use_cuda = "cuda" in self.config.system.compute.embed_device
            
            # Instanciación estándar para modelos existentes
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
        self.provider = provider

    def generate_embedding(self, text: str) -> List[float]:
        """
        Converts text into a dense vector using the loaded model.
        """
        if not text:
            raise ValueError("Input text cannot be empty")
        
        model = self.provider.get_model()
        
        # embed() devuelve un iterador de numpy arrays
        generator = model.embed([text])
        vector = next(generator)
        
        # Convertimos a lista para compatibilidad con el resto del backend
        return vector.tolist()