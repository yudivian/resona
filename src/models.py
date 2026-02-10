import time
import uuid
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, ConfigDict

class ComputeConfig(BaseModel):
    """
    Configuration for hardware acceleration and precision.
    """
    tts_device: str
    embed_device: str
    precision: str

class SystemConfig(BaseModel):
    """
    General system metadata.
    """
    name: str
    version: str
    compute: ComputeConfig

class PathsConfig(BaseModel):
    """
    File system paths for storage.
    """
    db_file: str
    assets_dir: str
    temp_dir: str

class TTSModelConfig(BaseModel):
    """
    Settings for the Qwen-TTS model.
    """
    repo_id: str
    sample_rate: int

class SemanticModelConfig(BaseModel):
    """
    Settings for the semantic search model.
    """
    repo_id: str

class ModelsConfig(BaseModel):
    """
    Aggregated model configurations.
    """
    tts: TTSModelConfig
    semantic: SemanticModelConfig

class AppConfig(BaseModel):
    """
    Root configuration object for the application.
    """
    system: SystemConfig
    paths: PathsConfig
    models: ModelsConfig

class VoiceProfile(BaseModel):
    """
    Represents a synthetic voice identity in the CAD system.
    
    This model serves as the single source of truth for a voice design.
    The identity is primarily defined by the `identity_embedding` vector.
    """
    model_config = ConfigDict(extra="ignore")

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    created_at: float = Field(default_factory=time.time)
    
    identity_embedding: List[float]
    
    semantic_embedding: Optional[List[float]] = None
    
    base_language: str = "en"
    
    anchor_audio_path: Optional[str] = None
    
    parameters: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
    description: Optional[str] = None