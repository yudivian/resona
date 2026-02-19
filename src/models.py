import time
import uuid
from typing import List, Dict, Any, Optional
from enum import Enum
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
    tunespace_dir: str

class FineTuneConfig(BaseModel):
    tune_repo_id: str
    max_concurrent_jobs: int = 1
    checkpoints_keep_limit: int = 3

class TTSModelConfig(BaseModel):
    """
    Settings for the Qwen-TTS model.
    """
    design_repo_id: str
    base_repo_id: str
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
    finetune: FineTuneConfig
    semantic: SemanticModelConfig

class AppConfig(BaseModel):
    """
    Root configuration object for the application.
    """
    system: SystemConfig
    paths: PathsConfig
    models: ModelsConfig

class SourceType(str, Enum):
    CLONE = "clone"
    DESIGN = "design"
    BLEND = "blend"

class TrackType(str, Enum):
    CLONE = "clone"
    DESIGN = "design"
    PREEXISTING = "preexisting"

class VoiceProfile(BaseModel):
    model_config = ConfigDict(extra="ignore", use_enum_values=True)

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    created_at: float = Field(default_factory=time.time)
    
    identity_embedding: List[float]
    seed: int
    language: str

    source_type: SourceType
    track_a_type: Optional[TrackType] = None
    track_b_type: Optional[TrackType] = None
    
    is_refined: bool = False
    refinement_prompt: Optional[str] = None
    source_prompts: Dict[str, str] = Field(default_factory=dict)

    description: Optional[str] = None
    semantic_embedding: Optional[List[float]] = None
    tags: List[str] = Field(default_factory=list)

    anchor_audio_path: Optional[str] = None
    anchor_text: Optional[str] = None
    
    parameters: Dict[str, Any] = Field(default_factory=dict)
    
class VoiceEndpoints(BaseModel):
    audio: str
    bundle: str
    text: str
class VoiceSummary(BaseModel):
    id: str
    name: str
    language: str
    tags: List[str]
    source_type: str
    created_at: float
    description: Optional[str] = None
    preview_url: str 
    score: Optional[float] = None

class VoiceDetail(BaseModel):
    id: str
    name: str
    description: Optional[str]
    anchor_text: Optional[str]
    language: str
    tags: List[str]
    source_type: str
    created_at: float
    params: Dict[str, Any]
    endpoints: VoiceEndpoints