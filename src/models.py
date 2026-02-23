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
    tune_device: str
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
    emotionspace_dir: str
    apispace_dir: str

class FineTuneConfig(BaseModel):
    """
    Settings related to the fine-tuning operations and orchestration.
    """
    tune_repo_id: str
    max_concurrent_jobs: int = 1
    checkpoints_keep_limit: int = 3
    sync_interval_seconds: int = 3

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
    """
    Source types
    """
    CLONE = "clone"
    DESIGN = "design"
    BLEND = "blend"
    TUNE = "tune"

class TrackType(str, Enum):
    """
    Track types
    """
    CLONE = "clone"
    DESIGN = "design"
    PREEXISTING = "preexisting"

class VoiceProfile(BaseModel):
    """
    Detailed metadata and information for a Voice profile.
    """
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
    """
    API endpoints corresponding to a voice resource.
    """
    audio: str
    bundle: str
    text: str
class VoiceSummary(BaseModel):
    """
    Lightweight representation of a voice for listing.
    """
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
    """
    Detailed metadata and parameters of a registered voice.
    """
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
    
class TuneStatus(str, Enum):
    """
    Lifecycle states of a fine-tuning job.
    """
    PENDING = "pending"
    PROCESSING_DATA = "processing_data"
    TRAINING = "training"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"
    
class TuneProgress(BaseModel):
    """
    Tracks real-time telemetry and metrics for a tuning job in memory.
    """
    percent: float = 0.0
    current_epoch: int = 0
    total_epochs: int = 0
    current_step: int = 0
    total_steps: int = 0
    current_loss: float = 0.0
    loss_history: List[float] = Field(default_factory=list)

class TuneRecord(BaseModel):
    """
    Complete persistent record of a fine-tuning experiment, including its state and progress.
    """
    model_config = ConfigDict(extra="ignore", use_enum_values=True)

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    status: TuneStatus = TuneStatus.PENDING
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    error_message: Optional[str] = None
    
    workspace_path: str 
    
    dataset_manifest_path: str 
    reference_audio_path: str       
    training_params: Dict[str, Any] = Field(default_factory=dict)

    name: str
    language: str = "Auto"
    description: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    
    progress: TuneProgress = Field(default_factory=TuneProgress)
    
class EmotionSummary(BaseModel):
    id: str
    name: str
    modifiers: Dict[str, float]

class IntensifierSummary(BaseModel):
    id: str
    name: str
    multiplier: float

class SynthesisRequest(BaseModel):
    text: str = Field(..., min_length=1, description="The script to synthesize.")
    language: Optional[str] = Field(None, description="ISO language code. Falls back to voice default if omitted.")

class EmotionSynthesisRequest(SynthesisRequest):
    emotion_id: str = Field(..., description="Canonical ID of the emotion.")
    intensifier_id: Optional[str] = Field(None, description="Canonical ID of the intensifier.")

class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    download_url: Optional[str] = None
    error: Optional[str] = None