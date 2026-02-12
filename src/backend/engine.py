import time
import torch
import numpy as np
import soundfile as sf
from typing import List, Optional, Union
from src.models import AppConfig
from qwen_tts import Qwen3TTSModel
from qwen_tts.inference.qwen3_tts_model import VoiceClonePromptItem

class TTSModelProvider:
    """
    Singleton resource manager for the Qwen3-TTS model.
    """
    _model = None

    def __init__(self, config: AppConfig):
        self.config = config
        self.device = config.system.compute.tts_device
        self.precision = torch.bfloat16 if config.system.compute.precision == "bf16" else torch.float16

    def get_model(self) -> Qwen3TTSModel:
        """
        Initializes and returns the Qwen3-TTS wrapper.
        """
        if TTSModelProvider._model is None:
            use_flash_attn = (
                "cuda" in self.device and 
                self.config.system.compute.precision in ["fp16", "bf16"]
            )
            
            TTSModelProvider._model = Qwen3TTSModel.from_pretrained(
                self.config.models.tts.repo_id,
                device_map=self.device,
                dtype=self.precision,
                attn_implementation="flash_attention_2" if use_flash_attn else "eager"
            )
        return TTSModelProvider._model

class InferenceEngine:
    """
    Stateful execution engine for a single voice design track.
    Manages identity state and deterministic seed synchronization.
    """
    def __init__(self, config: AppConfig, provider: TTSModelProvider, lang: str, seed: Optional[int] = None):
        self.config = config
        self.lang = lang
        self.tts = provider.get_model()
        self.device = self.tts.device
        self.active_identity: Optional[torch.Tensor] = None
        
        # Initialize or restore seed
        self._seed = seed if seed is not None else int(torch.randint(0, 2**32, (1,)).item())
        self._apply_seed()

    @property
    def seed(self) -> int:
        """Exposes the active seed for persistence and tracking."""
        return self._seed

    def _apply_seed(self):
        """Synchronizes random generators to ensure reproducibility."""
        torch.manual_seed(self._seed)
        torch.cuda.manual_seed_all(self._seed)
        np.random.seed(self._seed)

    def set_identity(self, vector: Union[List[float], torch.Tensor], seed: Optional[int] = None):
        """
        Loads an identity and synchronizes its original seed if provided.
        """
        if seed is not None:
            self._seed = seed
            self._apply_seed()

        if isinstance(vector, list):
            self.active_identity = torch.tensor(
                vector, 
                device=self.device, 
                dtype=self.tts.model.dtype
            )
        else:
            self.active_identity = vector.to(device=self.device, dtype=self.tts.model.dtype)

    def extract_identity(self, audio_path: str, transcript: str):
        """Extracts speaker embedding and stores it in the instance state."""
        prompt_items = self.tts.create_voice_clone_prompt(
            ref_audio=audio_path,
            ref_text=transcript,
            x_vector_only_mode=True
        )
        self.active_identity = prompt_items[0].ref_spk_embedding
        return self.active_identity.squeeze().tolist()

    def design_identity(self, prompt: str):
        """Generates a new identity from text and updates the active state."""
        self._apply_seed()
        with torch.no_grad():
            instruct_text = self.tts._build_instruct_text(prompt)
            inputs = self.tts.processor(text="", instruct=instruct_text, return_tensors="pt")
            input_ids = inputs["input_ids"].to(self.device)
            
            spk_emb = self.tts.model.extract_speaker_embedding(
                input_ids=input_ids,
                language=self.lang
            )
        self.active_identity = spk_emb
        return self.active_identity.squeeze().tolist()

    def _render(self, text: str, output_path: str, refinement: Optional[str] = None) -> str:
        """Internal synthesis logic using encapsulated state and seed."""
        if self.active_identity is None:
            raise ValueError("InferenceEngine: Identity state is empty.")

        self._apply_seed()
        
        prompt_item = VoiceClonePromptItem(
            ref_code=None,
            ref_spk_embedding=self.active_identity,
            x_vector_only_mode=True,
            icl_mode=False
        )
        
        wavs, sr = self.tts.generate_voice_clone(
            text=text,
            language=self.lang,
            voice_clone_prompt=[prompt_item],
            instruct=refinement if refinement else ""
        )
        
        sf.write(output_path, wavs[0], sr)
        return output_path

    def generate_preview(self, text: str, refinement: Optional[str] = None) -> str:
        """Renders temporary audio for the current workspace state."""
        timestamp = int(time.time() * 1000)
        path = f"{self.config.paths.temp_dir}/preview_{timestamp}_{self._seed}.wav"
        return self._render(text, path, refinement)

    def render_anchor(self, calibration_script: str, asset_name: str) -> str:
        """Persists the final identity anchor to the project storage."""
        path = f"{self.config.paths.assets_dir}/{asset_name}_anchor.wav"
        return self._render(calibration_script, path)

class VoiceBlender:
    """Mathematical utility for identity vector interpolation."""
    @staticmethod
    def blend(engine_a: InferenceEngine, engine_b: InferenceEngine, alpha: float) -> List[float]:
        """Interpolates between the active identities of two engine instances."""
        if engine_a.active_identity is None or engine_b.active_identity is None:
            raise ValueError("VoiceBlender: Both engines must have active identities.")
            
        v_a = engine_a.active_identity.cpu().numpy()
        v_b = engine_b.active_identity.cpu().numpy()
        
        blended = (1.0 - alpha) * v_a + alpha * v_b
        return blended.squeeze().tolist()