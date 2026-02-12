import time
import logging
import soundfile as sf
import numpy as np
import torch
import os
from typing import Optional, Union, List, Any
from src.models import AppConfig
from qwen_tts import Qwen3TTSModel
from qwen_tts.inference.qwen3_tts_model import VoiceClonePromptItem

logger = logging.getLogger(__name__)

CALIBRATION_TEXTS = {
    # English: "The Rainbow Passage"
    "en": (
        "When the sunlight strikes raindrops in the air, they act like a prism and form a rainbow. "
        "The rainbow is a division of white light into many beautiful colors. "
        "These take the shape of a long round arch, with its path high above, "
        "and its two ends apparently beyond the horizon."
    ),
    # Spanish: "The North Wind and the Sun" (Aesop's Fable)
    "es": (
        "El viento norte y el sol discutían sobre cuál de ellos era el más fuerte, "
        "cuando pasó un viajero envuelto en una capa pesada. "
        "Se pusieron de acuerdo en que el primero que lograra que el viajero "
        "se quitara la capa sería considerado más fuerte que el otro."
    ),
    # French: "La Bise et le Soleil" (Aesop's Fable)
    "fr": (
        "La bise et le soleil se disputaient, chacun assurant qu'il était le plus fort, "
        "quand ils ont vu un voyageur qui s'avançait, enveloppé dans son manteau. "
        "Ils sont tombés d'accord que celui qui arriverait le premier "
        "à faire ôter son manteau au voyageur serait regardé comme le plus fort."
    ),
    # German: "Der Nordwind und die Sonne" (Aesop's Fable)
    "de": (
        "Einst stritten sich Nordwind und Sonne, wer von ihnen beiden wohl der Stärkere wäre, "
        "als ein Wanderer, der in einen warmen Mantel gehüllt war, des Weges daherkam. "
        "Sie wurden einig, dass derjenige für den Stärkeren gelten sollte, "
        "der den Wanderer zwingen würde, seinen Mantel abzulegen."
    ),
    # Italian: "La Tramontana e il Sole" (Aesop's Fable)
    "it": (
        "Si disputavano la tramontana e il sole, chi di loro due fosse il più forte, "
        "quando passò un viaggiatore avvolto in un pesante mantello. "
        "Convennero che si sarebbe creduto più forte quello che "
        "fosse riuscito a far togliere il mantello al viaggiatore."
    ),
    # Portuguese: "O Vento Norte e o Sol" (Aesop's Fable)
    "pt": (
        "O vento norte e o sol discutiam qual deles era o mais forte, "
        "quando passou um viajante envolto numa capa pesada. "
        "Concordaram que o que primeiro conseguisse obrigar o viajante "
        "a tirar a capa seria considerado o más forte."
    ),
    # Chinese (Mandarin): "Spring Breeze"
    "zh": (
        "春风仿佛一位灵巧的画家，把大自然描绘得五彩斑斓。柳树抽出了嫩绿的枝条，"
        "小草从泥土里探出头来，好奇地张望着这个世界。燕子从南方飞回来，"
        "叽叽喳喳地唱着春天的赞歌。"
    ),
    # Japanese: Opening of "I Am a Cat" (Wagahai wa Neko de Aru)
    "ja": (
        "吾輩は猫である。名前はまだ無い。どこで生れたかとんと見当がつかぬ。"
        "何でも薄暗いじめじめした所でニャーニャー泣いていた事だけは記憶している。"
        "吾輩はここで始めて人間というものを見た。"
    ),
    # Korean: Standard nature description passage
    "ko": (
        "바람이 불면 나뭇잎이 흔들리고, 강물은 쉴 새 없이 흐릅니다. "
        "계절이 바뀌면서 세상은 다양한 색으로 물들고, "
        "우리는 그 속에서 자연의 위대함을 배웁니다."
    ),
    # Russian: "The North Wind and the Sun" (Aesop's Fable)
    "ru": (
        "Северный ветер и солнце спорили, кто из них сильнее, "
        "когда увидели путешественника, закутанного в плотный плащ. "
        "Они договорились, что победит тот, кто первым "
        "заставит путешественника снять этот плащ."
    )
}

class TTSModelProvider:
    """
    Singleton resource manager for the Qwen3-TTS model.
    """
    _model = None

    def __init__(self, config: AppConfig):
        """
        Initialize the provider with application configuration.

        Args:
            config (AppConfig): Global application configuration.
        """
        self.config = config
        self.device = config.system.compute.tts_device
        self.precision = torch.bfloat16 if config.system.compute.precision == "bf16" else torch.float16

    def get_model(self) -> Qwen3TTSModel:
        """
        Retrieves or initializes the Qwen3-TTS model instance.

        Returns:
            Qwen3TTSModel: The loaded model wrapper.
        """
        if TTSModelProvider._model is None:
            use_flash_attn = (
                "cuda" in self.device and 
                self.config.system.compute.precision in ["fp16", "bf16"]
            )
            logger.info(f"Loading Qwen3-TTS model on {self.device} (FlashAttn: {use_flash_attn})")
            
            TTSModelProvider._model = Qwen3TTSModel.from_pretrained(
                self.config.models.tts.repo_id,
                device_map=self.device,
                dtype=self.precision,
                attn_implementation="flash_attention_2" if use_flash_attn else "eager"
            )
        return TTSModelProvider._model

class InferenceEngine:
    """
    Stateful execution engine for voice identity lifecycle management.
    """
    def __init__(self, config: AppConfig, tts_provider: TTSModelProvider, lang: str = "en"):
        """
        Initialize the engine for a specific language context.

        Args:
            config (AppConfig): App configuration.
            tts_provider (TTSModelProvider): Shared model provider.
            lang (str): ISO language code.
        """
        self.config = config
        self.tts_provider = tts_provider
        self.lang = lang
        self.model = None
        self.active_identity: Optional[VoiceClonePromptItem] = None
        self.last_anchor_path: Optional[str] = None
        self._seed = 42

    def _load_model(self):
        """
        Ensures the model is loaded before any operation.
        """
        if self.model is None:
            self.model = self.tts_provider.get_model()

    def design_identity(self, prompt: str):
        """
        Generates a synthetic voice identity from a text description.

        Args:
            prompt (str): Natural language description of the voice.
        """
        self._load_model()
        calibration_text = CALIBRATION_TEXTS.get(self.lang, CALIBRATION_TEXTS["en"])
        
        logger.info(f"Designing voice for lang={self.lang} with prompt='{prompt}'")

        try:
            design_outputs = self.model.generate_voice_design(
                prompt=prompt,
                text=calibration_text
            )
            audio_data = design_outputs[0]

            timestamp = int(time.time() * 1000)
            self.last_anchor_path = os.path.join(
                self.config.system.paths.temp, 
                f"design_anchor_{timestamp}.wav"
            )
            sf.write(self.last_anchor_path, audio_data, 24000)

            self.active_identity = self.model.create_voice_clone_prompt(
                audio_sample=audio_data
            )
            logger.info("Identity designed and extracted successfully.")
            
        except Exception as e:
            logger.error(f"Failed to design identity: {e}")
            raise

    def extract_identity(self, audio_path: str, transcript: Optional[str] = None):
        """
        Extracts a voice identity from an existing audio file.

        Args:
            audio_path (str): Path to the source audio file.
            transcript (Optional[str]): Text spoken in the audio for ICL mode.
        """
        self._load_model()
        self.last_anchor_path = audio_path
        
        logger.info(f"Extracting identity from audio: {audio_path}")
        
        use_x_vector = False
        if not transcript:
            logger.warning("No transcript provided. Using X-Vector mode.")
            use_x_vector = True
            transcript = ""

        try:
            self.active_identity = self.model.create_voice_clone_prompt(
                ref_audio=audio_path,
                ref_text=transcript,
                x_vector_only_mode=use_x_vector
            )
            logger.info("Identity extracted successfully.")
        except Exception as e:
            logger.error(f"Failed to extract identity: {e}")
            raise

    def load_identity_from_vector(self, vector: List[float]):
        """
        Reconstructs an identity from a stored embedding vector.

        Args:
            vector (List[float]): Identity embedding vector.
        """
        self._load_model()
        logger.info("Loading identity from raw vector.")
        
        try:
            device = self.model.device
            embedding_tensor = torch.tensor(vector, device=device).unsqueeze(0)
            
            self.active_identity = VoiceClonePromptItem(
                ref_code=None,
                ref_spk_embedding=embedding_tensor,
                ref_text=None,
                x_vector_only_mode=True,
                icl_mode=False
            )
            logger.info("Identity reconstructed from vector.")
        except Exception as e:
            logger.error(f"Failed to load vector: {e}")
            raise

    def set_identity(self, identity_item: VoiceClonePromptItem, seed: int = 42):
        """
        Sets the identity object directly.

        Args:
            identity_item (VoiceClonePromptItem): The identity object.
            seed (int): Reproducibility seed.
        """
        self.active_identity = identity_item
        self._seed = seed
        logger.debug("Identity set directly.")

    def get_identity_vector(self) -> List[float]:
        """
        Retrieves the raw speaker embedding vector.

        Returns:
            List[float]: The speaker embedding.
        """
        if self.active_identity is None:
            raise ValueError("No active identity.")
        
        embedding = self.active_identity.ref_spk_embedding
        return embedding.squeeze().cpu().detach().numpy().tolist()

    def generate_preview(self, text: str, refinement: Optional[str] = None) -> str:
        """
        Generates a temporary audio preview.

        Args:
            text (str): Content to synthesize.
            refinement (Optional[str]): Style instruction.

        Returns:
            str: Path to the preview file.
        """
        if self.active_identity is None:
            raise ValueError("Identity state is empty.")
        
        return self._render(text, refinement=refinement)

    def render_anchor(self, asset_name: str, refinement: Optional[str] = None) -> str:
        """
        Solidifies current identity into a permanent anchor audio.

        Args:
            asset_name (str): Base name for the asset.
            refinement (Optional[str]): Style prompt to bake in.

        Returns:
            str: Path to the permanent anchor audio.
        """
        self._load_model()
        calibration_script = CALIBRATION_TEXTS.get(self.lang, CALIBRATION_TEXTS["en"])
        
        path = os.path.join(
            self.config.system.paths.assets, 
            f"{asset_name}_anchor.wav"
        )
        
        logger.info(f"Solidifying identity anchor to {path}")
        return self._render(calibration_script, output_path=path, refinement=refinement)

    def _render(self, text: str, output_path: Optional[str] = None, refinement: Optional[str] = None) -> str:
        """
        Internal synthesis helper.
        """
        self._load_model()
        
        if output_path is None:
            timestamp = int(time.time() * 1000)
            output_path = os.path.join(
                self.config.system.paths.temp, 
                f"render_{timestamp}.wav"
            )

        try:
            wavs = self.model.generate_voice_clone(
                text=text,
                voice_clone_prompt=self.active_identity,
                instruct=refinement if refinement else ""
            )
            
            sf.write(output_path, wavs[0], 24000)
            logger.debug(f"Rendered audio to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Rendering failed: {e}")
            raise

class VoiceBlender:
    """
    Utility for blending voice identities.
    """
    @staticmethod
    def blend(engine_a: InferenceEngine, engine_b: InferenceEngine, alpha: float) -> InferenceEngine:
        """
        Creates a new InferenceEngine with a blended identity.

        Args:
            engine_a (InferenceEngine): Primary track.
            engine_b (InferenceEngine): Secondary track.
            alpha (float): Blending ratio.

        Returns:
            InferenceEngine: Engine with the mixed identity.
        """
        if engine_a.active_identity is None or engine_b.active_identity is None:
            logger.error("Missing identities for blending.")
            raise ValueError("Both engines must have active identities.")
            
        logger.info(f"Blending engines with alpha={alpha}")

        emb_a = engine_a.active_identity.ref_spk_embedding
        emb_b = engine_b.active_identity.ref_spk_embedding

        if emb_a.device != emb_b.device:
            emb_b = emb_b.to(emb_a.device)

        mixed_embedding = (1.0 - alpha) * emb_a + alpha * emb_b

        hybrid_identity = VoiceClonePromptItem(
            ref_code=None,
            ref_spk_embedding=mixed_embedding,
            ref_text=None,
            x_vector_only_mode=True, 
            icl_mode=False
        )
        
        mixed_engine = InferenceEngine(
            config=engine_a.config, 
            tts_provider=engine_a.tts_provider,
            lang=engine_a.lang 
        )
        
        mixed_engine.set_identity(hybrid_identity)
        
        return mixed_engine