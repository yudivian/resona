import time
import logging
import soundfile as sf
import numpy as np
import torch
import random
import os
from typing import Optional, Union, List, Dict
from src.models import AppConfig
from qwen_tts import Qwen3TTSModel
from qwen_tts.inference.qwen3_tts_model import VoiceClonePromptItem

logger = logging.getLogger(__name__)

CALIBRATION_TEXTS = {
    # English: "The Rainbow Passage". Standard text for phonetic analysis.
    "en": (
        "When the sunlight strikes raindrops in the air, they act like a prism and form a rainbow. "
        "The rainbow is a division of white light into many beautiful colors. "
        "These take the shape of a long round arch, with its path high above, "
        "and its two ends apparently beyond the horizon."
    ),
    # Spanish: "The North Wind and the Sun". IPA standard for Spanish.
    "es": (
        "El viento norte y el sol discutían sobre cuál de ellos era el más fuerte, "
        "cuando pasó un viajero envuelto en una capa pesada. "
        "Se pusieron de acuerdo en que el primero que lograra que el viajero "
        "se quitara la capa sería considerado más fuerte que el otro."
    ),
    # French: "La Bise et le Soleil". Captures liaison and nasals.
    "fr": (
        "La bise et le soleil se disputaient, chacun assurant qu'il était le plus fort, "
        "quand ils ont vu un voyageur qui s'avançait, enveloppé dans son manteau. "
        "Ils sont tombés d'accord que celui qui arriverait le premier "
        "à faire ôter son manteau au voyageur serait regardé comme le plus fort."
    ),
    # German: "Der Nordwind und die Sonne". Captures consonant clusters.
    "de": (
        "Einst stritten sich Nordwind und Sonne, wer von ihnen beiden wohl der Stärkere wäre, "
        "als ein Wanderer, der in einen warmen Mantel gehüllt war, des Weges daherkam. "
        "Sie wurden einig, dass derjenige für den Stärkeren gelten sollte, "
        "der den Wanderer zwingen würde, seinen Mantel abzulegen."
    ),
    # Italian: "La Tramontana e il Sole". Captures gemination.
    "it": (
        "Si disputavano la tramontana e il sole, chi di loro due fosse il più forte, "
        "quando passò un viaggiatore avvolto in un pesante mantello. "
        "Convennero che si sarebbe creduto più forte quello che "
        "fosse riuscito a far togliere il mantello al viaggiatore."
    ),
    # Portuguese: "O vento norte e o sol". Captures nasal vowels.
    "pt": (
        "O vento norte e o sol discutiam qual deles era o mais forte, "
        "quando passou um viajante envolto numa capa pesada. "
        "Concordaram que o que primeiro conseguisse obrigar o viajante "
        "a tirar a capa seria considerado o mais forte."
    ),
    # Chinese: "Spring Breeze". Balanced tones and sibilants.
    "zh": (
        "春风仿佛一位灵巧的画家，把大自然描绘得五彩斑斓。柳树抽出了嫩绿的枝条，"
        "小草从泥土里探出头来，好奇地张望着这个世界。燕子从南方飞回来，"
        "叽叽喳喳地唱着春天的赞歌。"
    ),
    # Japanese: "I Am a Cat" (Soseki). Standard for intonation.
    "ja": (
        "吾輩は猫である。名前はまだ無い。どこで生れたかとんと見当がつかぬ。"
        "何でも薄暗いじめじめした所でニャーニャー泣いていた事だけは記憶している。"
        "吾輩はここで始めて人間というものを見た。"
    ),
    # Korean: Nature description. Balanced phonemes.
    "ko": (
        "바람이 불면 나뭇잎이 흔들리고, 강물은 쉴 새 없이 흐릅니다. "
        "계절이 바뀌면서 세상은 다양한 색으로 물들고, "
        "우리는 그 속에서 자연의 위대함을 배웁니다."
    ),
    # Russian: "The North Wind and the Sun". Palatalization check.
    "ru": (
        "Северный ветер и солнце спорили, кто из них сильнее, "
        "когда увидели путешественника, закутанного в плотный плащ. "
        "Они договорились, что победит тот, кто первым "
        "заставит путешественника снять этот плащ."
    )
}

LANGUAGE_MAP = {
    "en": "English", "es": "Spanish", "fr": "French", "de": "German",
    "it": "Italian", "pt": "Portuguese", "zh": "Chinese", "ja": "Japanese",
    "ko": "Korean", "ru": "Russian"
}

def set_global_seed(seed: int):
    """
    Sets the random seed across Python, Numpy, and Torch to ensure
    deterministic audio generation.

    Args:
        seed (int): The integer seed value to apply.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class TTSModelProvider:
    """
    Singleton resource manager for the TTS models.
    """
    _creator_model = None
    _synthesis_model = None

    def __init__(self, config: AppConfig):
        """
        Initializes the provider with the application configuration.

        Args:
            config (AppConfig): The global application configuration object.
        """
        self.config = config
        self.device = config.system.compute.tts_device
        self.precision = torch.bfloat16 if config.system.compute.precision == "bf16" else torch.float16

    def _prepare_vram(self):
        """
        Synchronizes CUDA devices and clears the cache to prevent OOM errors.
        """
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    def get_creator_model(self) -> Qwen3TTSModel:
        """
        Retrieves the 1.7B parameter model used for Voice Design.

        Returns:
            Qwen3TTSModel: The loaded Design model instance.
        """
        if TTSModelProvider._creator_model is None:
            self._prepare_vram()
            repo = self.config.models.tts.design_repo_id
            TTSModelProvider._creator_model = Qwen3TTSModel.from_pretrained(
                repo, device_map=self.device, dtype=self.precision, attn_implementation="flash_attention_2"
            )
        return TTSModelProvider._creator_model

    def get_synthesis_model(self) -> Qwen3TTSModel:
        """
        Retrieves the 0.6B parameter model used for Vector Extraction and Synthesis.

        Returns:
            Qwen3TTSModel: The loaded Base model instance.
        """
        if TTSModelProvider._synthesis_model is None:
            self._prepare_vram()
            repo = self.config.models.tts.base_repo_id
            TTSModelProvider._synthesis_model = Qwen3TTSModel.from_pretrained(
                repo, device_map=self.device, dtype=self.precision, attn_implementation="flash_attention_2"
            )
        return TTSModelProvider._synthesis_model

class InferenceEngine:
    """
    Manages the lifecycle of a specific voice identity (Track).
    """
    
    def __init__(self, config: AppConfig, tts_provider: TTSModelProvider, lang: str = "en"):
        """
        Initializes the inference engine for a specific track context.

        Args:
            config (AppConfig): Global app configuration.
            tts_provider (TTSModelProvider): Access to model resources.
            lang (str, optional): ISO 639-1 language code. Defaults to "en".
        """
        self.config = config
        self.tts_provider = tts_provider
        self.lang = lang
        
        self.active_identity: Optional[VoiceClonePromptItem] = None
        self.active_seed: Optional[int] = None
        self.last_anchor_path: Optional[str] = None

    def _extract_vectors(self, audio_path: str, text: str):
        """
        Extracts the speaker embedding vector from a physical audio file.

        Args:
            audio_path (str): Path to the anchor audio file.
            text (str): Transcript or calibration text matching the audio.
        """
        model = self.tts_provider.get_synthesis_model()
        prompts = model.create_voice_clone_prompt(ref_audio=audio_path, ref_text=text)
        self.active_identity = prompts[0]
        self.active_identity.ref_spk_embedding = self.active_identity.ref_spk_embedding.clone().detach()

    def design_identity(self, prompt: str, seed: Optional[int] = None):
        """
        Creates a new voice identity from a text description.

        Args:
            prompt (str): The natural language description of the voice.
            seed (Optional[int]): Specific seed for reproducibility. If None, random.
        """
        if seed is None:
            seed = random.randint(0, 2**32 - 1)
        self.active_seed = seed
        set_global_seed(self.active_seed)

        creator = self.tts_provider.get_creator_model()
        cal_text = CALIBRATION_TEXTS.get(self.lang, CALIBRATION_TEXTS["en"])
        full_lang = LANGUAGE_MAP.get(self.lang, "English")
        
        wavs, fs = creator.generate_voice_design(
            text=cal_text,
            language=full_lang,
            instruct=prompt
        )
        
        temp_path = os.path.join(self.config.paths.temp_dir, f"anchor_design_{seed}.wav")
        sf.write(temp_path, wavs[0], fs)
        
        self.last_anchor_path = temp_path
        self._extract_vectors(temp_path, cal_text)

    def extract_identity(self, audio_path: str, transcript: Optional[str] = None):
        """
        Creates a voice identity from an existing audio file (Cloning).

        Args:
            audio_path (str): Path to the source audio file.
            transcript (Optional[str]): The exact text spoken in the audio.
        """
        self.active_seed = 42
        set_global_seed(self.active_seed)
        
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        self.last_anchor_path = audio_path
        self._extract_vectors(audio_path, transcript if transcript else "")

    def load_identity_from_state(self, vector: List[float], seed: int, anchor_path: Optional[str] = None):
        """
        Restores identity from Database state.

        Args:
            vector (List[float]): Speaker embedding.
            seed (int): Generator seed.
            anchor_path (Optional[str]): Path to the original anchor file.
        """
        device = self.tts_provider.device
        dtype = self.tts_provider.precision
        emb = torch.tensor(vector, device=device, dtype=dtype).unsqueeze(0).clone().detach()
        
        self.active_identity = VoiceClonePromptItem(
            ref_code=None, ref_spk_embedding=emb, x_vector_only_mode=True, icl_mode=False, ref_text=None
        )
        self.active_seed = seed
        self.last_anchor_path = anchor_path

    def render(self, text: str, output_path: Optional[str] = None) -> str:
        """
        Generates final audio using the Base Model, Active Vector, and Active Seed.

        Args:
            text (str): Text to synthesize.
            output_path (Optional[str]): Target file path.

        Returns:
            str: Path to generated audio file.
        """
        if not self.active_identity or self.active_seed is None:
            raise ValueError("Identity incomplete (Vector or Seed missing).")

        set_global_seed(self.active_seed)
        
        model = self.tts_provider.get_synthesis_model()
        full_lang = LANGUAGE_MAP.get(self.lang, "English")
        
        if not output_path:
            output_path = os.path.join(self.config.paths.temp_dir, f"render_{int(time.time()*1000)}.wav")

        if torch.cuda.is_available(): torch.cuda.synchronize()

        wavs, fs = model.generate_voice_clone(
            text=text,
            language=full_lang,
            voice_clone_prompt=[self.active_identity]
        )

        if torch.cuda.is_available(): torch.cuda.synchronize()

        sf.write(output_path, wavs[0], fs)
        return output_path

class VoiceBlender:
    """
    Handles interpolation between two InferenceEngines.
    """
    @staticmethod
    def blend(engine_a: InferenceEngine, engine_b: InferenceEngine, alpha: float) -> InferenceEngine:
        """
        Creates a new InferenceEngine by mathematically blending the vectors.

        Args:
            engine_a (InferenceEngine): Source A.
            engine_b (InferenceEngine): Source B.
            alpha (float): Mixing factor (0.0 to 1.0).

        Returns:
            InferenceEngine: A new engine with the blended identity.
        """
        if engine_a.active_identity is None or engine_b.active_identity is None:
            raise ValueError("Missing identities for blending.")

        emb_a = engine_a.active_identity.ref_spk_embedding
        emb_b = engine_b.active_identity.ref_spk_embedding
        if emb_a.device != emb_b.device:
            emb_b = emb_b.to(emb_a.device)

        mixed_emb = ((1.0 - alpha) * emb_a + alpha * emb_b).clone().detach()
        
        temp_identity = VoiceClonePromptItem(
            ref_code=None, ref_spk_embedding=mixed_emb, x_vector_only_mode=True, icl_mode=False, ref_text=None
        )
        
        mixed_seed = random.randint(0, 999999)
        set_global_seed(mixed_seed)
        
        temp_engine = InferenceEngine(engine_a.config, engine_a.tts_provider, engine_a.lang)
        temp_engine.active_identity = temp_identity
        temp_engine.active_seed = mixed_seed
        
        cal_text = CALIBRATION_TEXTS.get(engine_a.lang, CALIBRATION_TEXTS["en"])
        anchor_path = os.path.join(engine_a.config.paths.temp_dir, f"anchor_blend_{mixed_seed}.wav")
        
        temp_engine.render(cal_text, output_path=anchor_path)

        final_engine = InferenceEngine(engine_a.config, engine_a.tts_provider, engine_a.lang)
        final_engine.extract_identity(anchor_path, cal_text)
        final_engine.active_seed = mixed_seed
        
        return final_engine