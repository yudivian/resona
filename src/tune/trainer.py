import os
import sys
import json
import shutil
import logging
from pathlib import Path
from typing import Dict, Any, Tuple

import torch
from accelerate import Accelerator
from safetensors.torch import save_file
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoConfig, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

from src.models import TuneRecord, TuneStatus
from src.tune.registry import TuneRegistry
from config import settings

logger = logging.getLogger(__name__)

class HardwareProfiler:
    """
    Evaluates the system's compute capabilities to determine safe training parameters.
    """

    @staticmethod
    def get_capabilities() -> Dict[str, Any]:
        """
        Retrieves the GPU VRAM and infers the maximum supported training strategies.

        Returns:
            Dict[str, Any]: Hardware metrics and recommended configuration flags.
        """
        if not torch.cuda.is_available():
            return {
                "vram_gb": 0.0,
                "can_full_sft": False,
                "recommended_lora": True,
                "recommended_batch_size": 1
            }
            
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        return {
            "vram_gb": round(vram_gb, 2),
            "can_full_sft": vram_gb >= 20.0,
            "recommended_lora": vram_gb < 20.0,
            "recommended_batch_size": 1 if vram_gb < 12.0 else 2
        }


class TuneTrainer:
    """
    Handles the execution of customized training loops for Qwen3-TTS, supporting 
    both parameter-efficient (LoRA) and full fine-tuning strategies dynamically.
    """

    def __init__(self, registry: TuneRegistry):
        """
        Initializes the TuneTrainer and integrates third-party dependencies.

        Args:
            registry (TuneRegistry): The persistence layer for tune tracking.
        """
        self.registry = registry
        self.base_repo_id = settings.models.tts.base_repo_id
        
        self.qwen_scripts_dir = Path(settings.paths.temp_dir).parent / "qwen3-tts-finetune"
        if str(self.qwen_scripts_dir) not in sys.path:
            sys.path.append(str(self.qwen_scripts_dir))

    def _prepare_model_and_optimizer(
        self, 
        use_lora: bool, 
        force_4bit: bool, 
        learning_rate: float
    ) -> Tuple[torch.nn.Module, Any, Any, Any]:
        """
        Instantiates the Qwen3-TTS architecture, applying quantization and 
        PEFT adapters safely while preserving required computational graphs.

        Args:
            use_lora (bool): Whether to inject LoRA adapters.
            force_4bit (bool): Whether to load the base model in 4-bit precision.
            learning_rate (float): The learning rate for the optimizer.

        Returns:
            Tuple: Initialized model, optimizer, processor, and configuration.
        """
        from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
        
        quantization_config = None
        if use_lora and force_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )

        qwen3tts = Qwen3TTSModel.from_pretrained(
            self.base_repo_id,
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config,
            attn_implementation="flash_attention_2"
        )
        
        config = AutoConfig.from_pretrained(self.base_repo_id)
        model_core = qwen3tts.model

        if use_lora:
            if force_4bit:
                model_core = prepare_model_for_kbit_training(model_core)

            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.FEATURE_EXTRACTION
            )
            model_core = get_peft_model(model_core, lora_config)

        optimizer = AdamW(model_core.parameters(), lr=learning_rate, weight_decay=0.01)
        
        return model_core, optimizer, qwen3tts.processor, config

    def _save_lora_artifacts(self, model: torch.nn.Module, target_embedding: torch.Tensor, output_dir: Path) -> None:
        """
        Persists strictly the PEFT adapter weights and the isolated speaker identity vector.

        Args:
            model (torch.nn.Module): The trained model containing adapters.
            target_embedding (torch.Tensor): The extracted speaker vector.
            output_dir (Path): The destination directory.
        """
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(str(output_dir))
        
        target_cpu = target_embedding.detach().cpu().to(torch.bfloat16)
        torch.save(target_cpu, output_dir / "target_speaker_embedding.pt")

    def _save_full_artifacts(
        self, 
        model: torch.nn.Module, 
        target_embedding: torch.Tensor, 
        speaker_name: str, 
        output_dir: Path
    ) -> None:
        """
        Executes the specialized architecture export for full fine-tuning.

        Args:
            model (torch.nn.Module): The fully trained model.
            target_embedding (torch.Tensor): The extracted speaker vector.
            speaker_name (str): The display name for the registry.
            output_dir (Path): The destination directory.
        """
        os.makedirs(output_dir, exist_ok=True)
        shutil.copytree(self.base_repo_id, output_dir, dirs_exist_ok=True)

        input_config_file = os.path.join(self.base_repo_id, "config.json")
        output_config_file = os.path.join(output_dir, "config.json")
        
        with open(input_config_file, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
            
        config_dict["tts_model_type"] = "custom_voice"
        talker_config = config_dict.get("talker_config", {})
        talker_config["spk_id"] = {speaker_name: 3000}
        talker_config["spk_is_dialect"] = {speaker_name: False}
        config_dict["talker_config"] = talker_config

        with open(output_config_file, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)

        state_dict = {k: v.detach().to("cpu") for k, v in model.state_dict().items()}

        keys_to_drop = [k for k in state_dict.keys() if k.startswith("speaker_encoder")]
        for k in keys_to_drop:
            del state_dict[k]

        weight_key = 'talker.model.codec_embedding.weight'
        new_weight = state_dict[weight_key].clone()
        new_weight[3000] = target_embedding[0].detach().to(new_weight.device).to(new_weight.dtype)
        state_dict[weight_key] = new_weight
        
        save_file(state_dict, os.path.join(output_dir, "model.safetensors"))

    def _execute_training_loop(self, tune_id: str, record: TuneRecord) -> bool:
        """
        Executes the customized PyTorch optimization loop matching the architecture specifications.

        Args:
            tune_id (str): The identifier of the tune job.
            record (TuneRecord): The database record containing training parameters.

        Returns:
            bool: True if training completes successfully.
        """
        try:
            from dataset import TTSDataset
        except ImportError as e:
            logger.error(f"Failed to import dataset module: {e}")
            return False

        hw_profile = HardwareProfiler.get_capabilities()
        
        requested_lora = record.training_params.get("use_lora", True)
        use_lora = requested_lora if hw_profile["can_full_sft"] else True
        force_4bit = use_lora and (hw_profile["vram_gb"] < 12.0)
        
        batch_size = record.training_params.get("batch_size", hw_profile["recommended_batch_size"])
        epochs = record.training_params.get("epochs", 3)
        lr = record.training_params.get("learning_rate", 2e-4 if use_lora else 2e-5)

        workspace_path = Path(record.workspace_path)
        processed_manifest = workspace_path / "processed" / "train_with_codes.jsonl"
        checkpoint_dir = workspace_path / "checkpoints" / ("lora_adapter" if use_lora else "final_model")

        accelerator = Accelerator(
            gradient_accumulation_steps=record.training_params.get("gradient_accumulation_steps", 4), 
            mixed_precision="bf16"
        )

        try:
            model_core, optimizer, processor, config = self._prepare_model_and_optimizer(
                use_lora, force_4bit, lr
            )

            train_data = []
            with open(processed_manifest, "r", encoding="utf-8") as f:
                for line in f:
                    train_data.append(json.loads(line.strip()))

            dataset = TTSDataset(train_data, processor, config)
            train_dataloader = DataLoader(
                dataset, 
                batch_size=batch_size, 
                shuffle=True, 
                collate_fn=dataset.collate_fn
            )

            model_core, optimizer, train_dataloader = accelerator.prepare(
                model_core, optimizer, train_dataloader
            )

            model_core.train()
            target_speaker_embedding = None

            for epoch in range(epochs):
                for step, batch in enumerate(train_dataloader):
                    with accelerator.accumulate(model_core):
                        
                        input_ids = batch['input_ids']
                        codec_ids = batch['codec_ids']
                        ref_mels = batch['ref_mels']
                        text_embedding_mask = batch['text_embedding_mask']
                        codec_embedding_mask = batch['codec_embedding_mask']
                        attention_mask = batch['attention_mask']
                        codec_0_labels = batch['codec_0_labels']
                        codec_mask = batch['codec_mask']

                        unwrapped = accelerator.unwrap_model(model_core)
                        if hasattr(unwrapped, "base_model") and hasattr(unwrapped.base_model, "model"):
                            base_model = unwrapped.base_model.model
                        else:
                            base_model = unwrapped

                        speaker_embedding = base_model.speaker_encoder(
                            ref_mels.to(accelerator.device).to(torch.bfloat16)
                        ).detach()
                        
                        if target_speaker_embedding is None:
                            target_speaker_embedding = speaker_embedding.clone()

                        input_text_ids = input_ids[:, :, 0]
                        input_codec_ids = input_ids[:, :, 1]

                        input_text_embedding = base_model.talker.model.text_embedding(input_text_ids) * text_embedding_mask
                        input_codec_embedding = base_model.talker.model.codec_embedding(input_codec_ids) * codec_embedding_mask
                        input_codec_embedding[:, 6, :] = speaker_embedding

                        input_embeddings = input_text_embedding + input_codec_embedding

                        for i in range(1, 16):
                            codec_i_embedding = base_model.talker.code_predictor.get_input_embeddings()[i - 1](codec_ids[:, :, i])
                            codec_i_embedding = codec_i_embedding * codec_mask.unsqueeze(-1)
                            input_embeddings = input_embeddings + codec_i_embedding

                        outputs = base_model.talker(
                            inputs_embeds=input_embeddings[:, :-1, :],
                            attention_mask=attention_mask[:, :-1],
                            labels=codec_0_labels[:, 1:],
                            output_hidden_states=True
                        )

                        hidden_states = outputs.hidden_states[0][-1]
                        talker_hidden_states = hidden_states[codec_mask[:, 1:]]
                        talker_codec_ids = codec_ids[codec_mask]

                        sub_talker_logits, sub_talker_loss = base_model.talker.forward_sub_talker_finetune(
                            talker_codec_ids, 
                            talker_hidden_states
                        )

                        loss = outputs.loss + 0.3 * sub_talker_loss

                        accelerator.backward(loss)

                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(model_core.parameters(), 1.0)

                        optimizer.step()
                        optimizer.zero_grad()

            if accelerator.is_main_process:
                unwrapped_final = accelerator.unwrap_model(model_core)
                
                if target_speaker_embedding is None:
                    raise ValueError("Target speaker embedding was not captured.")

                if use_lora:
                    self._save_lora_artifacts(unwrapped_final, target_speaker_embedding, checkpoint_dir)
                else:
                    self._save_full_artifacts(unwrapped_final, target_speaker_embedding, record.name, checkpoint_dir)
                
            return True

        except Exception as e:
            logger.error(f"Training loop failed for tune {tune_id}: {e}")
            return False
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def train(self, tune_id: str) -> bool:
        """
        Main orchestration method for the tuning phase.

        Args:
            tune_id (str): The identifier of the tune job.

        Returns:
            bool: True if the process successfully yields a persisted model.
        """
        record = self.registry.get_tune(tune_id)
        if not record:
            return False

        self.registry.update_status(tune_id, TuneStatus.TRAINING)

        train_success = self._execute_training_loop(tune_id, record)

        if train_success:
            self.registry.update_status(tune_id, TuneStatus.COMPLETED)
            return True
        else:
            self.registry.update_status(tune_id, TuneStatus.FAILED, error_message="Execution failed.")
            return False