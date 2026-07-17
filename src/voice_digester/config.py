"""Digest-pipeline configuration (pydantic-settings), loaded from a YAML in configs/.

One config file per model/experiment (house style, D002). `role` separates the
on-device target from the Modal-hosted quality-ceiling reference (D004).
DigestConfig drives inference; TrainingConfig drives the LoRA run (D019) and
points at a DigestConfig file so train and serve share the same prompts.
"""

from datetime import datetime
from typing import Literal, Self

import yaml
from pydantic import model_validator
from pydantic_settings import BaseSettings

from .paths import project_root


class DigestConfig(BaseSettings):
    seed: int = 23

    # Model (D004)
    role: Literal["on-device", "reference"] = "on-device"
    model_name: str = "sarvamai/sarvam-translate"
    gguf_repo: str | None = None
    gguf_file: str | None = None
    gguf_path: str | None = None  # direct file path (e.g. on the Modal volume); wins over repo/file

    # STT (D003)
    stt_model_name: str = "ai4bharat/indic-conformer-600m-multilingual"

    # Digest call (D005) — system_prompt may contain a {target_language} placeholder
    target_language: str = "English"
    system_prompt: str
    user_instruction: str

    # Generation
    temperature: float = 0.1
    max_new_tokens: int = 1024

    @classmethod
    def from_yaml(cls, file_name: str) -> Self:
        path = project_root() / "configs" / file_name
        with open(path) as f:
            return cls(**yaml.safe_load(f))

    def rendered_system_prompt(self) -> str:
        return self.system_prompt.format(target_language=self.target_language)


class TrainingConfig(BaseSettings):
    seed: int = 23

    base_model: str = "google/gemma-3-4b-it"
    digest_config_file: str = "gemma-3-4b-it.yaml"  # train/serve-shared prompts

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    ]

    # Training
    learning_rate: float = 2e-4
    num_train_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    logging_steps: int = 10
    max_steps: int | None = None  # smoke-test override; None = full epochs

    # Output (on the Modal volume)
    output_dir: str = "/vol/checkpoints"
    run_name: str | None = None

    @classmethod
    def from_yaml(cls, file_name: str) -> Self:
        path = project_root() / "configs" / file_name
        with open(path) as f:
            return cls(**yaml.safe_load(f))

    @model_validator(mode="after")
    def _set_run_name(self) -> Self:
        if self.run_name is None:
            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            self.run_name = f"{self.base_model.split('/')[-1]}-ft-{ts}"
        return self
