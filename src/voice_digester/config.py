"""Digest-pipeline configuration (pydantic-settings), loaded from a YAML in configs/.

One config file per model/experiment (house style, D002). `role` separates the
on-device target from the Modal-hosted quality-ceiling reference (D004).
"""

from typing import Literal, Self

import yaml
from pydantic_settings import BaseSettings

from .paths import project_root


class DigestConfig(BaseSettings):
    seed: int = 23

    # Model (D004)
    role: Literal["on-device", "reference"] = "on-device"
    model_name: str = "sarvamai/sarvam-translate"
    gguf_repo: str | None = None
    gguf_file: str | None = None

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
