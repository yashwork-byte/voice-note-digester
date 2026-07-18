"""Push the final artifacts to Hugging Face (D024).

    uv run --env-file .env --group demo python scripts/push_to_hub.py

Three repos under yashwork-byte (needs a write-scope HF_TOKEN):
- gemma-3-4b-it-voice-digest-lora   <- data/models/adapter/ (fetched from the volume)
- gemma-3-4b-it-voice-digest-GGUF   <- data/models/gemma-3-4b-it-ft-Q4_K_M.gguf
- indic-conformer-600m-int8-onnx    <- data/models/indic-conformer-int8/

Model cards come from hf/card-*.md; the LoRA repo also gets the exact system
prompt (system_prompt.txt) extracted from the shipping config.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from huggingface_hub import HfApi

from voice_digester.config import DigestConfig

ROOT = Path(__file__).resolve().parents[1]
OWNER = "yashwork-byte"

PUSHES = [  # (repo, card, path, is_dir)
    ("gemma-3-4b-it-voice-digest-lora", "card-lora.md", ROOT / "data/models/adapter", True),
    ("gemma-3-4b-it-voice-digest-GGUF", "card-gguf.md",
     ROOT / "data/models/gemma-3-4b-it-ft-Q4_K_M.gguf", False),
    ("indic-conformer-600m-int8-onnx", "card-stt.md",
     ROOT / "data/models/indic-conformer-int8", True),
]


def main() -> None:
    api = HfApi(token=os.environ["HF_TOKEN"])
    prompt_txt = ROOT / "data/models/adapter/system_prompt.txt"
    prompt_txt.parent.mkdir(parents=True, exist_ok=True)
    prompt_txt.write_text(DigestConfig.from_yaml("gemma-3-4b-it-ft.yaml").rendered_system_prompt())

    for repo, card, path, is_dir in PUSHES:
        repo_id = f"{OWNER}/{repo}"
        if not path.exists():
            raise SystemExit(f"missing artifact: {path}")
        api.create_repo(repo_id, exist_ok=True)
        api.upload_file(path_or_fileobj=ROOT / "hf" / card, path_in_repo="README.md",
                        repo_id=repo_id)
        if is_dir:
            api.upload_folder(folder_path=str(path), repo_id=repo_id,
                              ignore_patterns=["README.md"])
        else:
            api.upload_file(path_or_fileobj=str(path), path_in_repo=path.name,
                            repo_id=repo_id)
        print(f"pushed {repo_id}")


if __name__ == "__main__":
    main()
