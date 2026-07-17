"""Combined digest stage (D004, D005): one structured llama.cpp call.

Quantized Sarvam-Translate 4B (GGUF) produces summary + translation + action
items in a single pass, with the output constrained to the NoteDigest JSON
schema via llama.cpp's grammar support. The GGUF is pulled from HF on first
use and cached. Raises ValueError on unparseable output (evaluate.py counts
that as a full failure — strict house style).
"""

from functools import lru_cache

from pydantic import ValidationError

from .config import DigestConfig
from .schema import NoteDigest


@lru_cache(maxsize=1)
def _llama(gguf_repo: str, gguf_file: str):
    # hf_hub_download falls back to the local cache when offline;
    # Llama.from_pretrained does not (it queries the HF API first).
    from huggingface_hub import hf_hub_download
    from llama_cpp import Llama

    model_path = hf_hub_download(gguf_repo, gguf_file)
    return Llama(
        model_path=model_path,
        n_ctx=4096,
        n_gpu_layers=-1,  # Metal on the dev Mac; harmless no-op on CPU-only builds
        verbose=False,
    )


def digest(transcript: str, config: DigestConfig) -> NoteDigest:
    llm = _llama(config.gguf_repo, config.gguf_file)
    out = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": config.rendered_system_prompt()},
            {"role": "user", "content": f"{config.user_instruction}\n\n{transcript}"},
        ],
        response_format={"type": "json_object", "schema": NoteDigest.model_json_schema()},
        temperature=config.temperature,
        max_tokens=config.max_new_tokens,
        seed=config.seed,
    )
    raw = out["choices"][0]["message"]["content"]
    try:
        return NoteDigest.model_validate_json(raw)
    except ValidationError as e:
        raise ValueError(f"unparseable digest output: {raw[:200]}") from e


if __name__ == "__main__":
    import sys

    from .config import DigestConfig

    if len(sys.argv) != 2:
        raise SystemExit('usage: python -m voice_digester.digest "<transcript text>"')
    config = DigestConfig.from_yaml("sarvam-translate-4b.yaml")
    print(digest(sys.argv[1], config).model_dump_json(indent=2))
