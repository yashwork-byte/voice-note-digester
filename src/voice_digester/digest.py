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
def _llama(gguf_repo: str | None, gguf_file: str | None, gguf_path: str | None = None):
    from llama_cpp import Llama

    if gguf_path is None:
        # hf_hub_download falls back to the local cache when offline;
        # Llama.from_pretrained does not (it queries the HF API first).
        from huggingface_hub import hf_hub_download

        gguf_path = hf_hub_download(gguf_repo, gguf_file)
    return Llama(
        model_path=gguf_path,
        n_ctx=2048,  # prompts are ~700 tokens; smaller KV cache eases 8 GB machines
        n_gpu_layers=-1,  # Metal on the dev Mac; harmless no-op on CPU-only builds
        verbose=False,
    )


# Output-format variants are TRAINED, not grammar-forced (D023 amendment: pure
# grammar field-skipping degraded quality). These strings are imported by
# fine_tune.py so train and serve cannot skew.
CORE_INSTRUCTION_SUFFIX = " Skip the translation."
TRANSLATION_INSTRUCTION = "Translate this voice note transcript into {target_language}."


def _generate(transcript: str, config: DigestConfig, sender: str, schema_model,
              instruction: str | None = None):
    # Sender is app-known metadata fed INTO the model (D022): summaries may then
    # name the sender legitimately instead of inventing one. The shared prompt
    # prefix (system prompt) is KV-cache-reused across calls because the Llama
    # instance is persistent (D023).
    llm = _llama(config.gguf_repo, config.gguf_file, config.gguf_path)
    out = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": config.rendered_system_prompt()},
            {"role": "user",
             "content": f"{instruction or config.user_instruction}\n\nFrom: {sender}\n{transcript}"},
        ],
        response_format={"type": "json_object", "schema": schema_model.model_json_schema()},
        temperature=config.temperature,
        max_tokens=config.max_new_tokens,
        seed=config.seed,
    )
    raw = out["choices"][0]["message"]["content"]
    try:
        return schema_model.model_validate_json(raw)
    except ValidationError as e:
        raise ValueError(f"unparseable digest output: {raw[:200]}") from e


def digest(transcript: str, config: DigestConfig, sender: str = "Unknown") -> NoteDigest:
    """Full trained format — used by the eval harness."""
    return _generate(transcript, config, sender, NoteDigest)


def digest_core(transcript: str, config: DigestConfig, sender: str = "Unknown"):
    """Summary + action items only (~half the tokens); translation is lazy (D023)."""
    from .schema import NoteCore

    return _generate(transcript, config, sender, NoteCore,
                     instruction=config.user_instruction + CORE_INSTRUCTION_SUFFIX)


def digest_translation(transcript: str, config: DigestConfig, sender: str = "Unknown") -> str:
    from .schema import NoteTranslation

    return _generate(
        transcript, config, sender, NoteTranslation,
        instruction=TRANSLATION_INSTRUCTION.format(target_language=config.target_language),
    ).translation


if __name__ == "__main__":
    import sys

    from .config import DigestConfig

    if len(sys.argv) != 2:
        raise SystemExit('usage: python -m voice_digester.digest "<transcript text>"')
    config = DigestConfig.from_yaml("sarvam-translate-4b.yaml")
    print(digest(sys.argv[1], config).model_dump_json(indent=2))
