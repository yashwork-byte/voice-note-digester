"""STT stage: IndicConformer-600M (D003).

Loads via HF trust_remote_code; audio is decoded with ffmpeg (already a project
requirement) to a mono 16 kHz float32 tensor, sidestepping torchaudio backend
issues with WhatsApp's Ogg/Opus. Output is unpunctuated text in the language's
native script — code-mixed Hinglish comes back as Devanagari (no Latin), and
there is no language ID: the caller must supply the language (D013, open).

Heavy deps (torch/transformers) are imported lazily; install with the `stt`
group locally, or run batch jobs on Modal (asr_round_trip.py) — the 600M fp32
model is too heavy for the 8 GB dev laptop.
"""

import subprocess
from functools import lru_cache
from pathlib import Path

STT_MODEL = "ai4bharat/indic-conformer-600m-multilingual"
STT_LANGUAGE = {"hi": "hi", "hi-en": "hi", "ta": "ta", "bn": "bn"}


def decode_audio(audio_path: Path):
    """Any ffmpeg-readable audio -> mono 16 kHz float32 tensor of shape [1, T]."""
    import numpy as np
    import torch

    raw = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(audio_path),
         "-f", "f32le", "-ac", "1", "-ar", "16000", "pipe:1"],
        capture_output=True, check=True,
    ).stdout
    return torch.from_numpy(np.frombuffer(raw, dtype=np.float32).copy()).unsqueeze(0)


def _int8_dir() -> Path | None:
    """Local int8-ONNX copy built by scripts/quantize_stt.py (D023). The repo
    runs on ONNX internally, so quantization happens at the ONNX level —
    torch-level dynamic quantization was a verified no-op."""
    candidate = Path(__file__).resolve().parents[2] / "data" / "models" / "indic-conformer-int8"
    return candidate if (candidate / "assets" / "encoder.onnx").exists() else None


def _allow_local_snapshots() -> None:
    """The repo's remote code calls snapshot_download unconditionally, so a
    local model directory crashes it. Shim: an existing local path is returned
    as-is. Removable once the int8 copy lives in a HF repo of our own."""
    import huggingface_hub

    original = huggingface_hub.snapshot_download

    def patched(repo_id=None, **kwargs):
        if repo_id and Path(str(repo_id)).exists():
            return str(repo_id)
        return original(repo_id=repo_id, **kwargs)

    huggingface_hub.snapshot_download = patched


@lru_cache(maxsize=1)
def _model():
    from transformers import AutoModel

    source = _int8_dir() or STT_MODEL
    if source != STT_MODEL:
        _allow_local_snapshots()
    return AutoModel.from_pretrained(str(source), trust_remote_code=True)


def transcribe_wav(wav, language: str, decoding: str = "ctc") -> str:
    """`wav` is a mono 16 kHz float32 tensor [1, T] (see decode_audio)."""
    lang = STT_LANGUAGE.get(language, language)
    return _model()(wav, lang, decoding)


def transcribe(audio_path: Path, language: str, decoding: str = "ctc") -> str:
    """`language` is an eval-set tag ("hi", "hi-en", "ta", "bn") or a raw model code."""
    return transcribe_wav(decode_audio(audio_path), language, decoding)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        raise SystemExit("usage: python -m voice_digester.stt <audio-file> <language>")
    print(transcribe(Path(sys.argv[1]), sys.argv[2]))
