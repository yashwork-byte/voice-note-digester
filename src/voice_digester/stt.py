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


@lru_cache(maxsize=1)
def _model():
    from transformers import AutoModel

    return AutoModel.from_pretrained(STT_MODEL, trust_remote_code=True)


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
