"""Deployed Modal inference service for the live demo (D021).

    uv run modal deploy -m voice_digester.service

One warm-capable container holds both heavy models (IndicConformer STT + the
fine-tuned Gemma GGUF) so the 8 GB dev laptop never does; the demo backend
calls it via modal.Cls.from_name. First request after idle is slow (model
loads); the container then stays warm for 10 minutes.
"""

import modal

from .modal_infra import get_app, get_image, get_volume

VOL = "/vol"

app = get_app("voice-digester-service")
image = (
    get_image()
    .apt_install("build-essential", "cmake")
    .uv_pip_install("llama-cpp-python>=0.3", "pillow", "torchvision", "speechbrain>=1.0")
    .add_local_python_source("voice_digester")
    .add_local_dir("configs", "/configs")  # project_root() resolves to "/" in-container
)
volume = get_volume("voice-digester")

# Spoken language ID (D013, demo path): VoxLingua107 ECAPA restricted to a
# 3-way choice — code-mixed Hinglish lands on "hi", which is its STT code anyway.
LID_MODEL = "speechbrain/lang-id-voxlingua107-ecapa"
LID_CANDIDATES = {"hi", "ta", "bn"}


@app.cls(
    image=image,
    cpu=8.0,
    memory=16384,
    volumes={VOL: volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    scaledown_window=600,
    timeout=600,
)
class Digester:
    @modal.enter()
    def load(self):
        from speechbrain.inference import EncoderClassifier

        from .config import DigestConfig

        self.config = DigestConfig.from_yaml("gemma-3-4b-it-ft.yaml")
        self.lid = EncoderClassifier.from_hparams(
            source=LID_MODEL, savedir=f"{VOL}/hf_cache/speechbrain-lid")
        ind2lab = self.lid.hparams.label_encoder.ind2lab
        self.lid_index = {lab.split(":")[0].strip(): ind for ind, lab in ind2lab.items()
                          if lab.split(":")[0].strip() in LID_CANDIDATES}

    def _detect_language(self, wav) -> str:
        log_probs = self.lid.classify_batch(wav)[0][0]
        return max(self.lid_index, key=lambda code: log_probs[self.lid_index[code]])

    @modal.method()
    def process(self, audio: bytes, language: str = "auto") -> dict:
        """audio: any ffmpeg-readable container (the demo sends webm/opus)."""
        import tempfile
        from pathlib import Path

        from .digest import digest
        from .stt import decode_audio, transcribe_wav

        with tempfile.NamedTemporaryFile(suffix=".webm") as f:
            f.write(audio)
            f.flush()
            wav = decode_audio(Path(f.name))
        if language == "auto":
            language = self._detect_language(wav)
        transcript = transcribe_wav(wav, language)
        result = digest(transcript, self.config)
        return {"transcript": transcript, "digest": result.model_dump(), "language": language}
