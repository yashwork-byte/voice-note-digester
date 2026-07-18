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
    .uv_pip_install("llama-cpp-python>=0.3", "pillow", "torchvision")
    .add_local_python_source("voice_digester")
    .add_local_dir("configs", "/configs")  # project_root() resolves to "/" in-container
)
volume = get_volume("voice-digester")


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
        from .config import DigestConfig

        self.config = DigestConfig.from_yaml("gemma-3-4b-it-ft.yaml")

    @modal.method()
    def process(self, audio: bytes, language: str) -> dict:
        """audio: any ffmpeg-readable container (the demo sends webm/opus)."""
        import tempfile
        from pathlib import Path

        from .digest import digest
        from .stt import transcribe

        with tempfile.NamedTemporaryFile(suffix=".webm") as f:
            f.write(audio)
            f.flush()
            transcript = transcribe(Path(f.name), language)
        result = digest(transcript, self.config)
        return {"transcript": transcript, "digest": result.model_dump()}
