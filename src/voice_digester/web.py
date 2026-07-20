"""Deployed Modal web app for the portfolio demo (D024).

    uv run --group eval modal deploy -m voice_digester.web

One CPU container holds every model (int8 STT + fine-tuned Gemma GGUF + LID +
embeddings), reading weights from the Modal volume — no re-download. CPU-only
by choice: GPU on Modal fought this mixed torch+llama.cpp workload (torch
NVRTC fuser + CUDA-wheel offload) for no reliable win, since cold-start model
loading dominates latency regardless. Warm digest ~25-30 s/note; scales to
zero after 5 min idle (cold start ~60 s), so a portfolio demo costs ~nothing.

Honest scope: a CLOUD demo. The project's point is that the SAME quantized
GGUF runs on-device (`make demo`); this just gives a clickable link. The
frontend (Vercel) hits this app's /api/* with CORS. Store is ephemeral per
container (starts empty, as the demo intends).
"""

import modal

from .modal_infra import get_app, get_image, get_volume

VOL = "/vol"

app = get_app("suno-web")
image = (
    get_image()  # debian + ffmpeg + torch/onnx/transformers, HF_HOME on the volume
    .env({"STT_INT8_DIR": f"{VOL}/stt-int8"})
    .uv_pip_install(
        "llama-cpp-python>=0.3",  # prebuilt CPU wheel from PyPI (no compile)
        "speechbrain>=1.0", "sentence-transformers>=3.0", "sqlite-vec>=0.1",
        "fastapi>=0.110", "python-multipart>=0.0.9", "pillow",
    )
    .add_local_python_source("voice_digester")
    .add_local_dir("configs", "/configs")  # project_root() -> "/" in-container
)
volume = get_volume("voice-digester")


@app.function(
    image=image,
    cpu=8.0,
    memory=16384,
    volumes={VOL: volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    scaledown_window=300,
    max_containers=1,   # one shared store is right for a portfolio demo
    timeout=600,
)
@modal.concurrent(max_inputs=8)
@modal.asgi_app()
def fastapi_app():
    from .api import create_api, warm_models
    from .config import DigestConfig

    config = DigestConfig.from_yaml("gemma-3-4b-it-ft.yaml")  # gguf_path -> /vol/gguf/...
    warm_models(config)  # load before serving so the container is warm when ready
    return create_api(config, "/tmp/notes.db", cors=True)
