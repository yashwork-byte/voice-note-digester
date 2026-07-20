"""Deployed Modal web app for the portfolio demo (D024).

    uv run --group eval modal deploy -m voice_digester.web

One L4 container holds every model (int8 STT + fine-tuned Gemma GGUF + LID +
embeddings), reading weights from the Modal volume — no re-download. GPU
offload makes the digest ~8-12 s/note (vs ~25 s CPU). Scales to zero after
5 min idle (cold start ~30-60 s while models load), so a portfolio demo costs
cents. Store is ephemeral per container (starts empty, as the demo intends).

Honest scope: this is a CLOUD demo. The project's point is that the SAME
quantized GGUF runs on-device (`make demo`); this deployment just gives a
clickable link. The frontend (Vercel) hits this app's /api/* with CORS.
"""

import modal

from .modal_infra import get_app, get_volume

VOL = "/vol"

app = get_app("suno-web")
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("ffmpeg", "build-essential", "cmake", "git")
    .env({"HF_HOME": f"{VOL}/hf_cache", "STT_INT8_DIR": f"{VOL}/stt-int8"})
    .uv_pip_install(
        "torch", "torchaudio", "transformers", "numpy", "onnx", "onnxruntime",
        "huggingface_hub", "pydantic", "pyyaml", "pydantic-settings", "pillow",
        "speechbrain>=1.0", "sentence-transformers>=3.0", "sqlite-vec>=0.1",
        "fastapi>=0.110", "python-multipart>=0.0.9",
    )
    # Prebuilt CUDA wheel so digest generation offloads to the L4.
    .run_commands("pip install --no-cache-dir llama-cpp-python"
                  " --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124")
    .add_local_python_source("voice_digester")
    .add_local_dir("configs", "/configs")  # project_root() -> "/" in-container
)
volume = get_volume("voice-digester")


@app.function(
    image=image,
    gpu="L4",
    volumes={VOL: volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    scaledown_window=300,
    max_containers=1,   # one shared store/GPU is right for a portfolio demo
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
