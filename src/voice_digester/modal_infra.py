"""Modal infrastructure factories (house style).

Heavy STT deps are installed here in the image, not in the local project.
HF_HOME points at the volume so model downloads cache across runs.
"""

import modal


def get_app(name: str) -> modal.App:
    return modal.App(name)


def get_image() -> modal.Image:
    return (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("ffmpeg")
        .uv_pip_install(
            "torch",
            "torchaudio",
            "transformers",
            "numpy",
            "onnx",          # IndicConformer remote code needs onnx/onnxruntime
            "onnxruntime",
            "huggingface_hub",
            "pydantic",
            "pyyaml",
            "pydantic-settings",
        )
        .env({"HF_HOME": "/vol/hf_cache"})
    )


def get_volume(name: str) -> modal.Volume:
    return modal.Volume.from_name(name, create_if_missing=True)
