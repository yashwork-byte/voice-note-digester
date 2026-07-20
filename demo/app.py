"""Local demo backend (D023/D024): ALL inference runs on this machine.

    make fetch-model   (once: pulls the fine-tuned GGUF from the Modal volume)
    make demo          (uvicorn on http://localhost:8000)

Routes come from voice_digester.api.create_api (shared with the deployed
Modal web app so the two can't drift). Modal is training + evaluation only.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from voice_digester.api import create_api, warm_models
from voice_digester.config import DigestConfig
from voice_digester.paths import db_path, project_root

LOCAL_GGUF = project_root() / "data" / "models" / "gemma-3-4b-it-ft-Q4_K_M.gguf"

_config = DigestConfig.from_yaml("gemma-3-4b-it-ft.yaml")
if not Path(_config.gguf_path).exists():  # the yaml points at the Modal volume
    if not LOCAL_GGUF.exists():
        raise SystemExit(f"model not found — run `make fetch-model` (expects {LOCAL_GGUF})")
    _config.gguf_path = str(LOCAL_GGUF)

app = create_api(_config, db_path(), static_dir=Path(__file__).parent / "static",
                 local_config_js=True)


@app.on_event("startup")
def _warm() -> None:
    """Background warmup so the first real note isn't a ~3-min cold load."""
    import threading

    def run():
        try:
            warm_models(_config)
            print("warmup complete — models resident, prompt cache primed")
        except Exception as e:
            print(f"warmup failed (first request will be slow): {e}")

    threading.Thread(target=run, daemon=True).start()
