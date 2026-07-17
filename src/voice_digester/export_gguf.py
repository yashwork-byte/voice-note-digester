"""Modal entrypoint: adapter -> merged model -> quantized GGUF (D019).

    uv run modal run --detach -m voice_digester.export_gguf --checkpoint <run_name>

Merges the LoRA adapter at /vol/checkpoints/<run_name> into the base model,
converts to GGUF with llama.cpp, and quantizes to Q4_K_M at a FIXED path:

    /vol/gguf/gemma-3-4b-it-ft-Q4_K_M.gguf

configs/gemma-3-4b-it-ft.yaml points at that path, so re-exporting a new run
updates what the Modal eval sees without touching any config. For on-device
use, download the file or push it to a HF repo.
"""

import modal

from .modal_infra import get_app, get_image, get_volume

VOL = "/vol"
GGUF_OUT = f"{VOL}/gguf/gemma-3-4b-it-ft-Q4_K_M.gguf"

app = get_app("voice-digester-export-gguf")
image = (
    get_image()
    .apt_install("build-essential", "cmake", "git")
    .uv_pip_install("peft>=0.13", "accelerate>=0.34", "gguf", "sentencepiece")
    .run_commands(
        "git clone --depth 1 https://github.com/ggml-org/llama.cpp /opt/llama.cpp",
        "cmake -S /opt/llama.cpp -B /opt/llama.cpp/build -DLLAMA_CURL=OFF",
        "cmake --build /opt/llama.cpp/build --target llama-quantize -j",
    )
    .add_local_python_source("voice_digester")
)
volume = get_volume("voice-digester")


@app.function(
    image=image,
    gpu="L40S",  # merge in bf16 quickly; conversion/quantization are CPU-bound
    volumes={VOL: volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=2 * 60 * 60,
)
def export(checkpoint: str):
    import subprocess
    from pathlib import Path

    import torch
    from peft import AutoPeftModelForCausalLM
    from transformers import AutoTokenizer

    ckpt_dir = Path(VOL) / "checkpoints" / checkpoint
    merged_dir = Path(VOL) / "merged" / checkpoint
    print(f"merging {ckpt_dir} ...")
    model = AutoPeftModelForCausalLM.from_pretrained(ckpt_dir, torch_dtype=torch.bfloat16)
    model = model.merge_and_unload()
    model.save_pretrained(merged_dir)
    AutoTokenizer.from_pretrained(ckpt_dir).save_pretrained(merged_dir)

    bf16_gguf = Path(VOL) / "gguf" / f"{checkpoint}-bf16.gguf"
    bf16_gguf.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["python", "/opt/llama.cpp/convert_hf_to_gguf.py", str(merged_dir),
         "--outfile", str(bf16_gguf), "--outtype", "bf16"],
        check=True,
    )
    subprocess.run(
        ["/opt/llama.cpp/build/bin/llama-quantize", str(bf16_gguf), GGUF_OUT, "Q4_K_M"],
        check=True,
    )
    bf16_gguf.unlink()  # keep only the deployable quant (bf16 intermediate is ~8 GB)
    volume.commit()
    print(f"wrote {GGUF_OUT}")


@app.local_entrypoint()
def main(checkpoint: str):
    export.remote(checkpoint)
