"""Modal entrypoint: build the int8 ONNX copy of IndicConformer (D023).

    uv run modal run -m voice_digester.quantize_stt
    uv run modal volume get --force voice-digester stt-int8 data/models/indic-conformer-int8

Runs on Modal because quantizing the 2.4 GB external-data encoder needs more
RAM+disk headroom than the 8 GB dev laptop has (it swap-thrashed twice).
This is tooling, not inference — inference stays local (D023). Output:
/vol/stt-int8 with a self-contained int8 encoder (~0.6 GB) and the fp32
external weight files pruned.
"""

import modal

from .modal_infra import get_app, get_image, get_volume

VOL = "/vol"

app = get_app("voice-digester-quantize-stt")
image = get_image().add_local_python_source("voice_digester")
volume = get_volume("voice-digester")


@app.function(image=image, cpu=8.0, memory=32768,
              volumes={VOL: volume},
              secrets=[modal.Secret.from_name("huggingface-secret")], timeout=3600)
def quantize() -> None:
    import shutil
    from pathlib import Path

    from huggingface_hub import snapshot_download
    from onnxruntime.quantization import QuantType, quantize_dynamic

    from .stt import STT_MODEL

    src = Path(snapshot_download(STT_MODEL))
    work = Path("/tmp/stt-work")
    shutil.copytree(src, work)  # real copies: onnx rejects sym/hardlinked external data

    encoder = work / "assets" / "encoder.onnx"
    # Delete EXACTLY the files the encoder proto declares as external tensor
    # data — a suffix heuristic wrongly swept in preprocessor.ts (TorchScript).
    import onnx

    proto = onnx.load(str(encoder), load_external_data=False)
    locations = {entry.value for init in proto.graph.initializer
                 for entry in init.external_data if entry.key == "location"}
    ext_data = [work / "assets" / loc for loc in locations
                if (work / "assets" / loc).is_file()]
    size_fp32 = (encoder.stat().st_size + sum(p.stat().st_size for p in ext_data)) / 1e9

    int8_path = work / "assets" / "encoder.int8.onnx"
    quantize_dynamic(str(encoder), str(int8_path), weight_type=QuantType.QInt8)
    int8_path.replace(encoder)  # int8 is single-file: external fp32 data now unused
    for p in ext_data:
        p.unlink()

    out = Path(VOL) / "stt-int8"
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(work, out)
    volume.commit()
    print(f"encoder: {size_fp32:.2f} GB fp32 -> {encoder.stat().st_size / 1e9:.2f} GB int8")
    print(f"wrote {out}")


@app.local_entrypoint()
def main():
    quantize.remote()
