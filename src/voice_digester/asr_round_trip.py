"""Modal entrypoint: D008 ASR round-trip validation of the synthesized eval audio.

    uv run modal run -m voice_digester.asr_round_trip

Transcribes every data/eval_audio/<note_id>.ogg with IndicConformer-600M and
scores WER against the authored script. High-WER outliers flag TTS artifacts
(bad synthesis) rather than model failures. Runs on Modal because the 600M
fp32 model doesn't fit comfortably on the 8 GB dev laptop; only synthetic
data leaves the machine (D001). Results land in data/processed/.

Caveat: hi-en scripts are Latin-script while the model emits Devanagari, so
hi-en "WER" measures the script mismatch too — compare within a language,
never across.
"""

import json
from collections import defaultdict
from statistics import mean, median

import modal

from .modal_infra import get_app, get_image, get_volume
from .paths import processed_dir

app = get_app("voice-digester-asr-round-trip")
image = (
    get_image()
    .add_local_python_source("voice_digester")
    .add_local_dir("data/eval_scripts", "/root/data/eval_scripts")
    .add_local_dir("data/eval_audio", "/root/data/eval_audio")
)
volume = get_volume("voice-digester")


# huggingface-secret must hold an HF_TOKEN whose account accepted the
# IndicConformer gate: modal secret create huggingface-secret HF_TOKEN=hf_xxx
@app.function(
    image=image,
    cpu=8.0,
    memory=16384,
    volumes={"/vol": volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=1800,
)
def transcribe_all() -> list[dict]:
    from pathlib import Path

    from .eval_data import load_eval_notes
    from .eval_metrics import wer
    from .stt import transcribe

    rows = []
    for note in load_eval_notes(Path("/root/data/eval_scripts")):
        audio = Path("/root/data/eval_audio") / f"{note.note_id}.ogg"
        transcript = transcribe(audio, note.language)
        rows.append({
            "note_id": note.note_id,
            "language": note.language,
            "wer": round(wer(note.script, transcript), 3),
            "transcript": transcript,
        })
        print(f"{note.note_id}  wer={rows[-1]['wer']}")
    return rows


@app.local_entrypoint()
def main():
    rows = transcribe_all.remote()
    out_path = processed_dir() / "asr_round_trip.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=1))

    by_language = defaultdict(list)
    for row in rows:
        by_language[row["language"]].append(row)
    print(f"\n{len(rows)} notes -> {out_path}")
    for language in sorted(by_language):
        wers = [r["wer"] for r in by_language[language]]
        med = median(wers)
        flagged = [r["note_id"] for r in by_language[language] if r["wer"] > max(2 * med, 0.5)]
        print(f"{language}: mean={mean(wers):.3f} median={med:.3f} max={max(wers):.3f}"
              + (f"  OUTLIERS: {', '.join(flagged)}" if flagged else ""))
    print("(hi-en WER includes Devanagari-vs-Latin script mismatch — see module docstring)")
