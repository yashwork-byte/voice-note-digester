"""Digest-stage eval on the gold set (D004/D015 validation). Runs on Modal.

    uv run --group eval modal run -m voice_digester.evaluate \
        --config-file-name gemma-3-4b-it.yaml --source transcript

--source script:     gold scripts as input — isolates digest quality.
--source transcript: IndicConformer transcripts (data/processed/
                     asr_round_trip.json) — the real pipeline, including STT
                     errors. Run both; the gap between them is the STT cost.

The container runs the identical GGUF through llama.cpp (CPU) — same model
the phone would run; Modal keeps the dev laptop free (D014 amendment). Only
synthetic data leaves the machine (D001). Strict scoring: an unparseable
output scores 0 on every metric. Hallucination rate is the fraction of
no-item notes that got at least one predicted item. Results land in
data/processed/.
"""

import json
from collections import defaultdict
from statistics import mean

from .config import DigestConfig
from .modal_infra import get_app, get_image, get_volume
from .paths import processed_dir

app = get_app("voice-digester-digest-eval")
image = (
    get_image()
    .apt_install("build-essential", "cmake")
    .uv_pip_install("llama-cpp-python>=0.3", "sacrebleu>=2.4")
    .add_local_python_source("voice_digester")
    .add_local_dir("data/eval_scripts", "/root/data/eval_scripts")
    .add_local_file("data/processed/asr_round_trip.json", "/root/data/asr_round_trip.json")
)
volume = get_volume("voice-digester")


def _chrf(hyp: str, ref: str) -> float:
    from sacrebleu import sentence_chrf

    return round(sentence_chrf(hyp, [ref]).score, 1)


def evaluate_note(note, text: str, config: DigestConfig) -> dict:
    from .digest import digest
    from .eval_metrics import action_item_prf, confidence_accuracy

    row = {"note_id": note.note_id, "language": note.language, "phrasing": note.phrasing}
    try:
        pred = digest(text, config)
    except Exception as e:
        return row | {"parse_ok": False, "error": str(e)[:200], "ai_precision": 0.0,
                      "ai_recall": 0.0, "ai_f1": 0.0, "confidence_acc": None,
                      "chrf_translation": 0.0, "chrf_summary": 0.0,
                      "hallucinated": bool(not note.gold.action_items)}
    p, r, f1 = action_item_prf(pred.action_items, note.gold.action_items)
    return row | {
        "parse_ok": True,
        "ai_precision": p, "ai_recall": r, "ai_f1": f1,
        "confidence_acc": confidence_accuracy(pred.action_items, note.gold.action_items),
        "chrf_translation": _chrf(pred.translation, note.gold.translation),
        "chrf_summary": _chrf(pred.summary, note.gold.summary),
        "hallucinated": bool(not note.gold.action_items and pred.action_items),
        "pred": pred.model_dump(),
    }


@app.function(image=image, cpu=16.0, memory=16384, volumes={"/vol": volume}, timeout=3600)
def run_eval(config: DigestConfig, source: str, tag: str, limit: int | None = None) -> list[dict]:
    from pathlib import Path

    from .eval_data import load_eval_notes

    notes = load_eval_notes(Path("/root/data/eval_scripts"))[:limit]
    transcripts = {}
    if source == "transcript":
        rt = json.loads(Path("/root/data/asr_round_trip.json").read_text())
        transcripts = {r["note_id"]: r["transcript"] for r in rt}

    rows = []
    for note in notes:
        text = transcripts[note.note_id] if source == "transcript" else note.script
        row = evaluate_note(note, text, config)
        rows.append(row)
        print(f"{note.note_id}  parse_ok={row['parse_ok']}  ai_f1={row['ai_f1']:.2f}")

    # Also persist to the volume so a dropped local client can't lose the run:
    #   modal volume get voice-digester results/digest_eval_<tag>_<source>.json
    results_dir = Path("/vol/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / f"digest_eval_{tag}_{source}.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=1)
    )
    volume.commit()
    return rows


def _report(rows: list[dict], by: str) -> None:
    groups = defaultdict(list)
    for r in rows:
        groups[r[by] or "-"].append(r)
    for key in sorted(groups):
        g = groups[key]
        conf = [r["confidence_acc"] for r in g if r["confidence_acc"] is not None]
        conf_str = f"{mean(conf):.2f}" if conf else "n/a"
        print(f"  {key:12s} n={len(g):2d}  ai_f1={mean(r['ai_f1'] for r in g):.2f}  "
              f"conf_acc={conf_str}  chrf_tr={mean(r['chrf_translation'] for r in g):.1f}  "
              f"chrf_sum={mean(r['chrf_summary'] for r in g):.1f}")


@app.local_entrypoint()
def main(config_file_name: str = "gemma-3-4b-it.yaml", source: str = "transcript",
         limit: int | None = None):
    from .eval_data import load_eval_notes

    assert source in ("script", "transcript")
    config = DigestConfig.from_yaml(config_file_name)
    tag = config_file_name.removesuffix(".yaml")
    rows = run_eval.remote(config, source, tag, limit)
    out_path = processed_dir() / f"digest_eval_{tag}_{source}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=1))

    no_item_ids = {n.note_id for n in load_eval_notes() if not n.gold.action_items}
    no_item_rows = [r for r in rows if r["note_id"] in no_item_ids]
    print(f"\n{len(rows)} notes ({source} input) -> {out_path}")
    print(f"parse failures: {sum(1 for r in rows if not r['parse_ok'])}")
    if no_item_rows:
        print(f"hallucination rate on {len(no_item_rows)} no-item notes: "
              f"{mean(1.0 if r['hallucinated'] else 0.0 for r in no_item_rows):.2f}")
    print("by language:")
    _report(rows, "language")
    print("by phrasing:")
    _report(rows, "phrasing")
