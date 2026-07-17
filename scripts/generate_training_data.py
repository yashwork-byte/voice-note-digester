"""Generate the fine-tuning dataset (D018): Cartesian templates x slot fills.

    uv run python scripts/generate_training_data.py

Each of the 4 language files defines 12 templates covering the D016 failure
modes: role attribution (sender-did-it / sender-handles-it notes), due-field
discipline (due = English time phrase or null), confidence policy (confirmed
vs tentative, incl. mixed notes), implicit and question phrasings, and pure
chat with task-y vocabulary. Gold labels are constructed from the same slot
fills as the text, so they are exact by construction. Inputs are noised to
STT shape (punctuation stripped; hi-en lowercased).

Every row is validated against schema.NoteDigest before writing. Output:
data/train/train.jsonl + val.jsonl (every 10th row).
"""

import itertools
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import traindata_bn
import traindata_hi
import traindata_hien
import traindata_ta
from voice_digester.schema import NoteDigest

LANGS = {"hi": traindata_hi, "hi-en": traindata_hien, "ta": traindata_ta, "bn": traindata_bn}
FILLS_WITH_ITEMS = 20
FILLS_NO_ITEMS = 28
SEED = 23


def stt_noise(script: str, lang: str) -> str:
    text = re.sub(r"[।,.!?;:—\-()'\"]", " ", script)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower() if lang == "hi-en" else text


def realize(template: dict, vocab: dict, sender: str, choice: dict, lang: str, note_id: str) -> dict:
    fmt = {"sender": sender}
    for key, (native, english) in choice.items():
        fmt[key] = native
        fmt[key[0].upper() + key[1:].upper()] = english  # {t0} native, {T0} English
    script = template["script"].format(**fmt)
    items = []
    for task_tpl, due_key, confidence in template["items"]:
        due = fmt[due_key[0].upper() + due_key[1:].upper()] if due_key in choice else due_key
        items.append({"task": task_tpl.format(**fmt), "due": due, "confidence": confidence})
    gold = {"summary": template["summary"].format(**fmt),
            "translation": template["translation"].format(**fmt),
            "action_items": items}
    NoteDigest(**gold)  # schema-validate every row before it is written
    return {"note_id": note_id, "language": lang, "register": template["register"],
            "phrasing": template["phrasing"], "sender": sender,
            "script": script, "transcript": stt_noise(script, lang), "gold": gold}


def fills_for(template: dict, vocab: dict, rng: random.Random) -> list[tuple[str, dict]]:
    keys = list(template["slots"])
    pools = [vocab[template["slots"][k]] for k in keys]
    combos = [dict(zip(keys, values)) for values in itertools.product(*pools)
              if len({v[0] for v in values}) == len(values)]  # distinct slot values
    combos = [(sender, c) for c in combos for sender in template["senders"]]
    rng.shuffle(combos)
    n = FILLS_WITH_ITEMS if template["items"] else FILLS_NO_ITEMS
    return combos[:n]


def main() -> None:
    rng = random.Random(SEED)
    rows = []
    for lang, module in LANGS.items():
        for t_idx, template in enumerate(module.TEMPLATES):
            for k, (sender, choice) in enumerate(fills_for(template, module.VOCAB, rng)):
                note_id = f"tr-{lang}-{t_idx:02d}-{k:03d}"
                rows.append(realize(template, module.VOCAB, sender, choice, lang, note_id))
    rng.shuffle(rows)

    no_items = sum(1 for r in rows if not r["gold"]["action_items"])
    tentative = sum(1 for r in rows
                    if any(i["confidence"] == "tentative" for i in r["gold"]["action_items"]))
    out_dir = Path(__file__).resolve().parents[1] / "data" / "train"
    out_dir.mkdir(parents=True, exist_ok=True)
    splits = {"val": rows[::10], "train": [r for i, r in enumerate(rows) if i % 10]}
    for name, split in splits.items():
        path = out_dir / f"{name}.jsonl"
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in split))
        print(f"{path}: {len(split)} rows")
    print(f"total {len(rows)}: {no_items} no-item ({no_items / len(rows):.0%}), "
          f"{tentative} with tentative items ({tentative / len(rows):.0%})")


if __name__ == "__main__":
    main()
