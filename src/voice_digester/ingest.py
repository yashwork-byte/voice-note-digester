"""Build the local store from saved digest-eval predictions (D017).

    uv run --group store python -m voice_digester.ingest [results-json]

Populates data/notes.db from an eval run's predictions — no model runs, so it
is fast and cheap. Defaults to the end-to-end (transcript-source) results.
Sender/language come from the eval notes; note dates are synthetic (spread
deterministically over recent weeks) since the gold set has none. The store
is rebuilt from scratch each run.

The real runtime path (audio in, store out) is pipeline.process_note().
"""

import json
import sys
from datetime import date, timedelta
from pathlib import Path

from . import vector_store
from .eval_data import load_eval_notes
from .paths import db_path, processed_dir
from .schema import ActionItemRecord, NoteDigest, NoteRecord

DEFAULT_RESULTS = "digest_eval_gemma-3-4b-it-promptv2_transcript.json"


def main() -> None:
    results_path = Path(sys.argv[1]) if len(sys.argv) > 1 else processed_dir() / DEFAULT_RESULTS
    rows = {r["note_id"]: r for r in json.loads(results_path.read_text())}
    rt_path = processed_dir() / "asr_round_trip.json"
    transcripts = ({r["note_id"]: r["transcript"] for r in json.loads(rt_path.read_text())}
                   if rt_path.exists() else {})
    notes = load_eval_notes()

    db_path().unlink(missing_ok=True)
    db = vector_store.connect()
    from .embed import embed

    n_notes = n_items = 0
    base = date(2026, 6, 1)
    for i, note in enumerate(notes):
        row = rows.get(note.note_id)
        if not row or not row["parse_ok"]:
            continue
        digest = NoteDigest(**row["pred"])
        note_date = (base + timedelta(days=i % 45)).isoformat()
        record = NoteRecord(note_id=note.note_id, sender=note.sender, note_date=note_date,
                            language=note.language,
                            transcript=transcripts.get(note.note_id, note.script),
                            digest=digest)
        vector_store.add_note(db, record, embed([digest.summary])[0])
        n_notes += 1
        for item in digest.action_items:
            vector_store.add_action_item(
                db, ActionItemRecord(note_id=note.note_id, sender=note.sender,
                                     note_date=note_date, item=item),
                embed([item.task])[0])
            n_items += 1
    print(f"{db_path()}: {n_notes} notes, {n_items} action items (from {results_path.name})")


if __name__ == "__main__":
    main()
