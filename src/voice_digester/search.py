"""Search interface (D009): no intent routing — query both entity types.

The query is embedded once and run against both vec tables; results carry
their entity kind ("note" / "task") so the UI can group them. A task-shaped
query naturally ranks action items higher and vice versa; a wrong router
would silently hide the right answer, an unrouted merge at worst shows one
extra labeled section.
"""

import sqlite3

from . import vector_store


def search(query: str, top_k: int = 5, db: sqlite3.Connection | None = None,
           sender: str | None = None) -> list[dict]:
    from .embed import embed

    q = embed([query])[0]
    db = db or vector_store.connect()
    results = [{"kind": "note", **r} for r in vector_store.query_notes(db, q, top_k, sender)]
    results += [{"kind": "task", **r}
                for r in vector_store.query_action_items(db, q, top_k, sender)]
    return sorted(results, key=lambda r: r["distance"])


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        raise SystemExit('usage: python -m voice_digester.search "<query>"')
    for r in search(sys.argv[1]):
        if r["kind"] == "task":
            due = f" (due {r['due']})" if r["due"] else ""
            print(f"[task {r['confidence']:9s}] {r['task']}{due}"
                  f"  — {r['sender']}, {r['note_date']}  d={r['distance']:.3f}")
        else:
            print(f"[note           ] {r['summary'][:100]}"
                  f"  — {r['sender']}, {r['note_date']}  d={r['distance']:.3f}")
