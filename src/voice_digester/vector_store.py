"""Local vector store: sqlite-vec, two entity types (D006).

Summaries and action items get separate embeddings and separate vec tables so
task queries ("what do I need to do for the wedding") and content queries
("what did papa say about the wedding") resolve against different entities.
The vec tables share rowids with their metadata tables.
"""

import sqlite3
import struct
from pathlib import Path

from .embed import EMBEDDING_DIM
from .paths import db_path
from .schema import ActionItemRecord, NoteRecord

DDL = f"""
CREATE TABLE IF NOT EXISTS notes (
    note_id   TEXT PRIMARY KEY,
    sender    TEXT NOT NULL,
    note_date TEXT NOT NULL,           -- ISO 8601
    language  TEXT NOT NULL,
    transcript  TEXT NOT NULL,
    summary     TEXT NOT NULL,
    translation TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS action_items (
    id         INTEGER PRIMARY KEY,
    note_id    TEXT NOT NULL REFERENCES notes(note_id),
    sender     TEXT NOT NULL,
    note_date  TEXT NOT NULL,
    task       TEXT NOT NULL,
    due        TEXT,                   -- verbatim as spoken, may be NULL
    confidence TEXT NOT NULL CHECK (confidence IN ('confirmed', 'tentative'))
);

-- sqlite-vec virtual tables; rowids mirror the metadata tables.
CREATE VIRTUAL TABLE IF NOT EXISTS vec_notes USING vec0(
    embedding float[{EMBEDDING_DIM}]
);
CREATE VIRTUAL TABLE IF NOT EXISTS vec_action_items USING vec0(
    embedding float[{EMBEDDING_DIM}]
);
"""


def _f32(vector: list[float]) -> bytes:
    return struct.pack(f"{len(vector)}f", *vector)


def connect(path: Path | None = None) -> sqlite3.Connection:
    import sqlite_vec

    # check_same_thread=False: the demo's FastAPI serves from a threadpool; the
    # single shared connection is fine at single-user demo scale.
    db = sqlite3.connect(str(path or db_path()), check_same_thread=False)
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)
    db.executescript(DDL)
    return db


def add_note(db: sqlite3.Connection, record: NoteRecord, embedding: list[float]) -> None:
    """Embedding is of the digest summary."""
    cur = db.execute(
        "INSERT INTO notes (note_id, sender, note_date, language, transcript, summary, translation)"
        " VALUES (?, ?, ?, ?, ?, ?, ?)",
        (record.note_id, record.sender, record.note_date, record.language,
         record.transcript, record.digest.summary, record.digest.translation),
    )
    db.execute("INSERT INTO vec_notes (rowid, embedding) VALUES (?, ?)",
               (cur.lastrowid, _f32(embedding)))
    db.commit()


def add_action_item(db: sqlite3.Connection, record: ActionItemRecord, embedding: list[float]) -> None:
    """Embedding is of the task text."""
    cur = db.execute(
        "INSERT INTO action_items (note_id, sender, note_date, task, due, confidence)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        (record.note_id, record.sender, record.note_date,
         record.item.task, record.item.due, record.item.confidence),
    )
    db.execute("INSERT INTO vec_action_items (rowid, embedding) VALUES (?, ?)",
               (cur.lastrowid, _f32(embedding)))
    db.commit()


def _sender_filter(rows: list[dict], sender: str | None, top_k: int) -> list[dict]:
    if sender:
        rows = [r for r in rows if r["sender"].lower() == sender.lower()]
    return rows[:top_k]


def query_notes(db: sqlite3.Connection, embedding: list[float], top_k: int = 5,
                sender: str | None = None) -> list[dict]:
    # Sender is metadata, not semantics (D021): over-fetch the KNN, filter in SQL-land.
    k = top_k * 6 if sender else top_k
    rows = db.execute(
        "SELECT n.note_id, n.sender, n.note_date, n.summary, v.distance"
        " FROM vec_notes v JOIN notes n ON n.rowid = v.rowid"
        " WHERE v.embedding MATCH ? AND k = ? ORDER BY v.distance",
        (_f32(embedding), k),
    ).fetchall()
    return _sender_filter(
        [{"note_id": r[0], "sender": r[1], "note_date": r[2], "summary": r[3],
          "distance": r[4]} for r in rows], sender, top_k)


def query_action_items(db: sqlite3.Connection, embedding: list[float], top_k: int = 5,
                       sender: str | None = None) -> list[dict]:
    k = top_k * 6 if sender else top_k
    rows = db.execute(
        "SELECT a.note_id, a.sender, a.note_date, a.task, a.due, a.confidence, v.distance"
        " FROM vec_action_items v JOIN action_items a ON a.rowid = v.rowid"
        " WHERE v.embedding MATCH ? AND k = ? ORDER BY v.distance",
        (_f32(embedding), k),
    ).fetchall()
    return _sender_filter(
        [{"note_id": r[0], "sender": r[1], "note_date": r[2], "task": r[3],
          "due": r[4], "confidence": r[5], "distance": r[6]} for r in rows], sender, top_k)


def list_notes(db: sqlite3.Connection, limit: int = 20) -> list[dict]:
    rows = db.execute(
        "SELECT note_id, sender, note_date, language, summary FROM notes"
        " ORDER BY rowid DESC LIMIT ?", (limit,)).fetchall()
    return [{"note_id": r[0], "sender": r[1], "note_date": r[2], "language": r[3],
             "summary": r[4]} for r in rows]


def senders(db: sqlite3.Connection) -> list[str]:
    return [r[0] for r in db.execute("SELECT DISTINCT sender FROM notes ORDER BY sender")]
