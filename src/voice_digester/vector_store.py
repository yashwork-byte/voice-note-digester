"""Local vector store placeholder: sqlite-vec, two entity types (D006).

Summaries and action items get separate embeddings and separate vec tables so
task queries ("what do I need to do for the wedding") and content queries
("what did papa say about the wedding") resolve against different entities.
Embedding model is an open decision (D006); EMBEDDING_DIM is provisional.
"""

from .schema import ActionItemRecord, NoteRecord

EMBEDDING_DIM = 384  # provisional until the embedding model is chosen (D006 open)

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


def init_db() -> None:
    raise NotImplementedError("Vector store not implemented yet — see decisions.md D006")


def add_note(record: NoteRecord) -> None:
    raise NotImplementedError("Vector store not implemented yet — see decisions.md D006")


def add_action_item(record: ActionItemRecord) -> None:
    raise NotImplementedError("Vector store not implemented yet — see decisions.md D006")


if __name__ == "__main__":
    raise SystemExit("Vector store not implemented yet — see decisions.md D006")
