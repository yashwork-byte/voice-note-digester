"""The runtime pipeline (D001): one voice note in, searchable entities out.

    process_note(audio, sender, note_date, language, config)
      -> stt.transcribe -> digest.digest -> embed -> vector_store

This is the code path the on-device app runs; every stage is config-driven,
so a fine-tuned digest model slots in as a new configs/*.yaml (gguf_repo/
gguf_file) with no code change. Needs the stt, digest, and store dependency
groups (or the Modal image).
"""

import sqlite3
from pathlib import Path

from . import vector_store
from .config import DigestConfig
from .schema import ActionItemRecord, NoteRecord


def process_note(audio_path: Path, sender: str, note_date: str, language: str,
                 config: DigestConfig, db: sqlite3.Connection | None = None) -> NoteRecord:
    from .digest import digest as run_digest
    from .embed import embed
    from .stt import transcribe

    transcript = transcribe(audio_path, language)
    digest = run_digest(transcript, config)
    record = NoteRecord(note_id=audio_path.stem, sender=sender, note_date=note_date,
                        language=language, transcript=transcript, digest=digest)

    db = db or vector_store.connect()
    vector_store.add_note(db, record, embed([digest.summary])[0])
    for item in digest.action_items:
        vector_store.add_action_item(
            db, ActionItemRecord(note_id=record.note_id, sender=sender,
                                 note_date=note_date, item=item),
            embed([item.task])[0])
    return record
