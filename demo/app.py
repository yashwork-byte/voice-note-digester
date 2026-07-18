"""Live demo backend (D021): mic audio in the browser -> Modal inference ->
local sqlite-vec store -> search. Run with:

    make demo    (uvicorn on http://localhost:8000)

Heavy models live in the deployed Modal service (voice-digester-service);
this process only embeds (MiniLM) and stores/searches locally.
"""

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import modal
from fastapi import FastAPI, Form, UploadFile
from fastapi.staticfiles import StaticFiles

from voice_digester import vector_store
from voice_digester.schema import ActionItemRecord, NoteDigest, NoteRecord
from voice_digester.search import search as run_search

app = FastAPI(title="voice-note-digester demo")
_db = None
_digester = None


def db():
    global _db
    if _db is None:
        _db = vector_store.connect()
    return _db


def digester():
    global _digester
    if _digester is None:
        _digester = modal.Cls.from_name("voice-digester-service", "Digester")()
    return _digester


@app.post("/api/notes")
async def create_note(audio: UploadFile, sender: str = Form(...), language: str = Form(...)):
    from voice_digester.embed import embed

    payload = await audio.read()
    result = digester().process.remote(payload, language)
    digest = NoteDigest(**result["digest"])

    conn = db()
    count = conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0]
    record = NoteRecord(
        note_id=f"live-{count + 1:04d}", sender=sender, note_date=date.today().isoformat(),
        language=language, transcript=result["transcript"], digest=digest,
    )
    vector_store.add_note(conn, record, embed([digest.summary])[0])
    for item in digest.action_items:
        vector_store.add_action_item(
            conn, ActionItemRecord(note_id=record.note_id, sender=sender,
                                   note_date=record.note_date, item=item),
            embed([item.task])[0])
    return record.model_dump()


@app.get("/api/search")
def search_notes(q: str, sender: str | None = None):
    return run_search(q, top_k=6, db=db(), sender=sender or None)


@app.get("/api/notes")
def recent_notes():
    return vector_store.list_notes(db())


@app.get("/api/senders")
def get_senders():
    return vector_store.senders(db())


app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True))
