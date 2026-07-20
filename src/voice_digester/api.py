"""Shared FastAPI factory for the demo (D024).

Both the local demo (demo/app.py) and the deployed Modal web app
(voice_digester.web) build their API from this one factory, so the two
deployments cannot drift. Inference functions lazy-load their models on first
call; the caller passes a ready DigestConfig (with gguf_path resolved for its
environment) and a sqlite path.
"""

import tempfile
from datetime import date
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles

from . import vector_store
from .config import DigestConfig
from .schema import ActionItemRecord, NoteDigest, NoteRecord
from .search import search as run_search


def warm_models(config: DigestConfig) -> None:
    """Load STT + LID + digest and prime the llama.cpp prompt cache."""
    from .digest import digest_core
    from .lid import _classifier
    from .stt import _model

    _classifier()
    _model()
    digest_core("नमस्ते", config, sender="warmup")


def create_api(config: DigestConfig, db_path: str | Path,
               static_dir: Path | None = None, cors: bool = False) -> FastAPI:
    app = FastAPI(title="suno — voice-note digester")
    if cors:
        from fastapi.middleware.cors import CORSMiddleware

        app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                           allow_headers=["*"])
    db = vector_store.connect(Path(db_path))

    @app.post("/api/notes")
    async def create_note(audio: UploadFile, sender: str = Form(...)):
        from .digest import digest_core
        from .embed import embed
        from .lid import detect
        from .stt import decode_audio, transcribe_wav

        payload = await audio.read()
        with tempfile.NamedTemporaryFile(suffix=".webm") as f:
            f.write(payload)
            f.flush()
            wav = decode_audio(Path(f.name))
        language = detect(wav)
        transcript = transcribe_wav(wav, language)
        core = digest_core(transcript, config, sender=sender)

        count = db.execute("SELECT COUNT(*) FROM notes").fetchone()[0]
        record = NoteRecord(
            note_id=f"live-{count + 1:04d}", sender=sender,
            note_date=date.today().isoformat(), language=language, transcript=transcript,
            digest=NoteDigest(summary=core.summary, translation="",  # lazy (D023)
                              action_items=core.action_items),
        )
        vector_store.add_note(db, record, embed([core.summary])[0])
        for item in core.action_items:
            vector_store.add_action_item(
                db, ActionItemRecord(note_id=record.note_id, sender=sender,
                                     note_date=record.note_date, item=item),
                embed([item.task])[0])
        return record.model_dump() | {"language": language}

    @app.get("/api/translation")
    def translation(note_id: str):
        from .digest import digest_translation

        note = vector_store.get_note(db, note_id)
        if note is None:
            raise HTTPException(404, "unknown note")
        if not note["translation"]:
            note["translation"] = digest_translation(note["transcript"], config,
                                                     sender=note["sender"])
            vector_store.update_translation(db, note_id, note["translation"])
        return {"note_id": note_id, "translation": note["translation"]}

    @app.get("/api/search")
    def search_notes(q: str, sender: str | None = None):
        return run_search(q, top_k=6, db=db, sender=sender or None)

    @app.get("/api/notes")
    def recent_notes():
        return vector_store.list_notes(db)

    @app.get("/api/senders")
    def get_senders():
        return vector_store.senders(db)

    if static_dir is not None:
        app.mount("/", StaticFiles(directory=static_dir, html=True))
    return app
