"""Live demo backend (D023): ALL inference runs locally on this machine.

    make fetch-model   (once: pulls the fine-tuned GGUF from the Modal volume)
    make demo          (uvicorn on http://localhost:8000)

Modal is used for training and evaluation only. Latency levers (D023):
lazy translation (generated on demand), persistent llama.cpp instance with
prompt-prefix KV reuse, int8-quantized STT.
"""

import sys
import tempfile
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles

from voice_digester import vector_store
from voice_digester.config import DigestConfig
from voice_digester.paths import project_root
from voice_digester.schema import ActionItemRecord, NoteDigest, NoteRecord
from voice_digester.search import search as run_search

LOCAL_GGUF = project_root() / "data" / "models" / "gemma-3-4b-it-ft-Q4_K_M.gguf"

app = FastAPI(title="voice-note-digester demo")
_db = None
_config = None


@app.on_event("startup")
def warmup():
    """Load all models and prime the llama.cpp prompt cache in the background,
    so the first real note pays ~10 s instead of ~3 min (D023)."""
    import threading

    def _warm():
        try:
            from voice_digester.digest import digest_core
            from voice_digester.lid import _classifier
            from voice_digester.stt import _model

            _classifier()
            _model(True)
            digest_core("नमस्ते", config(), sender="warmup")
            print("warmup complete — models resident, prompt cache primed")
        except Exception as e:
            print(f"warmup failed (first request will be slow): {e}")

    threading.Thread(target=_warm, daemon=True).start()


def db():
    global _db
    if _db is None:
        _db = vector_store.connect()
    return _db


def config() -> DigestConfig:
    global _config
    if _config is None:
        c = DigestConfig.from_yaml("gemma-3-4b-it-ft.yaml")
        if not Path(c.gguf_path).exists():  # the yaml points at the Modal volume
            if not LOCAL_GGUF.exists():
                raise HTTPException(500, f"model not found — run `make fetch-model` "
                                         f"(expects {LOCAL_GGUF})")
            c.gguf_path = str(LOCAL_GGUF)
        _config = c
    return _config


@app.post("/api/notes")
async def create_note(audio: UploadFile, sender: str = Form(...)):
    from voice_digester.digest import digest_core
    from voice_digester.embed import embed
    from voice_digester.lid import detect
    from voice_digester.stt import decode_audio, transcribe_wav

    payload = await audio.read()
    with tempfile.NamedTemporaryFile(suffix=".webm") as f:
        f.write(payload)
        f.flush()
        wav = decode_audio(Path(f.name))
    language = detect(wav)
    transcript = transcribe_wav(wav, language, quantized=True)
    core = digest_core(transcript, config(), sender=sender)

    conn = db()
    count = conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0]
    record = NoteRecord(
        note_id=f"live-{count + 1:04d}", sender=sender, note_date=date.today().isoformat(),
        language=language,
        transcript=transcript,
        digest=NoteDigest(summary=core.summary, translation="",  # lazy (D023)
                          action_items=core.action_items),
    )
    vector_store.add_note(conn, record, embed([core.summary])[0])
    for item in core.action_items:
        vector_store.add_action_item(
            conn, ActionItemRecord(note_id=record.note_id, sender=sender,
                                   note_date=record.note_date, item=item),
            embed([item.task])[0])
    return record.model_dump() | {"language": language}


@app.get("/api/translation")
def translation(note_id: str):
    from voice_digester.digest import digest_translation

    note = vector_store.get_note(db(), note_id)
    if note is None:
        raise HTTPException(404, "unknown note")
    if not note["translation"]:
        note["translation"] = digest_translation(note["transcript"], config(),
                                                 sender=note["sender"])
        vector_store.update_translation(db(), note_id, note["translation"])
    return {"note_id": note_id, "translation": note["translation"]}


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
