import sqlite3

import pytest

from voice_digester import vector_store
from voice_digester.embed import EMBEDDING_DIM
from voice_digester.schema import ActionItem, ActionItemRecord, NoteDigest, NoteRecord


def vec(hot: int) -> list[float]:
    v = [0.0] * EMBEDDING_DIM
    v[hot] = 1.0
    return v


def note(note_id, sender, summary):
    return NoteRecord(note_id=note_id, sender=sender, note_date="2026-06-01", language="hi",
                      transcript="t", digest=NoteDigest(summary=summary, translation="tr",
                                                        action_items=[]))


@pytest.fixture
def db(tmp_path):
    return vector_store.connect(tmp_path / "test.db")


def test_note_round_trip_ranks_by_similarity(db):
    vector_store.add_note(db, note("n1", "Papa", "wedding venue advance"), vec(0))
    vector_store.add_note(db, note("n2", "Meera", "gossip about Neha"), vec(1))
    top = vector_store.query_notes(db, vec(0), top_k=2)
    assert [r["note_id"] for r in top] == ["n1", "n2"]
    assert top[0]["sender"] == "Papa" and top[0]["distance"] < top[1]["distance"]


def test_action_items_are_a_separate_entity_type(db):
    vector_store.add_note(db, note("n1", "Papa", "wedding planning"), vec(0))
    item = ActionItem(task="Book the venue", due="by Friday", confidence="confirmed")
    vector_store.add_action_item(
        db, ActionItemRecord(note_id="n1", sender="Papa", note_date="2026-06-01", item=item), vec(2))
    tasks = vector_store.query_action_items(db, vec(2), top_k=1)
    assert tasks[0]["task"] == "Book the venue"
    assert tasks[0]["confidence"] == "confirmed" and tasks[0]["due"] == "by Friday"
    # the two vec tables don't leak into each other
    assert vector_store.query_notes(db, vec(2), top_k=1)[0]["note_id"] == "n1"


def test_confidence_vocabulary_enforced_in_sql(db):
    with pytest.raises(sqlite3.IntegrityError):
        db.execute("INSERT INTO action_items (note_id, sender, note_date, task, due, confidence)"
                   " VALUES ('n1', 'x', '2026-06-01', 't', NULL, 'probably')")
