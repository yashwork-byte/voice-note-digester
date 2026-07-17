"""Structured-output and storage schemas (D005, D007).

Two layers, kept deliberately distinct:
- Model-facing (`NoteDigest`): only what the model can know from the transcript.
  This is the JSON schema the llama.cpp grammar is generated from.
- Storage-facing (`NoteRecord`, `ActionItemRecord`): adds app-known metadata
  (sender, date, note ID) at indexing time — the model is never asked to guess it.
"""

from typing import Literal

from pydantic import BaseModel

Confidence = Literal["confirmed", "tentative"]


class ActionItem(BaseModel):
    task: str
    due: str | None = None  # verbatim as spoken ("by Friday"), not resolved to a date
    confidence: Confidence


class NoteDigest(BaseModel):
    """The single combined model output (D005)."""

    summary: str
    translation: str
    action_items: list[ActionItem]  # empty when the note is pure chat (D007)


class NoteRecord(BaseModel):
    """A digested note as stored/indexed (entity type 1, D006)."""

    note_id: str
    sender: str
    note_date: str  # ISO 8601
    language: str  # BCP-47-ish source language tag, e.g. "hi", "ta"
    transcript: str
    digest: NoteDigest


class ActionItemRecord(BaseModel):
    """An action item as its own searchable entity (entity type 2, D006)."""

    note_id: str
    sender: str
    note_date: str
    item: ActionItem
