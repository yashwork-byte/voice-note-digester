"""Gold eval set loader (D008, D010).

One JSONL file per language in data/eval_scripts/; one note per line. Gold
labels were authored together with the scripts, so they are exact by
construction — TTS synthesis only adds the acoustic layer.
"""

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from .paths import eval_scripts_dir
from .schema import NoteDigest

# Dominant phrasing of the note's action items (D011, adapted from the Liquid
# home-assistant taxonomy); None for notes with no action items.
Phrasing = Literal["imperative", "colloquial", "implicit", "question"]


class EvalNote(BaseModel):
    note_id: str
    language: str  # "hi", "hi-en", "ta", "bn"
    speech_register: str = Field(alias="register")  # JSONL key is "register"
    phrasing: Phrasing | None = None
    sender: str
    script: str
    gold: NoteDigest


def load_eval_notes(scripts_dir: Path | None = None) -> list[EvalNote]:
    scripts_dir = scripts_dir or eval_scripts_dir()
    notes = []
    for path in sorted(scripts_dir.glob("*.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                notes.append(EvalNote(**json.loads(line)))
    return notes
