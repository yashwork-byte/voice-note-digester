import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from voice_digester.schema import NoteDigest

FIXTURES = Path(__file__).parent / "fixtures" / "eval_scripts"


@pytest.mark.parametrize("name", ["synth-hi-001", "synth-hien-002", "synth-ta-003"])
def test_gold_fixtures_parse_as_note_digest(name):
    payload = json.loads((FIXTURES / f"{name}.json").read_text())
    digest = NoteDigest(**payload["gold"])
    assert digest.summary and digest.translation


def test_no_action_items_is_first_class():
    payload = json.loads((FIXTURES / "synth-ta-003.json").read_text())
    assert NoteDigest(**payload["gold"]).action_items == []


def test_confidence_tag_is_closed_vocabulary():
    with pytest.raises(ValidationError):
        NoteDigest(
            summary="s",
            translation="t",
            action_items=[{"task": "call the caterer", "due": None, "confidence": "probably"}],
        )
