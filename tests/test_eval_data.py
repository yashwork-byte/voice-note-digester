"""Validates the authored gold eval set itself (D010) — composition, not just parsing."""

from collections import Counter

import pytest

from voice_digester.eval_data import load_eval_notes

NOTES = load_eval_notes()


def test_expected_size_and_languages():
    assert len(NOTES) == 68
    assert Counter(n.language for n in NOTES) == {"hi": 17, "hi-en": 17, "ta": 17, "bn": 17}


def test_note_ids_unique():
    ids = [n.note_id for n in NOTES]
    assert len(ids) == len(set(ids))


def test_adversarial_no_item_share_is_maintained():
    empty = sum(1 for n in NOTES if not n.gold.action_items)
    assert 0.25 <= empty / len(NOTES) <= 0.40


@pytest.mark.parametrize("language", ["hi", "hi-en", "ta", "bn"])
def test_each_language_covers_all_three_cases(language):
    notes = [n for n in NOTES if n.language == language]
    confidences = [i.confidence for n in notes for i in n.gold.action_items]
    assert "confirmed" in confidences
    assert "tentative" in confidences
    assert sum(1 for n in notes if not n.gold.action_items) >= 3


def test_phrasing_tags_are_consistent_and_cover_hard_styles():
    for n in NOTES:
        assert (n.phrasing is None) == (not n.gold.action_items), n.note_id
    for language in ["hi", "hi-en", "ta", "bn"]:
        phrasings = {n.phrasing for n in NOTES if n.language == language}
        assert {"imperative", "colloquial", "implicit", "question"} <= phrasings


def test_some_items_carry_due_dates_and_some_do_not():
    dues = [i.due for n in NOTES for i in n.gold.action_items]
    assert any(dues) and not all(dues)
