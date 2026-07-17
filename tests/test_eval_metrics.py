from voice_digester.eval_metrics import action_item_prf, confidence_accuracy, wer
from voice_digester.schema import ActionItem


def item(task, confidence="confirmed"):
    return ActionItem(task=task, confidence=confidence)


def test_hallucinated_item_on_pure_chat_note_zeroes_precision():
    assert action_item_prf([item("Book the venue")], []) == (0.0, 1.0, 0.0)


def test_correctly_empty_prediction_is_perfect():
    assert action_item_prf([], []) == (1.0, 1.0, 1.0)


def test_partial_match_prf():
    pred = [item("Pay the advance to the hall owner"), item("Buy flowers")]
    gold = [item("pay the advance to the hall-owner"), item("Finalize the menu")]
    p, r, f1 = action_item_prf(pred, gold)
    assert (p, r) == (0.5, 0.5)
    assert f1 == 0.5


def test_paraphrased_task_matches_on_overlap():
    pred = [item("Give the hall owner the advance")]
    gold = [item("Pay the advance to the hall owner")]
    assert action_item_prf(pred, gold) == (1.0, 1.0, 1.0)
    assert action_item_prf([item("Water the garden plants")], gold) == (0.0, 0.0, 0.0)


def test_terse_but_correct_prediction_matches_detailed_gold():
    gold = [item("Stay home for the repairman coming for the window work")]
    assert action_item_prf([item("Stay at home")], gold) == (1.0, 1.0, 1.0)
    # one shared word is not evidence of the same task
    assert action_item_prf([item("the plumber")], gold) == (0.0, 0.0, 0.0)


def test_wer_ignores_punctuation_and_counts_edits():
    assert wer("शुक्रवार तक हॉल वाले को एडवांस दे देना।", "शुक्रवार तक हॉल वाले को एडवांस दे देना") == 0.0
    assert wer("एक दो तीन चार पाँच", "एक दो तीन चार छह") == 0.2
    assert wer("", "kuch bhi") == 1.0
    assert wer("एक दो", "") == 1.0


def test_confidence_accuracy_on_matched_tasks():
    pred = [item("Think about the venue", "confirmed")]
    gold = [item("think about the venue", "tentative")]
    assert confidence_accuracy(pred, gold) == 0.0
    assert confidence_accuracy(pred, [item("unrelated task")]) is None
