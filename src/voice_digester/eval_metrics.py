"""Layered eval metrics for action items (D007, D008, D014).

Tasks match on normalized word-overlap ≥ 0.5 (D014: generated task phrasings
are paraphrases of the gold, so exact word-set equality scores ~0 and tells us
nothing; overlap keeps matching deterministic). Empty gold + any prediction =
zero precision, so hallucinated action items on purely-conversational notes
are penalized by construction. chrF for summary/translation lives in
evaluate.py (needs sacrebleu).
"""

import re

from .schema import ActionItem

MATCH_THRESHOLD = 0.5


def _words(s: str) -> frozenset:
    s = (s or "").lower().replace("-", " ").replace("_", " ").replace("/", " ").replace("'", "").replace("’", "")
    return frozenset(re.sub(r"\s+", " ", s).strip().split())


def _overlap(a: frozenset, b: frozenset) -> float:
    """Containment: common words over the SHORTER task, so a terse-but-correct
    prediction still matches a detailed gold string (D014). The ≥2 common-word
    guard blocks degenerate one-word matches."""
    common = len(a & b)
    if common < 2:
        return 0.0
    return common / min(len(a), len(b))


def _match_pairs(pred: list[ActionItem], gold: list[ActionItem]) -> list[tuple[ActionItem, ActionItem]]:
    """Greedy best-overlap matching, each gold item used at most once."""
    pairs, unmatched = [], list(gold)
    for p in pred:
        scored = [(g, _overlap(_words(p.task), _words(g.task))) for g in unmatched]
        best = max(scored, key=lambda x: x[1], default=None)
        if best and best[1] >= MATCH_THRESHOLD:
            pairs.append((p, best[0]))
            unmatched.remove(best[0])
    return pairs


def action_item_prf(pred: list[ActionItem], gold: list[ActionItem]) -> tuple[float, float, float]:
    if not gold:
        return (1.0, 1.0, 1.0) if not pred else (0.0, 1.0, 0.0)
    tp = len(_match_pairs(pred, gold))
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(gold)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def _text_tokens(s: str) -> list[str]:
    return re.sub(r"[।,.!?;:\"'()\-–—]", " ", (s or "").lower()).split()


def wer(ref: str, hyp: str) -> float:
    """Word error rate after punctuation stripping (CTC output has none anyway)."""
    r, h = _text_tokens(ref), _text_tokens(hyp)
    if not r:
        return 0.0 if not h else 1.0
    prev = list(range(len(h) + 1))
    for i, rt in enumerate(r, 1):
        cur = [i]
        for j, ht in enumerate(h, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rt != ht)))
        prev = cur
    return prev[-1] / len(r)


def confidence_accuracy(pred: list[ActionItem], gold: list[ActionItem]) -> float | None:
    """Fraction of task-matched items with the right confirmed/tentative tag.

    Returns None when nothing matched (undefined, not zero).
    """
    pairs = _match_pairs(pred, gold)
    if not pairs:
        return None
    return sum(p.confidence == g.confidence for p, g in pairs) / len(pairs)
