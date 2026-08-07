from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Sequence


@dataclass(frozen=True, slots=True)
class CorrectionMetrics:
    exact_match_accuracy: float
    token_precision: float
    token_recall: float
    token_f1: float
    edit_distance_reduction: float
    lattice_gold_coverage: float
    total_examples: int
    total_exact_matches: int


def compute_edit_distance(a: str, b: str) -> int:
    matcher = SequenceMatcher(None, a, b)
    return sum(
        max(size_a, size_b)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes()
        if tag != "equal"
        for size_a, size_b in [(i2 - i1, j2 - j1)]
    )


def compute_metrics(
    predictions: Sequence[str],
    targets: Sequence[str],
    raw_inputs: Sequence[str],
    lattice_coverages: Sequence[bool] | None = None,
) -> CorrectionMetrics:
    assert len(predictions) == len(targets) == len(raw_inputs)
    n = len(predictions)
    if n == 0:
        return CorrectionMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0)

    exact_matches = 0
    total_raw_edits = 0
    total_pred_edits = 0
    tp = 0
    fp = 0
    fn = 0

    for pred, tgt, raw in zip(predictions, targets, raw_inputs):
        if pred == tgt:
            exact_matches += 1

        raw_dist = compute_edit_distance(raw, tgt)
        pred_dist = compute_edit_distance(pred, tgt)
        total_raw_edits += raw_dist
        total_pred_edits += pred_dist

        pred_tokens = pred.split()
        tgt_tokens = tgt.split()
        raw_tokens = raw.split()

        # Token precision/recall on changed tokens relative to raw
        for p, t in zip(pred_tokens, tgt_tokens):
            if p == t:
                tp += 1
            else:
                fp += 1
                fn += 1

    accuracy = float(exact_matches) / float(n)
    precision = float(tp) / float(tp + fp) if (tp + fp) > 0 else 1.0
    recall = float(tp) / float(tp + fn) if (tp + fn) > 0 else 1.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    edit_red = (
        float(total_raw_edits - total_pred_edits) / float(total_raw_edits)
        if total_raw_edits > 0
        else 1.0
    )

    coverage = (
        sum(1 for c in lattice_coverages if c) / float(len(lattice_coverages))
        if lattice_coverages
        else 1.0
    )

    return CorrectionMetrics(
        exact_match_accuracy=accuracy,
        token_precision=precision,
        token_recall=recall,
        token_f1=f1,
        edit_distance_reduction=edit_red,
        lattice_gold_coverage=coverage,
        total_examples=n,
        total_exact_matches=exact_matches,
    )
