from __future__ import annotations

from difflib import SequenceMatcher


def edit_distance(source, target) -> int:
    matcher = SequenceMatcher(None, source, target, autojunk=False)
    return sum(
        max(i2 - i1, j2 - j1)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes()
        if tag != "equal"
    )


def edit_set(source: str, target: str) -> set[tuple]:
    matcher = SequenceMatcher(None, source, target, autojunk=False)
    return {
        (tag, i1, i2, target[j1:j2])
        for tag, i1, i2, j1, j2 in matcher.get_opcodes()
        if tag != "equal"
    }


def correction_counts(source: str, predicted: str, target: str) -> tuple[int, int, int]:
    predicted_edits = edit_set(source, predicted)
    gold_edits = edit_set(source, target)
    true_positive = len(predicted_edits & gold_edits)
    return true_positive, len(predicted_edits - gold_edits), len(gold_edits - predicted_edits)

