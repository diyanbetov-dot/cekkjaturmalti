from __future__ import annotations

import re
from difflib import SequenceMatcher

DIACRITIC_FOLD = str.maketrans(
    {"ċ": "c", "ġ": "g", "ħ": "h", "ż": "z", "Ċ": "C", "Ġ": "G", "Ħ": "H", "Ż": "Z"}
)


def _fold(text: str) -> str:
    return text.translate(DIACRITIC_FOLD).replace("għ", "gh").replace("Għ", "Gh")


def classify_edit(original: str, replacement: str) -> tuple[str, str]:
    combined = original + replacement
    if original.casefold() == replacement.casefold():
        return "capitalization", "Capitalization changed from contextual evidence"
    if _fold(original).casefold() == _fold(replacement).casefold():
        return "spelling", "Missing or incorrect Maltese diacritic"
    if "għ" in replacement.casefold() and "għ" not in original.casefold():
        return "spelling", "The neural model restored għ"
    if set(combined).issubset(set(" \t\r\n")):
        return "spacing", "Word spacing was adjusted"
    if "'" in combined or "’" in combined:
        return "apostrophe", "Apostrophe usage was adjusted"
    if "-" in combined:
        return "hyphenation", "Hyphenation or article attachment was adjusted"
    if not re.search(r"\w", combined, re.UNICODE):
        return "punctuation", "Punctuation was adjusted"
    if " " in original or " " in replacement:
        return "word_boundary", "A word boundary or short phrase was corrected"
    if abs(len(original) - len(replacement)) >= 2:
        return "morphology", "The learned correction changes the word form"
    return "spelling", "The neural model predicts this spelling"


def structured_edits(
    original: str,
    corrected: str,
    action_confidences: list[float],
    alternative_provider,
) -> list[dict]:
    matcher = SequenceMatcher(None, original, corrected, autojunk=False)
    edits: list[dict] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        original_text = original[i1:i2]
        replacement = corrected[j1:j2]
        positions = range(i1, i2) if i2 > i1 else range(max(0, i1 - 1), min(len(original), i1 + 1))
        confidence_values = [
            action_confidences[index]
            for index in positions
            if index < len(action_confidences)
        ]
        confidence = (
            sum(confidence_values) / len(confidence_values)
            if confidence_values
            else 0.0
        )
        category, explanation = classify_edit(original_text, replacement)
        alternatives = alternative_provider(i1, i2, replacement, original_text)
        edits.append(
            {
                "original": original_text,
                "replacement": replacement,
                "alternatives": alternatives,
                "type": category,
                "confidence": round(confidence, 4),
                "start": i1,
                "end": i2,
                "corrected_start": j1,
                "corrected_end": j2,
                "explanation": explanation,
            }
        )
    return edits

