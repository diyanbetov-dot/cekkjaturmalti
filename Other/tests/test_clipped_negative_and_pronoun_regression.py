# -*- coding: utf-8 -*-
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Essentials import app


spellchecker = app.spellchecker


def assert_word(word, expected):
    actual = spellchecker.correct_word(word)
    assert actual == expected, f"{word!r}: expected {expected!r}, got {actual!r}"


def assert_text(source, expected):
    actual = spellchecker.correct_text_rich(
        source,
        edit_distance_tolerance=2,
    )["corrected_text"]
    assert actual == expected, f"{source!r}: expected {expected!r}, got {actual!r}"


assert_word("tantx", "tantx")
assert_word("mhawx", "m'hawnx")

assert spellchecker.meaning_for("tantx") == "not much, not really"
assert spellchecker.meaning_for("m'hawnx") == "not here"
assert spellchecker.meaning_for("hu") == "he"
assert spellchecker.meaning_for("hi") == "she"
assert spellchecker.meaning_for("ħu") == "brother, male sibling"
assert spellchecker.meaning_for("ħi") == "friend"

assert_text("ma tantx", "Ma tantx.")
assert_text("mhawx hemm", "M'hawnx hemm.")
assert_text("hu mar", "Hu mar.")
assert_text("hu u hi", "Hu u hi.")

hu_result = spellchecker.correct_text_rich("hu mar", edit_distance_tolerance=2)
hu_token = [token for token in hu_result["tokens"] if token.get("original") == "hu"][0]
choice_map = {choice["word"]: choice["meaning"] for choice in hu_token.get("choices", [])}
assert choice_map.get("Hu") == "he"
assert choice_map.get("Ħu") == "brother, male sibling"

print("clipped negative and pronoun regression checks passed")
