# -*- coding: utf-8 -*-
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Essentials import app


spellchecker = app.spellchecker


def corrected_text(source: str) -> str:
    return spellchecker.correct_text_rich(
        source,
        edit_distance_tolerance=2,
    )["corrected_text"]


assert spellchecker._is_feminine_noun("ħajja") is True
assert spellchecker._noun_possessive_base_for_surface("ħajtek") == "ħajja"
assert spellchecker._noun_possessive_base_for_surface("ħajti") == "ħajja"

assert spellchecker.correct_word("ħajtek") == "ħajtek"
assert spellchecker.correct_word("hajtek") == "ħajtek"
assert spellchecker.correct_word("hajjtek") == "ħajtek"
assert spellchecker.correct_word("ħajjtek") == "ħajtek"
assert spellchecker.correct_word("hajti") == "ħajti"
assert spellchecker.correct_word("ħajti") == "ħajti"
assert spellchecker.correct_word("hajjti") == "ħajti"

assert corrected_text("hajti") == "Ħajti."
assert corrected_text("hajtek") == "Ħajtek."

assert spellchecker.correct_word("wwċuħ") == "wċuħ"
