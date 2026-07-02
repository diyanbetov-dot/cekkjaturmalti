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


def token_for(source: str, original: str):
    result = spellchecker.correct_text_rich(
        source,
        edit_distance_tolerance=2,
    )
    for token in result["tokens"]:
        if token.get("original") == original:
            return token
    raise AssertionError(f"{source!r}: token {original!r} not found")


assert corrected_text("qal, programm") == "Qal, programm."
assert corrected_text("qal programm") == "Qal iprogramm."

assert corrected_text("misterjuz") == "Misterjuż."
assert corrected_text("gustuz") == "Gustuż."
assert corrected_text("ma' gustuz") == "ma' gustuż."

gustuz_token = token_for("ma' gustuz", "ma' gustuz")
assert gustuz_token["corrected"] == "ma' gustuż"
assert gustuz_token.get("ambiguous") is False
assert gustuz_token.get("crucial") in {False, None}

assert corrected_text("kontestant") == "Kontestant."
assert corrected_text("apparti") == "Apparti."
assert corrected_text("sew") == "Sew."

assert spellchecker.correct_word("misterjuż") == "misterjuż"
assert spellchecker.correct_word("gustuż") == "gustuż"
assert spellchecker.correct_word("kontestant") == "kontestant"
assert spellchecker.correct_word("apparti") == "apparti"
assert spellchecker.correct_word("sew") == "sew"

print("contextual i and lexical coverage checks passed")
