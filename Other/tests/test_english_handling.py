# -*- coding: utf-8 -*-
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Essentials import app


spellchecker = app.spellchecker


def rich(source: str) -> dict:
    return spellchecker.correct_text_rich(source, edit_distance_tolerance=2)


def token_for(source: str, original: str) -> dict:
    result = rich(source)
    for token in result["tokens"]:
        if token.get("original") == original:
            return token
    raise AssertionError(f"{source!r}: token {original!r} not found")


result = rich("'Hi', fejn tista' ċċempel please")
assert result["corrected_text"] == "Hi, fejn tista' ċċempel please."
assert token_for("'Hi', fejn", "'Hi'")["corrected"] == "Hi"

please = token_for("fejn please issa", "please")
assert please["type"] == "english_phrase"
assert please["corrected"] == "please"
assert please["maltese_suggestion"] is None

car = token_for("ġiet car illum", "car")
assert car["type"] == "english_phrase"
assert car["maltese_suggestion"] == ["karozza"]

max_word = token_for("għandi max illum", "max")
assert max_word["type"] == "english_phrase"
assert max_word["maltese_suggestion"] == ["mgħax"]
assert max_word["english_note"] == "il-kelma massimu bl-Ingliż"

misspelled = token_for("għandi aotomatic illum", "aotomatic")
assert misspelled["type"] == "word"
assert misspelled.get("unrecognized") is True
