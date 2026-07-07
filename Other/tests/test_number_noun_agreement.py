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


def phrase_token(source: str):
    result = spellchecker.correct_text_rich(
        source,
        edit_distance_tolerance=2,
    )
    for token in result["tokens"]:
        if token.get("type") == "phrase":
            return token
    raise AssertionError(f"{source!r}: no phrase token found")


assert corrected_text("ħames bozza") == "Ħames bozoz."
assert corrected_text("ħames baqra") == "Ħames baqriet."
assert corrected_text("ħamest bozoz") == "Ħamest bozza."
assert corrected_text("għoxrin bozoz") == "Għoxrin bozza."
assert corrected_text("ħames tfal") == "Ħamest itfal."
assert corrected_text("il ħames bozoz") == "Il-ħames bozoz."

ambiguous_token = phrase_token("il-ħames bozoz")
assert ambiguous_token["corrected"] == "Il-ħames bozoz"
assert ambiguous_token["ambiguous"] is True
assert ambiguous_token["crucial"] is True
assert [choice["word"] for choice in ambiguous_token["choices"]] == [
    "Il-ħames bozoz",
    "Il-ħames bozza",
]

singular_ambiguous = phrase_token("il-ħames bozza")
assert [choice["word"] for choice in singular_ambiguous["choices"]] == [
    "Il-ħames bozza",
    "Il-ħames bozoz",
]

print("number+noun agreement regression checks passed")
