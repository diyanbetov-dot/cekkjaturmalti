# -*- coding: utf-8 -*-
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Essentials import app


spellchecker = app.spellchecker


def rich_result(source: str):
    return spellchecker.correct_text_rich(
        source,
        edit_distance_tolerance=2,
    )


def corrected_text(source: str) -> str:
    return rich_result(source)["corrected_text"]


def token_for(source: str, original: str):
    result = rich_result(source)
    for token in result["tokens"]:
        if token.get("original") == original:
            return token
    raise AssertionError(f"{source!r}: token {original!r} not found")


assert corrected_text("biex") == "Biex."
biex_token = token_for("biex", "biex")
assert biex_token["corrected"] == "Biex"
assert biex_token.get("ambiguous") is False
assert all(choice.get("word") != "Biegħx" for choice in biex_token.get("choices", []))

assert corrected_text("m huma") == "M'huma."
m_huma_token = token_for("m huma", "m")
assert m_huma_token["corrected"] == "M'huma"
assert m_huma_token.get("ambiguous") is False
assert m_huma_token.get("choices", []) == []

assert corrected_text("minn hekk") == "Minn hekk."
assert corrected_text("min hekk") == "Minn hekk."
min_hekk_token = token_for("min hekk", "min")
assert min_hekk_token["corrected"] == "Minn"
assert min_hekk_token.get("ambiguous") is True
assert [choice.get("word") for choice in min_hekk_token.get("choices", [])] == [
    "Minn hekk",
    "Min hekk",
]

assert corrected_text("f sormok") == "F'sormok."
f_sormok_token = token_for("f sormok", "f sormok")
assert f_sormok_token["corrected"] == "F'sormok"
assert f_sormok_token.get("ambiguous") is False

assert corrected_text("f'sormok") == "F'sormok."
f_apostrophe_token = token_for("f'sormok", "f'sormok")
assert f_apostrophe_token["corrected"] == "F'sormok"
assert f_apostrophe_token.get("ambiguous") is False

print("function word phrase checks passed")
