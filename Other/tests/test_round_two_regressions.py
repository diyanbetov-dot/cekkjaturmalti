# -*- coding: utf-8 -*-
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Essentials import app


spellchecker = app.spellchecker


def assert_text(source, expected):
    actual = spellchecker.correct_text_rich(
        source,
        edit_distance_tolerance=2,
    )["corrected_text"]
    assert actual == expected, f"{source!r}: expected {expected!r}, got {actual!r}"


def choices_for(source):
    result = spellchecker.correct_text_rich(
        source,
        edit_distance_tolerance=2,
    )
    for token in result["tokens"]:
        if token.get("choices"):
            return token["choices"]
    return []


def words_for(source):
    return [choice["word"] for choice in choices_for(source)]


assert_text("liktar", "L-iktar.")
assert words_for("liktar") == ["L-iktar", "L'iktar", "'l-iktar"]
assert words_for("likbar") == ["L-ikbar", "L'ikbar", "'l-ikbar"]

assert_text("sqallija", "Sqallija.")
assert_text("principju", "Prinċipju.")
assert_text("anka", "Anka.")
assert_text("kelmtu", "Kelmtu.")

assert words_for("min") == ["Min", "Minn"]
assert words_for("Qalu") == ["Qalu", "Qallu"]
assert words_for("hu") == ["Hu", "Ħu", "U"]
assert words_for("Ghal") == ["Għal", "Għall-"]

assert spellchecker.meaning_for("hu") == "he"
assert spellchecker.meaning_for("ħu") == "brother, male sibling"
assert spellchecker.meaning_for("u") == "and"

assert_text("faqqalu", "Faqqagħlu.")
assert_text("xi hadd", "Xi ħadd.")
assert_text("ghall xi hadd", "Għal xi ħadd.")

assert_text("X'taghmel", "X'tagħmel.")
assert_text("Mhemmx", "M'hemmx.")
assert_text("GHALL GOST", "GĦALL-GOST.")
assert_text("C-cans", "C-cans.")

assert_text("Nicole marret id-dar.", "Nicole marret id-dar.")
assert_text("Diyan qal hekk.", "Diyan qal hekk.")
assert_text("Basti gie tard.", "Basti ġie tard.")
assert_text("Mort ma' Nicole.", "Mort ma' Nicole.")

assert_text("'This is clearly English and should stay unchanged.'", "'This is clearly English and should stay unchanged.'")

assert words_for("kif") == ["Kif", "Kief"]
assert words_for("kief") == ["Kief", "Kif"]

kif_meaning = spellchecker.meaning_for("kif")
kief_meaning = spellchecker.meaning_for("kief")
assert " / " in kif_meaning
assert "how" in kif_meaning.lower()
assert "prove" in kif_meaning.lower() or "beat" in kif_meaning.lower()
assert kief_meaning

print("round two regression checks passed")
