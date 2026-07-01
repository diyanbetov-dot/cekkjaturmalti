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


def assert_only_choice(source, expected_word):
    result = spellchecker.correct_text_rich(
        source,
        edit_distance_tolerance=2,
    )
    phrase_tokens = [token for token in result["tokens"] if token.get("choices")]
    assert phrase_tokens, f"{source!r}: expected a suggestion-bearing token"
    choices = phrase_tokens[0]["choices"]
    actual_words = [choice["word"] for choice in choices]
    assert actual_words == [expected_word], (
        f"{source!r}: expected only {expected_word!r}, got {actual_words!r}"
    )


assert_text("ghall gost", "għall-gost.")
assert_text("ghall skola", "għas-skola.")
assert_text("ghall gieh", "għall-ġieħ.")
assert_text("ghall ghalqa", "għall-għalqa.")
assert_text("ghall xi hadd", "Għal xi ħadd.")

assert_text("f sormok", "f'sormok.")
assert_only_choice("f sormok", "f'sormok")

assert_text("f idejk", "f'idejk.")
assert_only_choice("f idejk", "f'idejk")

assert_text("b idejk", "b'idejk.")
assert_only_choice("b idejk", "b'idejk")

assert_text("bla", "Bla.")
assert_text("dal ftit", "dal-ftit.")

exact_contracted = spellchecker.correct_text_rich(
    "f'sormok b'idejk f'hawn b'hekk",
    edit_distance_tolerance=2,
)
assert exact_contracted["corrected_text"] == "F'sormok b'idejk f'hawn b'hekk."
for token in exact_contracted["tokens"]:
    if token.get("original") in {"f'sormok", "b'idejk", "f'hawn", "b'hekk"}:
        assert token.get("choices") == [], (
            f"{token['original']!r}: expected no ambiguity choices, got {token.get('choices')!r}"
        )

print("contraction phrase regression checks passed")
