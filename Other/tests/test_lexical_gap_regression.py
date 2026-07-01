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


assert_word("verita", "verità")
assert_word("inteligenti", "intelliġenti")
assert_word("koala", "koala")
assert_word("panda", "panda")
assert_word("innerdja", "innerdja")
assert_word("innerdjat", "innerdjat")
assert_word("jinnerdja", "jinnerdja")
assert_word("tinnerdjawx", "tinnerdjawx")

assert_text("Verita hi importanti", "Verità hi importanti.")
assert_text("Hu koala u panda", "Hu koala u panda.")
assert_text("kien innerdjat", "Kien innerdjat.")
assert_text("hu inteligenti", "Hu intelliġenti.")

panda_result = spellchecker.correct_text_rich(
    "Hu koala u panda",
    edit_distance_tolerance=2,
)
panda_tokens = [token for token in panda_result["tokens"] if token.get("original") == "panda"]
assert panda_tokens, "expected panda token in sentence result"
assert [choice["word"] for choice in panda_tokens[0].get("choices", [])] == []

for form, meaning_fragment in {
    "innerdja": "irritate",
    "jinnerdja": "irritate",
    "tinnerdjawx": "don't irritate",
    "innerdjat": "irritated",
}.items():
    meaning = spellchecker.meaning_for(form)
    assert meaning_fragment in meaning, f"{form!r}: expected {meaning_fragment!r} in {meaning!r}"

print("lexical gap regression checks passed")
