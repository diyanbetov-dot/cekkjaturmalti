# -*- coding: utf-8 -*-
import app


spellchecker = app.spellchecker


def assert_suggestions(word, expected):
    suggestions = spellchecker.suggest(word, limit=8, edit_distance_tolerance=1)
    for choice in expected:
        assert choice in suggestions, f"{word!r}: missing {choice!r} in {suggestions!r}"


def assert_correct(word, expected):
    actual = spellchecker.correct_word(word)
    assert actual == expected, f"{word!r}: expected {expected!r}, got {actual!r}"


assert_correct("bilqeda", "bilqiegħda")
assert_suggestions("bilqeda", ["bilqiegħda", "bil-qiegħda"])

assert_correct("bilqiegħda", "bilqiegħda")
assert_suggestions("bilqiegħda", ["bilqiegħda", "bil-qiegħda"])

assert_correct("bizzejjed", "biżżejjed")
assert_suggestions("bizzejjed", ["biżżejjed", "biż-żejjed"])
assert spellchecker.meaning_for("biżżejjed") == "enough, sufficiently"
assert spellchecker.meaning_for("biż-żejjed") == "extra or additional"

assert_correct("ghalfejn", "għalfejn")
assert_suggestions("ghalfejn", ["għalfejn", "għal fejn"])

assert_correct("ghalkollox", "għalkollox")
assert_suggestions("ghalkollox", ["għalkollox", "għal kollox"])

assert_correct("qabelxejn", "qabelxejn")
assert_suggestions("qabelxejn", ["qabelxejn", "qabel xejn"])

assert_correct("lanqas", "lanqas")
assert_suggestions("lanqas", ["lanqas", "l-anqas"])
assert spellchecker.meaning_for("l-anqas") == "the least, the smallest"
assert spellchecker.meaning_for("l-inqas") == "the least, the smallest"

print("lexicalized form checks passed")
