# -*- coding: utf-8 -*-
import sys, os
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, ".")

from Essentials.app import spellchecker

rules = spellchecker.article_phrase_rules
original_norm = "sal"
next_norm_for_phrase = "bahar"
separator_between = " "
index = 2
matches_len = 6

spaced_prepositions = {
    "fl", "bl", "sal", "tal", "fil", "bil", "mil", "mill",
    "mid", "mis", "miss", "lil", "lill",
    "ghall", "għall", "ghal", "għal",
    "fir", "ghat", "għat", "ghadd", "għadd", "ghacc", "għaċċ",
    "ghatt", "għatt", "mic", "miċ",
    "il", "ir", "in", "is", "it", "id", "iċ", "iż",
}

c1 = (
    original_norm in spaced_prepositions
    or (
        rules is not None
        and rules._assimilated_prefix_canonical(original_norm)
        and (
            not spellchecker._is_verb_tagged_word(original_norm)
            or next_norm_for_phrase.startswith(original_norm[-1:])
        )
        and not any(
            str(tag).startswith(("ADVERB", "CONJ", "PRON"))
            for tag in spellchecker.word_tags.get(original_norm, set())
        )
    )
)
c2 = original_norm not in {"il", "l"}
c3 = bool(rules)
c4 = separator_between.isspace()
c5 = not (
    original_norm in {"bhal", "bħal", "ghal", "għal"}
    and rules._is_function_word_tail(next_norm_for_phrase)
)
c6 = not (
    next_norm_for_phrase in {"il", "l", "ic", "iċ", "id", "in", "ir", "is", "it", "ix", "iz", "iż"}
    and index + 2 < matches_len
)
c7 = not (
    rules._assimilated_prefix_key(next_norm_for_phrase) in {"c", "d", "n", "r", "s", "t", "x", "z"}
    and index + 2 < matches_len
)

print(f"c1 (spaced_prep): {c1}")
print(f"c2 (not il/l): {c2}")
print(f"c3 (rules): {c3}")
print(f"c4 (space): {c4}")
print(f"c5 (not func tail): {c5}")
print(f"c6 (not next il/l): {c6}")
print(f"c7 (not assimilated key): {c7}")
print(f"TOTAL CONDITION: {c1 and c2 and c3 and c4 and c5 and c6 and c7}")
