# -*- coding: utf-8 -*-
import sys, os
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, ".")

from Essentials.app import spellchecker

rules = spellchecker.article_phrase_rules
lookup_norm = "bahar"
corrected_tail = rules._strict_dictionary_tail(lookup_norm)
print("Step 1 _strict_dictionary_tail('bahar'):", repr(corrected_tail))

if corrected_tail is None:
    corrected_next = spellchecker.correct_word(lookup_norm)
    print("Step 2 correct_word('bahar'):", repr(corrected_next))
    corrected_tail = rules._strict_dictionary_tail(corrected_next)
    print("Step 3 _strict_dictionary_tail(corrected_next):", repr(corrected_tail))
    if corrected_tail is None:
        corrected_tail = spellchecker._normalize_word(corrected_next)
        print("Step 4 fallback to normalized corrected_next:", repr(corrected_tail))

print("FINAL corrected_tail:", repr(corrected_tail))
form = rules.preposition_article_form("sal", corrected_tail)
print("FINAL preposition_article_form('sal', corrected_tail):", repr(form))
