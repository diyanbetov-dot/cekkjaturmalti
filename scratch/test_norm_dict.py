# -*- coding: utf-8 -*-
import sys, os
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, ".")

from Essentials.app import spellchecker

rules = spellchecker.article_phrase_rules
w1 = "baħar"
w2 = "bahar"

print("rules.normalize('baħar'):", repr(rules.normalize(w1)))
print("rules.normalize('bahar'):", repr(rules.normalize(w2)))

print("in dictionary_set:", rules.normalize(w1) in spellchecker.dictionary_set)
print("_strict_dictionary_tail('baħar'):", repr(rules._strict_dictionary_tail(w1)))
print("_strict_dictionary_tail('bahar'):", repr(rules._strict_dictionary_tail(w2)))
