# -*- coding: utf-8 -*-
import sys, os
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, ".")

from Essentials.app import spellchecker

rules = spellchecker.article_phrase_rules
print("_strict_dictionary_tail('baħar'):", repr(rules._strict_dictionary_tail("baħar")))
print("'baħar' in dictionary_set:", "baħar" in spellchecker.dictionary_set)
print("'bahar' in dictionary_set:", "bahar" in spellchecker.dictionary_set)
