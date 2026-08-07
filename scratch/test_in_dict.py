# -*- coding: utf-8 -*-
import sys, os
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, ".")

from Essentials.app import spellchecker

print("'bahar' in dictionary_set?", "bahar" in spellchecker.dictionary_set)
print("'baħar' in dictionary_set?", "baħar" in spellchecker.dictionary_set)
