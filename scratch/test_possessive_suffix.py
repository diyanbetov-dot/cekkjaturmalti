# -*- coding: utf-8 -*-
import sys, os
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, ".")

from Essentials.app import spellchecker

print("_correct_noun_possessive_suffix('bahar'):", spellchecker._correct_noun_possessive_suffix("bahar"))
print("_correct_noun_possessive_suffix('baħar'):", spellchecker._correct_noun_possessive_suffix("baħar"))
