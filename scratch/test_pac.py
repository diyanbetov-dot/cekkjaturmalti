# -*- coding: utf-8 -*-
import sys, os
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, ".")

from Essentials.app import spellchecker

rules = spellchecker.article_phrase_rules
print("preposition_article_choices('sal', 'baħar', 'mort'):", rules.preposition_article_choices("sal", "baħar", "mort"))
print("preposition_article_choices('sal', 'bahar', 'mort'):", rules.preposition_article_choices("sal", "bahar", "mort"))
