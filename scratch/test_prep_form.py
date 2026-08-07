# -*- coding: utf-8 -*-
import sys, os
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, ".")

from Essentials.app import spellchecker, article_phrase_rules

print("preposition_article_form('sal', 'baħar'):", article_phrase_rules.preposition_article_form("sal", "baħar"))
print("preposition_article_form('sal', 'bahar'):", article_phrase_rules.preposition_article_form("sal", "bahar"))
