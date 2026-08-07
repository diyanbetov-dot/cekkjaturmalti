# -*- coding: utf-8 -*-
import sys, os
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, ".")

from Essentials.app import spellchecker

rules = getattr(spellchecker, "article_phrase_rules", None)
print("rules.spellchecker is None?", getattr(rules, "spellchecker", "NOT_FOUND") is None)
