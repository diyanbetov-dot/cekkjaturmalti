# -*- coding: utf-8 -*-
import sys, os
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, ".")

from Essentials.app import spellchecker, article_phrase_rules

print("_is_article_target('bahar'):", article_phrase_rules._is_article_target("bahar"))
print("_is_article_target('baħar'):", article_phrase_rules._is_article_target("baħar"))

if hasattr(spellchecker, "correct_word"):
    print("spellchecker.correct_word('bahar'):", spellchecker.correct_word("bahar"))
