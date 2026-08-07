# -*- coding: utf-8 -*-
import sys, os
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, ".")

from Essentials.app import spellchecker, article_phrase_rules
from Essentials.helpers.article_phrase_rules import WordToken

words = [WordToken(w, i, i+len(w)) for i, w in enumerate("mort sal bahar biex nowm ftit".split())]

print("=== TESTING MATCHERS ON 'sal' 'bahar' ===")
split_match = article_phrase_rules.match_split_article(words, 1)
print("match_split_article(index=1):", split_match)

bare_match = article_phrase_rules.match_bare_preposition_article_phrase(words, 1)
print("match_bare_preposition_article_phrase(index=1):", bare_match)
