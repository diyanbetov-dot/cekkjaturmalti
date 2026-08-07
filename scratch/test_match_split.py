# -*- coding: utf-8 -*-
import sys, os
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, ".")

from Essentials.app import spellchecker
from Essentials.helpers.article_phrase_rules import WordToken

words = [WordToken("mort", 0, 4), WordToken("sal", 5, 8), WordToken("bahar", 9, 14), WordToken("biex", 15, 19), WordToken("nowm", 20, 24), WordToken("ftit", 25, 29)]

res = spellchecker.article_phrase_rules.match_split_article(words, 1)
print("match_split_article result:", res)
