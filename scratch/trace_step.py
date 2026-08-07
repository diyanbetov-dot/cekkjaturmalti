# -*- coding: utf-8 -*-
import sys, os
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, ".")

from Essentials.app import spellchecker
import re

text = "mort sal bahar biex nowm ftit."
tokens, matches = spellchecker._tokenize_rich(text) if hasattr(spellchecker, "_tokenize_rich") else (None, None)

# Let's inspect what happens in match_split_article with word_tokens:
from Essentials.helpers.article_phrase_rules import WordToken
matches_raw = list(spellchecker.WORD_PATTERN.finditer(text))
word_tokens = [WordToken(m.group(0), m.start(), m.end()) for m in matches_raw]

print("word_tokens:", word_tokens)
match = spellchecker.article_phrase_rules.match_split_article(word_tokens, 1)
print("match_split_article output:", match)
