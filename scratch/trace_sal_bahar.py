# -*- coding: utf-8 -*-
import sys, os
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, ".")

from Essentials.app import spellchecker

sentence = "mort sal bahar biex nowm ftit."

# Let's inspect each line in correct_text_rich for index=1 ("sal")
word_tokens, matches = spellchecker._tokenize_rich(sentence)

print("word_tokens[1]:", word_tokens[1])
print("word_tokens[2]:", word_tokens[2])

# Let's test the conditions for index 1:
print("article_phrase_rules exists?", getattr(spellchecker, "article_phrase_rules", None) is not None)
article_match = spellchecker.article_phrase_rules.match_split_article(word_tokens, 1)
print("match_split_article(word_tokens, 1):", article_match)

preposition_contraction = spellchecker.article_phrase_rules.match_preposition_article_contraction(word_tokens, 1)
print("match_preposition_article_contraction(word_tokens, 1):", preposition_contraction)
