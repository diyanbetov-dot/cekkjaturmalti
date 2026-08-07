# -*- coding: utf-8 -*-
import sys, os
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, ".")

from Essentials.app import spellchecker, article_phrase_rules

print("1. _strict_dictionary_tail('bahar'):", repr(article_phrase_rules._strict_dictionary_tail("bahar")))
print("2. correct_word('bahar'):", repr(spellchecker.correct_word("bahar")))
print("3. preposition_article_form('sal', 'baħar'):", repr(article_phrase_rules.preposition_article_form("sal", "baħar")))
print("4. preposition_article_form('sal', 'bahar'):", repr(article_phrase_rules.preposition_article_form("sal", "bahar")))

# Now let's trace inside correct_text_rich for "mort sal bahar biex nowm ftit."
res = spellchecker.correct_text_rich("mort sal bahar biex nowm ftit.")
print("5. Full output:", repr(res["corrected_text"]))
for t in res["tokens"]:
    if t.get("original") == "sal bahar":
        print("   Phrase Token:", t)
