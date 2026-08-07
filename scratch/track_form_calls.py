# -*- coding: utf-8 -*-
import sys, os
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, ".")

from Essentials.app import spellchecker

rules = spellchecker.article_phrase_rules
print("BEFORE correct_text_rich:")
print("  _strict_dictionary_tail('bahar'):", repr(rules._strict_dictionary_tail("bahar")))
print("  correct_word('bahar'):", repr(spellchecker.correct_word("bahar")))
print("  _strict_lookup_variants('bahar'):", list(spellchecker._strict_lookup_variants("bahar")))

res = spellchecker.correct_text_rich("mort sal bahar biex nowm ftit.")
print("AFTER correct_text_rich:")
print("  res:", res["corrected_text"])
