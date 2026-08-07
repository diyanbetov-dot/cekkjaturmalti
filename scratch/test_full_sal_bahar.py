# -*- coding: utf-8 -*-
import sys, os, traceback
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, ".")

from Essentials.app import spellchecker

orig_paf = spellchecker.article_phrase_rules.preposition_article_form
def logged_paf(prefix, tail):
    res = orig_paf(prefix, tail)
    print(f"DEBUG preposition_article_form({prefix!r}, {tail!r}) -> {res!r}")
    traceback.print_stack(limit=5)
    return res
spellchecker.article_phrase_rules.preposition_article_form = logged_paf

res = spellchecker.correct_text_rich("mort sal bahar biex nowm ftit.")
print("OUTPUT:", repr(res["corrected_text"]))
