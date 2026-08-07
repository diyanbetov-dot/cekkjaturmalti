# -*- coding: utf-8 -*-
import sys, os, traceback
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, ".")

from Essentials.app import spellchecker

orig_cw = spellchecker.correct_word
def logged_cw(w):
    res = orig_cw(w)
    if "bahar" in w or "baħar" in w:
        print(f"DEBUG correct_word({w!r}) -> {res!r}")
        traceback.print_stack(limit=5)
    return res

spellchecker.correct_word = logged_cw

res = spellchecker.correct_text_rich("mort sal bahar biex nowm ftit.")
print("FINAL RESULT:", res["corrected_text"])
