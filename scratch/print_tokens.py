# -*- coding: utf-8 -*-
import sys, os
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, ".")

from Essentials.app import spellchecker

sentence = "mort sal bahar biex nowm ftit."
res = spellchecker.correct_text_rich(sentence)

print("Final corrected text:", repr(res["corrected_text"]))
print("Tokens:")
for t in res["tokens"]:
    print(" ", t)
