# -*- coding: utf-8 -*-
import sys, os
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, ".")

from Essentials.app import spellchecker

text = "mort nghum ilbierah"
res = spellchecker.correct_text_rich(text)
print("INPUT:", repr(text))
print("CORRECTED:", repr(res["corrected_text"]))
for t in res["tokens"]:
    print(t)
