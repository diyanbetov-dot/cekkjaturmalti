# -*- coding: utf-8 -*-
import sys, os, inspect
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, ".")

from Essentials.app import spellchecker

# Let's inspect all tokens produced and where each token came from in spellchecker.py
res = spellchecker.correct_text_rich("ilbierah mort sal bahar biex nowm ftit.")
print("RESULT:", repr(res["corrected_text"]))
for i, token in enumerate(res["tokens"]):
    print(f"Token {i}: {token}")
