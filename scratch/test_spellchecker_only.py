# -*- coding: utf-8 -*-
import sys, os
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, ".")

from Essentials.app import spellchecker

res = spellchecker.correct_text_rich("mort sal bahar biex nowm ftit.")
print("SPELLCHECKER output:", repr(res["corrected_text"]))

token = [t for t in res["tokens"] if t.get("type") == "phrase"][0]
print("Phrase token corrected field:", repr(token["corrected"]))
