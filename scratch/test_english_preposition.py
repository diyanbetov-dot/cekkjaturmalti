# -*- coding: utf-8 -*-
import sys, os
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, ".")

from Essentials.app import spellchecker

s1 = "ilbierah mort sal bahar biex ngħum ftit."
s2 = "mort sal bus biex immur id-dar."
s3 = "konna fil park."

print("Sentence 1:", repr(spellchecker.correct_text_rich(s1)["corrected_text"]))
print("Sentence 2:", repr(spellchecker.correct_text_rich(s2)["corrected_text"]))
print("Sentence 3:", repr(spellchecker.correct_text_rich(s3)["corrected_text"]))
