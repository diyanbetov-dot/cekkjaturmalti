# -*- coding: utf-8 -*-
import sys, os
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, ".")

from Essentials.app import spellchecker

print("spellchecker.correct_word('bahar'):", repr(spellchecker.correct_word("bahar")))
