# -*- coding: utf-8 -*-
import sys, os
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, ".")

from Essentials.app import spellchecker

print("spellchecker._normalize_word('baħar'):", repr(spellchecker._normalize_word("baħar")))
