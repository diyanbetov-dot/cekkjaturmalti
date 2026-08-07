# -*- coding: utf-8 -*-
import sys, os
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, ".")

from Essentials.app import spellchecker

print("_normalize_word('baħar'):", spellchecker._normalize_word("baħar"))
print("normalize_word_cached('baħar'):", spellchecker._normalize_word_cached("baħar"))
