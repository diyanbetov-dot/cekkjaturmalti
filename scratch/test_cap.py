# -*- coding: utf-8 -*-
import sys, os
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, ".")

from Essentials.app import spellchecker

print("_match_capitalisation('sal bahar', 'sal-baħar'):", repr(spellchecker._match_capitalisation("sal bahar", "sal-baħar")))
print("_match_capitalisation('sal bahar', 'baħar'):", repr(spellchecker._match_capitalisation("sal bahar", "baħar")))
