# -*- coding: utf-8 -*-
import sys, os
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, ".")

from Essentials.app import spellchecker

print("self._capitalized_name_kind('Bahar'):", repr(spellchecker._capitalized_name_kind("Bahar")))
print("self._capitalized_name_kind('Baħar'):", repr(spellchecker._capitalized_name_kind("Baħar")))
