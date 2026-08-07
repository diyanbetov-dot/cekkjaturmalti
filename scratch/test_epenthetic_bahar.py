# -*- coding: utf-8 -*-
import sys, os
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, ".")

from Essentials.app import spellchecker

print("_epenthetic_place_surface('bahar'):", spellchecker._epenthetic_place_surface("bahar", prefer_initial_i=True))
