# -*- coding: utf-8 -*-
import sys, os
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, ".")

from Essentials.app import spellchecker, article_phrase_rules

print("_tail_surface_variants('bahar'):", article_phrase_rules._tail_surface_variants("bahar"))
