# -*- coding: utf-8 -*-
import sys, os
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, ".")

from Essentials.app import spellchecker

print("word_tags.get('sal'):", spellchecker.word_tags.get("sal", set()))
