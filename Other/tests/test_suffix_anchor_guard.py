# -*- coding: utf-8 -*-
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Essentials import app


spellchecker = app.spellchecker

# A suffix parse must still agree with the input consonant skeleton. This
# prevents unrelated imperative candidates from replacing an unknown word.
assert spellchecker.correct_word("sormok") == "sormok"

# The guard must not discard legitimate multi-letter suffix repairs.
assert spellchecker.correct_word("nitfalom") == "nitfagħlhom"
assert spellchecker.correct_word("ghamililhom") == "għamilhielhom"

print("suffix anchor guard checks passed")
