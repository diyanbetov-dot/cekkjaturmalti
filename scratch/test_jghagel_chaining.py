"""scratch/test_jghagel_chaining.py
Test candidate generation chaining for 'jghagel'.
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from Essentials.app import spellchecker

word = "jghagel"
print(f"--- Chaining test for '{word}' ---")

# Step 1: Shortcut variants
shortcut_vars = spellchecker.orthographic_generator.shortcut_letter_variants(word)
print("1. Shortcut variants of 'jghagel':", shortcut_vars)

# Step 2: For each shortcut variant, apply doubled_letter_generator
doubled_vars = []
for var in shortcut_vars:
    res = spellchecker.doubled_letter_generator.correct_missing_double(var)
    if res:
        doubled_vars.append(res)
print("2. Doubled missing letter applied to shortcut variants:", doubled_vars)

# Step 3: Check dictionary lookup of doubled variants
valid_candidates = [v for v in doubled_vars if v in spellchecker.dictionary_set]
print("3. Valid dictionary words found:", valid_candidates)
