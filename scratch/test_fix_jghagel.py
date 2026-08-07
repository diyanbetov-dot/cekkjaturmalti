"""scratch/test_fix_jghagel.py
Verify missing_double_variants vs correct_missing_double behavior on 'jghaġel'.
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from Essentials.app import spellchecker

seed = "jghaġel"
doubled_gen = spellchecker.doubled_letter_generator

print("seed:", seed)
print("missing_double_variants(seed):", doubled_gen.missing_double_variants(seed))
print("correct_missing_double(seed):", doubled_gen.correct_missing_double(seed))

variants = list(doubled_gen.missing_double_variants(seed))
if hasattr(spellchecker, "orthographic_generator"):
    shortcut_of_variants = []
    for v in variants:
        shortcut_of_variants.extend(spellchecker.orthographic_generator.shortcut_letter_variants(v))
    print("shortcut_letter_variants of missing_double_variants:", shortcut_of_variants)
    print("Valid dictionary forms:", [v for v in shortcut_of_variants if v in spellchecker.dictionary_set])
