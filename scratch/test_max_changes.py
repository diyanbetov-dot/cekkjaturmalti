"""scratch/test_max_changes.py
Test max_changes parameter in shortcut_letter_variants for 'jghagel'.
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
orthographic = spellchecker.orthographic_generator
doubled_generator = spellchecker.doubled_letter_generator

print("max_changes=1 variants:", orthographic.shortcut_letter_variants(word, max_changes=1))
print("max_changes=2 variants:", orthographic.shortcut_letter_variants(word, max_changes=2))

vars_2 = orthographic.shortcut_letter_variants(word, max_changes=2)
for v in vars_2:
    double_res = doubled_generator.correct_missing_double(v)
    print(f"  v={v!r} -> doubled={double_res!r} in_dict={double_res in spellchecker.dictionary_set if double_res else False}")
