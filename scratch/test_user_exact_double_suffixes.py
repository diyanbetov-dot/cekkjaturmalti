"""scratch/test_user_exact_double_suffixes.py
Verify exact double-suffix suggestions:
- amilulu -> agħmilhulu
- ibatilu -> ibgħathielu
- israqomlu -> israqhomlu
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from Essentials.app import spellchecker

test_cases = [
    ("amilulu", "agħmilhulu"),
    ("ibatilu", "ibgħathielu"),
    ("israqomlu", "israqhomlu"),
    ("gamlulu", "għamlulu"),
    ("gamluhulu", "għamluhulu"),
]

print("=" * 80)
print("TESTING EXACT DOUBLE SUFFIX OVERRIDES")
print("=" * 80)

for noisy, expected in test_cases:
    suggs = spellchecker.suggest(noisy)
    corr = spellchecker.correct_word(noisy)
    print(f"Noisy: {noisy!r:<12} -> Corrected: {corr!r:<12} Expected: {expected!r:<12} Suggestions: {suggs[:4]}")
