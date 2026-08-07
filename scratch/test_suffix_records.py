"""scratch/test_suffix_records.py
Check suffix_generator candidates_for_surface for double-suffixed verbs.
INSPECT ONLY.
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from Essentials.app import spellchecker

sg = spellchecker.suffix_generator

targets = [
    "għamlulu",
    "għamluhulu",
    "ibgħatulu",
    "jibgħatulu",
    "jisraqhomlu",
    "israqhomlu",
]

print("=" * 80)
print("SUFFIX GENERATOR CANDIDATES FOR SURFACE CHECK")
print("=" * 80)

for t in targets:
    cands = sg.candidates_for_surface(t)
    in_dict = t in spellchecker.dictionary
    print(f"Target: {t!r:<15} in_dict={in_dict!s:<5} num_candidates={len(cands)}")
    for c in cands[:3]:
        print(f"  -> candidate: {c.surface!r} (stem={c.stem_word!r}, suffix={c.suffix_surface!r}, tag={c.raw_tag!r})")
