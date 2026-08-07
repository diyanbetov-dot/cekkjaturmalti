"""scratch/inspect_algorithmic_rid.py
Algorithmic inspection for 'rid' -> did 'rrid' show up in candidates / suggestions?
INSPECT ONLY.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from Essentials.app import spellchecker

word = "rid"

print("=" * 80)
print(f"ALGORITHMIC INSPECTION FOR '{word}':")
print("=" * 80)

# 1. Check if 'rid' is in dictionary
print("\n1. DICTIONARY CHECK:")
print(f"  'rid' in dictionary_set: {'rid' in spellchecker.dictionary_set}")
print(f"  'rrid' in dictionary_set: {'rrid' in spellchecker.dictionary_set}")

# 2. Phase X candidate collection
px = spellchecker._phase_x_collect_candidates(word)
print("\n2. PHASE X CANDIDATES FOR 'rid':")
print(f"  phase: {px.phase}")
print(f"  corrected: {px.corrected}")
print(f"  basic_candidates: {list(px.basic_candidates)}")
print(f"  complex_candidates: {list(px.complex_candidates)}")

# 3. Suggestions for 'rid'
suggs = spellchecker.suggest(word, limit=12)
print("\n3. SUGGESTIONS FOR 'rid':")
print(f"  {suggs}")
print(f"  Is 'rrid' in suggestions? {'rrid' in suggs} (Index: {suggs.index('rrid') if 'rrid' in suggs else 'N/A'})")

# 4. Candidate evidence debug for 'rid'
print("\n4. CANDIDATE EVIDENCE DEBUG FOR 'rid':")
try:
    evidence = spellchecker.candidate_evidence_debug(word, limit=12)
    for i, item in enumerate(evidence.get("ranked", []), 1):
        print(f"  [{i}] word={item.get('word')!r:<12} score={item.get('score')} edit_dist={item.get('edit_distance')} rank_key={item.get('rank_key')}")
except Exception as e:
    print(f"  Evidence debug error: {e}")

# 5. Distance & Anchor Check
print("\n5. ANCHOR & DISTANCE CHECK:")
print(f"  Damerau-Levenshtein distance('rid', 'rrid'): {spellchecker._damerau_levenshtein_distance('rid', 'rrid')}")
print(f"  Anchor of 'rid': {spellchecker._extract_consonant_anchor('rid')}")
print(f"  Anchor of 'rrid': {spellchecker._extract_consonant_anchor('rrid')}")
