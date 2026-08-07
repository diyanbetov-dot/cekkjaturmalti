"""scratch/test_tabqizli_candidates.py
Inspect candidate generation and suggestions for 'tabqizli'.
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

word = "tabqizli"

print("=" * 80)
print(f"INSPECTING CANDIDATE GENERATION FOR '{word}'")
print("=" * 80)

print(f"spellchecker.correct_word('{word}'):", repr(spellchecker.correct_word(word)))
print(f"spellchecker.suggest('{word}'):", repr(spellchecker.suggest(word)))

trace = spellchecker.correction_trace(word)
print("\nCorrection Trace:")
print("  Phase:", trace.get("phase"))
print("  Phase corrected:", trace.get("phase_corrected"))
print("  Suggestions:", trace.get("suggestions"))
print("  Basic candidates:", trace.get("basic_candidates"))

for i, item in enumerate(trace.get("evidence", {}).get("ranked", []), 1):
    print(f"  [{i}] word={item.get('word')!r:<15} score={item.get('score')} edit_dist={item.get('edit_distance')}")
