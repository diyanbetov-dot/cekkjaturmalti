"""scratch/debug_jghagel.py
Investigate why 'jghagel' is not corrected to 'jgħaġġel' by Main/Hybrid spellchecker.
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

print(f"--- Debugging '{word}' ---")
print("correct_word:", repr(spellchecker.correct_word(word)))
print("in dictionary_set:", word in spellchecker.dictionary_set)
print("in dictionary:", word in spellchecker.dictionary)

norm = spellchecker._normalize_word(word)
print("normalized:", repr(norm))

print("\n--- Correction Trace ---")
try:
    trace = spellchecker.correction_trace(word)
    print("Phase:", trace.get("phase"))
    print("Phase corrected:", trace.get("phase_corrected"))
    print("Final correction:", trace.get("final_correction"))
    print("Suggestions:", trace.get("suggestions"))
    print("Basic candidates:", trace.get("basic_candidates"))
    print("Complex candidates:", trace.get("complex_candidates"))
except Exception as e:
    print("Trace error:", e)

print("\n--- Checking target word 'jgħaġġel' ---")
target = "jgħaġġel"
target_norm = spellchecker._normalize_word(target)
print("target normalized:", repr(target_norm))
print("target in dictionary_set:", target_norm in spellchecker.dictionary_set)
print("target in dictionary:", target_norm in spellchecker.dictionary)
print("target verb records:", bool(spellchecker._verb_records_for_surface(target)))

print("\n--- Distance check ---")
print("Damerau-Levenshtein between norm and target_norm:", spellchecker._damerau_levenshtein_distance(norm, target_norm))
