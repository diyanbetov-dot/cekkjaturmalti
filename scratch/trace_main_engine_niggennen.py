"""scratch/trace_main_engine_niggennen.py
Trace why Main Engine converts 'niggennen' to 'jiġġennen'.
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

word = "niggennen"

print("=" * 80)
print(f"TRACING MAIN ENGINE CORRECTION FOR '{word}'")
print("=" * 80)

print("correct_word('niggennen'):", repr(spellchecker.correct_word(word)))
print("correct_text_rich('ha niggennen'):", repr(spellchecker.correct_text_rich("ha niggennen")["corrected_text"]))

trace = spellchecker.correction_trace(word)
print("\nCorrection Trace for 'niggennen':")
print("  Phase:", trace.get("phase"))
print("  Phase corrected:", trace.get("phase_corrected"))
print("  Final correction:", trace.get("final_correction"))
print("  Basic candidates:", trace.get("basic_candidates"))
print("  Suggestions:", trace.get("suggestions"))
print("  Recognition sources:", trace.get("recognition_sources"))

print("\nCandidate evidence debug:")
for i, item in enumerate(trace.get("evidence", {}).get("ranked", []), 1):
    print(f"  [{i}] word={item.get('word')!r:<15} score={item.get('score')} edit_dist={item.get('edit_distance')}")
