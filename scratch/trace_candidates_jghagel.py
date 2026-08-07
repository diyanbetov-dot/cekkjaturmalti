"""scratch/trace_candidates_jghagel.py
Inspect candidate generation phases for 'jghagel'.
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
print(f"--- Candidates for '{word}' ---")

# Check orthographic generator candidates
if hasattr(spellchecker, "orthographic_generator"):
    og_cands = spellchecker.orthographic_generator.generate_candidates(word)
    print("orthographic_generator candidates:", og_cands)

# Check composite generator candidates
if hasattr(spellchecker, "composite_generator"):
    comp_cands = spellchecker.composite_generator.generate_candidates(word)
    print("composite_generator candidates:", comp_cands)

# Check doubled letter generator candidates
if hasattr(spellchecker, "doubled_letter_generator"):
    dlg = spellchecker.doubled_letter_generator
    print("doubled_letter_generator:", dlg.correct_missing_double(word))

# Check Phase X candidate collection breakdown
px = spellchecker._phase_x_collect_candidates(word)
print("\nPhase X object:", px)
print("  phase:", px.phase)
print("  corrected:", px.corrected)
print("  basic_candidates:", list(px.basic_candidates))
print("  complex_candidates:", list(px.complex_candidates))

# Check suggestions breakdown
suggs = spellchecker.suggest(word, limit=5)
print("\nSuggestions:", suggs)
