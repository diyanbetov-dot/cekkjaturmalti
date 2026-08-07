"""scratch/debug_neural_verb_restructuring.py
Check NeuralCorrector output for 'ħa niggennen' and 'ħa tabqizli'.
INSPECT ONLY.
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from neural_corrector.inference.corrector import NeuralCorrector

DEFAULT_ARTIFACT = Path("neural_corrector/artifacts/char_edit_bigru_v4")
nc = NeuralCorrector(DEFAULT_ARTIFACT)

test_phrases = [
    "ha niggennen",
    "ħa niggennen",
    "ha tabqizli",
    "ħa tabqizli",
    "ha taqbizli",
    "ħa taqbizli",
    "se niggennen",
    "se tabqizli",
]

print("=" * 80)
print("NEURAL MODEL INFERENCE ON SUFFIXED / PREFIXED VERBS")
print("=" * 80)

for p in test_phrases:
    res = nc.correct(p)
    print(f"Input: {p!r:<20} -> Corrected: {res['corrected_text']!r:<25} Decisions: {res.get('decisions')}")
