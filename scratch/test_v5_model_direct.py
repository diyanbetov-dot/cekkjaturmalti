"""scratch/test_v5_model_direct.py
Directly test NeuralCorrector BiGRU v5 on key agreement and suffixed test cases.
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

V5_ARTIFACT = Path("neural_corrector/artifacts/char_edit_bigru_v5")
print(f"Loading BiGRU v5 from {V5_ARTIFACT}...")
nc_v5 = NeuralCorrector(V5_ARTIFACT)

test_sentences = [
    "ha niggennen",
    "ħa niggennen",
    "ir-ragel ha niggennen",
    "ir-ragel ha thawdu",
    "il-mara ha tiggennen",
    "il-mara ha thawdha",
    "ha tabqizli",
    "seba jiem",
    "tlett gzejjer",
    "erba bozza",
    "fil-karozza",
]

print("=" * 80)
print("NEURAL MODEL BiGRU v5 DIRECT TEST RESULTS")
print("=" * 80)

for s in test_sentences:
    res = nc_v5.correct(s)
    print(f"Input: {s!r:<25} -> Corrected: {res['corrected_text']!r:<30} (changed: {res['changed']})")
