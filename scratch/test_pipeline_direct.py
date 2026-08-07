"""scratch/test_pipeline_direct.py
Direct pipeline.correct() test.
INSPECT ONLY.
"""
from __future__ import annotations
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from Essentials.app import spellchecker, grammar_rule_engine
from neural_corrector.inference.corrector import NeuralCorrector
from hybrid_corrector.pipeline import HybridFirstCorrector

DEFAULT_ARTIFACT = Path("neural_corrector/artifacts/char_edit_bigru_v4")
neural_corrector = NeuralCorrector(DEFAULT_ARTIFACT)

pipeline = HybridFirstCorrector(
    spellchecker=spellchecker,
    neural_corrector=neural_corrector,
    grammar_rule_engine=grammar_rule_engine,
)

for text in ["ha niggennen", "ħa niggennen", "ha tabqizli", "ħa tabqizli"]:
    res = pipeline.correct(text)
    print(f"Input: {text!r:<20} -> Corrected: {res['corrected_text']!r}")
