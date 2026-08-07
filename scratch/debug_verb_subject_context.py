"""scratch/debug_verb_subject_context.py
Test if a subject before 'ħa niggennen' / 'ħa tabqizli' triggers grammar rule rewriting.
INSPECT ONLY.
"""
from __future__ import annotations
import json
import sys
import urllib.request
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

hybrid_pipeline = HybridFirstCorrector(
    spellchecker=spellchecker,
    neural_corrector=neural_corrector,
    grammar_rule_engine=grammar_rule_engine,
)

test_sentences = [
    "ir-ragel ha niggennen",
    "ir-raġel ħa niggennen",
    "huwa ha tabqizli",
    "meta rajtu ha niggennen",
    "kważi ha niggennen",
    "kont ha niggennen",
    "ħa niggennen",
    "ħa tabqizli",
]

print("=" * 80)
print("INSPECTING SUBJECT CONTEXT & GRAMMAR RULES ON PRE-VERBS")
print("=" * 80)

for s in test_sentences:
    # 1. Main engine
    m_res = spellchecker.correct_text_rich(s)
    # 2. Hybrid engine
    h_res = hybrid_pipeline.correct(s)
    print(f"\nInput: {s!r}")
    print(f"  Main Engine (5000)  : {m_res['corrected_text']!r}")
    print(f"  Hybrid Engine (5002): {h_res['corrected_text']!r}")
