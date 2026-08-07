"""scratch/trace_hybrid_verb_restructuring.py
Trace why 'ha tabqizli' becomes 'ħa jaqbeż' and 'ha niggennen' becomes 'ħa jiġġennen' in Hybrid-First.
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
    neural_confidence_threshold=0.78,
)

inputs = ["ha tabqizli", "ha niggennen"]

for text in inputs:
    print("=" * 80)
    print(f"INPUT: {text!r}")
    print("=" * 80)

    # Stage 1: Spellchecker alone
    st1 = spellchecker.correct_text_rich(text)
    print("\nSTAGE 1 (Spellchecker alone):")
    print("  corrected_text:", repr(st1["corrected_text"]))

    # Stage 2: Neural model alone
    neur_res = neural_corrector.correct(text)
    print("\nNEURAL MODEL ALONE:")
    print("  corrected_text:", repr(neur_res["corrected_text"]))

    # Hybrid Pipeline Full Trace
    hyb_res = hybrid_pipeline.correct(text)
    print("\nHYBRID FIRST PIPELINE:")
    print("  Stage 1 text:", repr(hyb_res["debug"]["stage1_text"]))
    print("  Stage 2 text:", repr(hyb_res["debug"]["stage2_text"]))
    print("  Final text:  ", repr(hyb_res["corrected_text"]))
    print("  Neural edits applied in Stage 2:", hyb_res["debug"]["stage2_edits"])
