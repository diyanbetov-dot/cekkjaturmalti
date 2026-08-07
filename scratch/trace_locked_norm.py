"""scratch/trace_locked_norm.py
Trace locked_norm and arbitration for 'ħa niggennen'.
INSPECT ONLY.
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from Essentials.app import spellchecker, grammar_rule_engine
from neural_corrector.inference.corrector import NeuralCorrector
from hybrid_corrector.pipeline import HybridFirstCorrector, _WORD_RE

DEFAULT_ARTIFACT = Path("neural_corrector/artifacts/char_edit_bigru_v4")
neural_corrector = NeuralCorrector(DEFAULT_ARTIFACT)

hybrid_pipeline = HybridFirstCorrector(
    spellchecker=spellchecker,
    neural_corrector=neural_corrector,
    grammar_rule_engine=grammar_rule_engine,
)

text = "ħa niggennen"
stage1_result = spellchecker.correct_text_rich(text)
stage1_text = stage1_result["corrected_text"]
stage1_tokens = stage1_result["tokens"]

print("text:", text)
print("stage1_text:", stage1_text)

locked_norm = set()
unrecognised_norm = set()

for tok in stage1_tokens:
    if isinstance(tok, dict):
        orig = tok.get("original", "")
        corr = tok.get("corrected", "")
        print("token:", tok)
        if orig and corr and orig.lower() != corr.lower():
            locked_norm.add(corr.lower())
            locked_norm.add(orig.lower())
        if tok.get("unrecognized") or tok.get("dubious"):
            unrecognised_norm.add((corr or orig).lower())

print("locked_norm:", locked_norm)
print("unrecognised_norm:", unrecognised_norm)

neural_result = neural_corrector.correct(stage1_text)
neural_text = neural_result["corrected_text"]
print("neural_text:", neural_text)

s1_matches = list(_WORD_RE.finditer(stage1_text))
nn_matches = list(_WORD_RE.finditer(neural_text))

for s1_m, nn_m in zip(s1_matches, nn_matches):
    s1_w = s1_m.group(0)
    nn_w = nn_m.group(0)
    print(f"s1_w={s1_w!r}, s1_norm={s1_w.lower()!r}, in_locked={s1_w.lower() in locked_norm}, nn_w={nn_w!r}")
