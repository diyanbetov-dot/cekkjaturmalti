"""scratch/debug_grammar_engine_rewrites.py
Test if grammar_rule_engine is rewriting 'Ħa niġġennen' -> 'Ħa jiġġennen'.
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

test_inputs = [
    "Ħa niġġennen",
    "ħa niggennen",
    "Ħa tabqiżli",
    "ħa tabqizli",
]

print("=" * 80)
print("INSPECTING GRAMMAR RULE ENGINE REWRITES")
print("=" * 80)

for text in test_inputs:
    res = spellchecker.correct_text_rich(text)
    c1 = res["corrected_text"]
    tokens = res["tokens"]
    words = [m.group(0) for m in spellchecker.WORD_PATTERN.finditer(c1)]
    findings = grammar_rule_engine.analyze(text=c1, request_words=words, tokens=tokens)
    c2, _, _ = grammar_rule_engine.apply_safe_rewrites(original_text=c1, corrected_text=c1, tokens=tokens)
    
    print(f"\nInput: {text!r}")
    print(f"  After Spellchecker (c1): {c1!r}")
    print(f"  Grammar Findings        : {findings}")
    print(f"  After Grammar Rewrites  : {c2!r}")
