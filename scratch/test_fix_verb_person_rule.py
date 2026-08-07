"""scratch/test_fix_verb_person_rule.py
Test fixing VERB_VERB_PERSON_NUMBER to exclude particle/pre-verb 'ħa', 'ha', 'se', 'ser'.
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
    "ħa niggennen",
    "Ħa niġġennen",
    "ha tabqizli",
    "Ħa tabqiżli",
    "se niggennen",
    "ir-ragel ha niggennen",
]

print("=" * 80)
print("TESTING BEFORE/AFTER GRAMMAR RULE FIX FOR PARTICLE 'ĦA'")
print("=" * 80)

# 1. Test before fix
print("\n--- BEFORE FIX ---")
for text in test_inputs:
    res = spellchecker.correct_text_rich(text)
    c1 = res["corrected_text"]
    words = [m.group(0) for m in spellchecker.WORD_PATTERN.finditer(c1)]
    findings = grammar_rule_engine.analyze(text=c1, request_words=words, tokens=res["tokens"])
    c2, _, _ = grammar_rule_engine.apply_safe_rewrites(original_text=c1, corrected_text=c1, tokens=res["tokens"])
    print(f"Input: {text!r:<25} -> After Grammar: {c2!r:<25} Findings: {[f['rule_id'] for f in findings]}")
