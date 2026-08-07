"""scratch/debug_sentence_jghagel.py
Test why 'jghagel' is not corrected in sentence context.
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from Essentials.app import spellchecker

text = "hu kien qed jghagel hafna biex jilhaq il-vapur tal-Ghawdex."
print("Input:", text)

res = spellchecker.correct_text_rich(text)
print("Corrected text:", res["corrected_text"])

print("\nTokens:")
for tok in res.get("tokens", []):
    if isinstance(tok, dict) and tok.get("type") == "word":
        print("Word token:", tok.get("original"), "->", tok.get("corrected"), "choices:", tok.get("choices"))
