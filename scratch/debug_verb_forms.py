"""scratch/debug_verb_forms.py
Debug verb forms for 'tabqizli' and 'niggennen'.
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from Essentials.app import spellchecker

for word in ["tabqizli", "taqbizli", "niggennen", "nigennen", "niggenen"]:
    print(f"\n--- Word: '{word}' ---")
    print("  correct_word:", repr(spellchecker.correct_word(word)))
    print("  suggest:", spellchecker.suggest(word, limit=5))
    print("  in dictionary_set:", word in spellchecker.dictionary_set)
    if hasattr(spellchecker, "suffix_generator"):
        sg = spellchecker.suffix_generator
        print("  suffix_generator.candidates_for_surface:", sg.candidates_for_surface(word))
