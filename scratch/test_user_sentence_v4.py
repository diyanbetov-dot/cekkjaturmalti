# -*- coding: utf-8 -*-
import sys, os, json
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, ".")

from neural_corrector.inference.corrector import NeuralCorrector
from Essentials.app import spellchecker

test_input = "ilbierah amilt xniekol ax kelli l guh."

print("=== INPUT ===")
print(test_input)
print()

# 1. Neural Corrector v4 (Raw)
v4_dir = "neural_corrector/artifacts/char_edit_bigru_v4"
if os.path.exists(v4_dir):
    corrector_v4 = NeuralCorrector(v4_dir)
    res_v4 = corrector_v4.correct(test_input)
    print("=== NEURAL CORRECTOR v4 Output ===")
    print("Corrected text:", res_v4["corrected_text"])
    print("Processing time:", f"{res_v4['processing_time']*1000:.2f} ms")
    print("Edits count:", len(res_v4["edits"]))
    for edit in res_v4["edits"]:
        print(f"  - [{edit['type']}] '{edit['original']}' -> '{edit['replacement']}' (conf: {edit['confidence']:.2f})")
    print()

# 2. Traditional / Algorithmic Core Spellchecker (Word by Word)
print("=== CORE SPELLCHECKER (Algorithmic Word-by-Word Analysis) ===")
words = [w.strip(".,!?") for w in test_input.split() if w.strip(".,!?")]
for word in words:
    is_valid = spellchecker.is_valid_word(word) if hasattr(spellchecker, "is_valid_word") else (word in spellchecker.dictionary_set)
    suggestions = spellchecker.suggest(word) if hasattr(spellchecker, "suggest") else []
    print(f"Word: '{word}' | In Dictionary: {is_valid} | Top Suggestions: {suggestions[:3]}")
