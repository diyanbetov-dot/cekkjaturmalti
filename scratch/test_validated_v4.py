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

# Neural Corrector v4 + Dictionary & Suffix Validation
v4_dir = "neural_corrector/artifacts/char_edit_bigru_v4"
corrector_v4_validated = NeuralCorrector(
    v4_dir,
    use_dictionary_validation=True,
    use_suffix_validation=True,
)
res = corrector_v4_validated.correct(test_input)
print("=== NEURAL CORRECTOR v4 (with Dictionary & Suffix Rescue) ===")
print("Corrected text:", res["corrected_text"])
print("Processing time:", f"{res['processing_time']*1000:.2f} ms")
print("Edits count:", len(res["edits"]))
for edit in res["edits"]:
    print(f"  - [{edit['type']}] '{edit['original']}' -> '{edit['replacement']}' (conf: {edit['confidence']:.2f})")
