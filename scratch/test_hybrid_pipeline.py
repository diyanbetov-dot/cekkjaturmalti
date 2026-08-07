# -*- coding: utf-8 -*-
import sys, os, json
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, ".")

from neural_corrector.inference.corrector import NeuralCorrector
from Essentials.app import spellchecker

def hybrid_correct(text: str, corrector: NeuralCorrector) -> dict:
    # Pass 1: Algorithmic candidate correction & dictionary repair
    res_alg = spellchecker.correct_text_rich(text)
    stage1_text = res_alg["corrected_text"]

    # Pass 2: Neural context correction (capitalization, article hyphenation, context tagger)
    res_neural = corrector.correct(stage1_text)
    
    return {
        "input": text,
        "stage1_algorithmic": stage1_text,
        "final_hybrid": res_neural["corrected_text"],
        "neural_edits": res_neural["edits"],
        "processing_time_sec": res_neural["processing_time"],
    }

if __name__ == "__main__":
    v4_dir = "neural_corrector/artifacts/char_edit_bigru_v4"
    corrector_v4 = NeuralCorrector(v4_dir, use_dictionary_validation=True, use_suffix_validation=True)
    
    input_text = "ilbierah amilt xniekol ax kelli l guh."
    result = hybrid_correct(input_text, corrector_v4)
    print("\n" + "="*60)
    print("=== HYBRID PIPELINE EXECUTION RESULT ===")
    print("="*60)
    print("Input Text:          ", result["input"])
    print("Stage 1 (Algorithmic):", result["stage1_algorithmic"])
    print("Stage 2 (Final Hybrid):", result["final_hybrid"])
    print("="*60)
