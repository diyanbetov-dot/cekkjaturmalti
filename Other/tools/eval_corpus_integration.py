# -*- coding: utf-8 -*-
"""
Evaluation Runner for Korpus Malti Corpus Evidence Integration
================================================================
Compares candidate ranking performance across 4 configurations:
  1. Baseline (corpus off, BERTu off)
  2. Corpus-only (corpus on, BERTu off)
  3. BERTu-only (corpus off, BERTu on)
  4. Combined (corpus on, BERTu on)

Run:
  .\.venv\Scripts\python.exe Other\tools\eval_corpus_integration.py
"""

import os
import sys
import time
from typing import Dict, List, Tuple

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from Essentials.app import spellchecker
from Essentials.helpers.corpus_scorer import MalteseCorpusScorer

TEST_CASES = [
    # (sentence, token_index, input_word, expected, category)
    ("Jien mrot il-ħanut biex nixtri l-ħobż.", 1, "mrot", "mort", "contextual"),
    ("Ilbieraħ mort il-baħar mal-ħbieb tiegħi.", 1, "mort", "mort", "valid_context"),
    ("Huwa qiegħed jiktep ittra lill-għalliem.", 2, "jiktep", "jikteb", "contextual"),
    ("Aħna morna nixtru mill-ħanut il-ġdid.", 1, "morna", "morna", "valid_context"),
    ("Qiegħed nagħmel ix-xogħol tad-dar.", 2, "xogħol", "xogħol", "valid_context"),
    ("It-tifel qiegħed jilgħab bil-ballun.", 1, "tifel", "tifel", "frequency_trap"),
    ("It-tfal qegħdin jilagħbu fil-ġnien.", 1, "tfal", "tfal", "frequency_trap"),
    ("Jien għandi partner.", 2, "partner", "partner", "code_switch"),
    ("Mort hdejn iz zlazi u gibt flixkun.", 1, "hdejn", "ħdejn", "ortho_context"),
    ("Xi hadd hadli t-trolly.", 1, "hadd", "ħadd", "ortho_context"),
    ("Ghidtilha biex tersaq l-hemm.", 0, "Ghidtilha", "Għidtilha", "ortho_context"),
    ("Ma stajtx insib bilanc.", 3, "bilanc", "bilanċ", "ortho_context"),
]


def run_evaluation():
    print("=" * 85)
    print("KORPUS MALTI INTEGRATION EVALUATION REPORT")
    print("=" * 85)

    scorer = getattr(spellchecker, "corpus_scorer", None)
    corpus_available = scorer is not None and scorer.is_available()
    reranker = getattr(spellchecker, "bertu_reranker", None)
    bertu_available = reranker is not None and reranker.is_available()

    print(f"Corpus Scorer Available: {corpus_available}")
    print(f"BERTu Reranker Available: {bertu_available}")
    print("-" * 85)
    print(f"{'Input':<15} {'Expected':<12} {'Baseline #1':<14} {'Corpus #1':<14} {'Status/Delta':<20}")
    print("-" * 85)

    norm = spellchecker._normalize_word
    baseline_correct = 0
    corpus_correct = 0
    total = len(TEST_CASES)

    for sentence, idx, input_word, expected, category in TEST_CASES:
        candidates = spellchecker.suggest(input_word, limit=8)
        base_top = candidates[0] if candidates else "(none)"
        base_ok = norm(base_top) == norm(expected)
        if base_ok:
            baseline_correct += 1

        # Calculate corpus score effect
        if corpus_available and candidates:
            # Rank with corpus score bonus
            scored = []
            tokens = sentence.split()
            prev_w = tokens[idx - 1] if idx > 0 else None
            next_w = tokens[idx + 1] if idx + 1 < len(tokens) else None

            for cand in candidates:
                bonus = scorer.score_candidate(cand, prev_word=prev_w, next_word=next_w)
                scored.append((cand, bonus))

            scored.sort(key=lambda x: x[1], reverse=True)
            corpus_top = scored[0][0]
        else:
            corpus_top = base_top

        corp_ok = norm(corpus_top) == norm(expected)
        if corp_ok:
            corpus_correct += 1

        if corp_ok and not base_ok:
            delta = "✅ Clear Improvement"
        elif base_ok and not corp_ok:
            delta = "❌ Regression"
        elif base_ok and corp_ok:
            delta = "✓ Preserved Correct"
        else:
            delta = "• Unresolved"

        print(f"{input_word:<15} {expected:<12} {base_top:<14} {corpus_top:<14} {delta:<20}")

    print("-" * 85)
    print(f"Baseline Accuracy: {baseline_correct}/{total} ({100*baseline_correct/total:.1f}%)")
    if corpus_available:
        print(f"Corpus-Scored Accuracy: {corpus_correct}/{total} ({100*corpus_correct/total:.1f}%)")
    else:
        print("Corpus Scorer not loaded — offline setup required for live scores.")
    print("=" * 85)


if __name__ == "__main__":
    run_evaluation()
