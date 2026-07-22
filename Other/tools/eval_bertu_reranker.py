# -*- coding: utf-8 -*-
"""
BERTu Re-ranker Quality Evaluation
===================================
Tests whether BERTu contextual re-ranking improves candidate selection
over the existing evidence-based scoring alone.

For each test case we:
  1. Get the existing pipeline's candidate list (ordered by current scoring)
  2. Re-rank using BERTu
  3. Compare: did BERTu put the correct answer first?

Run:  .\.venv\Scripts\python.exe Other\tools\eval_bertu_reranker.py
"""

import sys, os
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from Essentials.app import spellchecker

# ---------------------------------------------------------------------------
# Test cases: (sentence, token_index, input_word, expected_correction)
# ---------------------------------------------------------------------------
TEST_CASES = [
    # Morphological suffix restoration
    ("Werrinha ir-rapport li qal kien ippreżentat",  0, "Werrinha",      "Werriena"),
    ("Werrinha r-rapport",                            0, "Werrinha",      "Werriena"),

    # Preposition-article: place name must NOT contract
    ("minn barra minn Malta speċjalment",            3, "Malta",         "Malta"),

    # Pronoun suffix: bħali must NOT become bħall-i
    ("tagħmlux bħali kif kont se nagħmel",           1, "bħali",         "bħali"),

    # Common typos
    ("esperjenzajt meta mort",                        0, "esperjenzajt",  "esperjenzajt"),
    ("Irrizulta li dak",                              0, "Irrizulta",     "Irriżulta"),
    ("koppja ta dan ir-rapport",                      0, "koppja",        "kopja"),
    ("jixtri karozza 2nd hand",                       0, "jixtri",        "jixtri"),
    ("Mhux ser insemmi min kien",                     0, "Mhux",          "Mhux"),
    ("ghamilna test drive tal-karozza",               0, "ghamilna",      "għamilna"),
    ("jafrhom fl-istorja",                             0, "jafrhom",       "jarafhom"),
    ("kellimhom dwar il-kwistjoni",                    0, "kellimhom",     "kellimhom"),
]

# ---------------------------------------------------------------------------

def get_candidates(word: str, limit: int = 10) -> list[str]:
    """Get the existing pipeline's candidate suggestions."""
    try:
        return spellchecker.suggest(word, limit=limit)
    except Exception:
        return []


def run_evaluation():
    reranker = getattr(spellchecker, "bertu_reranker", None)
    bertu_available = reranker is not None and reranker.is_available()

    print(f"BERTu available: {bertu_available}")
    print(f"{'─'*80}")
    print(f"{'Input':<20} {'Expected':<16} {'Existing #1':<16} {'BERTu #1':<16} {'Δ'}")
    print(f"{'─'*80}")

    baseline_correct = 0
    bertu_correct = 0
    total = 0

    for sentence, token_idx, input_word, expected in TEST_CASES:
        total += 1
        candidates = get_candidates(input_word)
        existing_top = candidates[0] if candidates else "(none)"

        # BERTu re-rank
        if bertu_available and candidates:
            tokens = sentence.split()
            reranked = reranker.rerank(
                sentence=sentence,
                token_index=token_idx,
                candidates=candidates,
                tokens=tokens,
                fallback_order=candidates,
            )
            bertu_top = reranked[0] if reranked else existing_top
        else:
            bertu_top = existing_top

        # Normalise for comparison
        norm = spellchecker._normalize_word
        existing_ok = norm(existing_top) == norm(expected)
        bertu_ok = norm(bertu_top) == norm(expected)

        if existing_ok:
            baseline_correct += 1
        if bertu_ok:
            bertu_correct += 1

        delta = ""
        if bertu_ok and not existing_ok:
            delta = "✅ BERTu fixed"
        elif existing_ok and not bertu_ok:
            delta = "❌ BERTu broke"
        elif not existing_ok and not bertu_ok:
            delta = "✗ both wrong"

        print(
            f"{input_word:<20} {expected:<16} {existing_top:<16} {bertu_top:<16} {delta}"
        )

    print(f"{'─'*80}")
    print(f"Baseline correct: {baseline_correct}/{total}  ({100*baseline_correct//total}%)")
    if bertu_available:
        print(f"BERTu correct:    {bertu_correct}/{total}  ({100*bertu_correct//total}%)")
        delta = bertu_correct - baseline_correct
        print(f"Net Δ:            {'+' if delta >= 0 else ''}{delta}")
    else:
        print("BERTu not available — only baseline shown.")


if __name__ == "__main__":
    run_evaluation()
