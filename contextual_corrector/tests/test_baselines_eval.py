from __future__ import annotations

import pytest
from pathlib import Path
from contextual_corrector.evaluation.splits import GroupedSplitter
from contextual_corrector.evaluation.metrics import compute_metrics
from contextual_corrector.evaluation.taxonomy import (
    categorize_failure,
    summarize_failures,
    FailureCategory,
)
from contextual_corrector.evaluation.baselines import Stage1Baseline, IdentityBaseline, MockNeuralBaseline
from contextual_corrector.evaluation.ablation import run_ablation_experiment


def test_grouped_splitter_determinism_and_no_leakage() -> None:
    splitter = GroupedSplitter(val_ratio=0.2, test_ratio=0.2, seed=42)
    examples = [
        {"id": f"ex_{i}", "raw_text": f"raw text {i}", "accepted": f"accepted text {i}"}
        for i in range(100)
    ]
    splits1 = splitter.split_examples(examples)
    splits2 = splitter.split_examples(examples)

    # Determinism assertion
    assert len(splits1["train"].examples) == len(splits2["train"].examples)
    assert len(splits1["validation"].examples) == len(splits2["validation"].examples)
    assert len(splits1["test"].examples) == len(splits2["test"].examples)

    # Disjoint sets assertion (no leakage)
    train_ids = {ex["id"] for ex in splits1["train"].examples}
    val_ids = {ex["id"] for ex in splits1["validation"].examples}
    test_ids = {ex["id"] for ex in splits1["test"].examples}

    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)
    assert len(train_ids) + len(val_ids) + len(test_ids) == 100


def test_compute_metrics_accuracy_and_f1() -> None:
    preds = ["Ċensu ħareġ", "it-tifel mar"]
    tgts = ["Ċensu ħareġ", "it-tifel ġie"]
    raws = ["Censu hareg", "it-tifel mar"]

    metrics = compute_metrics(preds, tgts, raws)

    assert metrics.total_examples == 2
    assert metrics.total_exact_matches == 1
    assert metrics.exact_match_accuracy == 0.5
    assert 0.0 <= metrics.token_f1 <= 1.0


def test_failure_taxonomy_categorization() -> None:
    rec1 = categorize_failure("1", "Censu hareg", "Censu hareg", "Ċensu ħareġ", gold_in_lattice=False)
    assert rec1.category == FailureCategory.GENERATION_FAILURE

    rec2 = categorize_failure("2", "Censu hareg", "Censu hareg", "Ċensu ħareġ", gold_in_lattice=True, gold_in_search=False)
    assert rec2.category == FailureCategory.PRUNING_FAILURE

    rec3 = categorize_failure("3", "Censu hareg", "Censu hareg", "Ċensu ħareġ", s1_corrupted_raw=True)
    assert rec3.category == FailureCategory.S1_CORRUPTION_LEAK

    rec4 = categorize_failure("4", "Censu hareg", "Censu hareg", "Ċensu ħareġ")
    assert rec4.category == FailureCategory.RANKING_FAILURE

    summary = summarize_failures([rec1, rec2, rec3, rec4])
    assert summary.total_failures == 4
    assert summary.category_counts[FailureCategory.GENERATION_FAILURE.value] == 1
    assert summary.category_counts[FailureCategory.RANKING_FAILURE.value] == 1


def test_baselines_and_ablation() -> None:
    ident = IdentityBaseline()
    assert ident.correct("Censu hareg") == "Censu hareg"

    mock = MockNeuralBaseline({"Censu hareg": "Ċensu ħareġ"})
    assert mock.correct("Censu hareg") == "Ċensu ħareġ"

    summaries = run_ablation_experiment(["full", "no_s1"])
    assert "full" in summaries
    assert "no_s1" in summaries
    assert summaries["full"].mean_accuracy >= summaries["no_s1"].mean_accuracy
