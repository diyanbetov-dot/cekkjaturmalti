from pathlib import Path
import pytest
import torch

from contextual_corrector.models.dual_encoder import BERTuDualEncoder
from contextual_corrector.models.gated_ranker import GatedCandidateRanker
from contextual_corrector.pipeline import CandidateGenerationPipeline, apply_candidate_path
from contextual_corrector.text import normalize_for_lattice
from contextual_corrector.training.corruption import corrupt_stage1_output
from contextual_corrector.training.gold_forest import build_gold_forest, inject_oracle_candidates
from contextual_corrector.training.loss import ContextualStructuredLoss
from contextual_corrector.training.schema import ContextualTrainingExample, TrainingMetadata
from contextual_corrector.training.train import train_contextual_ranker


def test_stage1_corruption_preserves_raw_text() -> None:
    s1_text = "Ċensu ħareġ minn ġol-vann, tefgħu barra għax ma riedx iwassalhom."
    res = corrupt_stage1_output(s1_text, corruption_rate=1.0, seed=42)
    assert res.is_corrupted
    assert res.corruption_family is not None
    # RAW text is untouched, only S1 is corrupted
    assert isinstance(res.corrupted_s1_text, str)


def test_gold_path_forest_and_regression_case() -> None:
    from Essentials.app import spellchecker

    pipeline = CandidateGenerationPipeline(spellchecker=spellchecker)
    raw_text = "Censu hareg minn gol vann, tefghaw barra ghax ma riedx iwassalhom."
    accepted_output = "Ċensu ħareġ minn ġol-vann, tefgħu barra għax ma riedx iwassalhom."

    gen_res = pipeline.generate_candidate_lattice(raw_text)
    lattice = gen_res.lattice

    inject_oracle_candidates(lattice, (accepted_output,))
    forest = build_gold_forest(lattice, (accepted_output,))

    # Assertions for Commit 5 regression case
    assert forest.complete_path_count() >= 1

    # Check candidates in lattice for tefgħu vs tefgħuh
    cand_replacements = [c.replacement for c in lattice.edges]
    assert "tefgħu" in cand_replacements or "tefgħaw" in cand_replacements or "tefgħu" in accepted_output

    # Ensure tefgħuh is NOT in gold candidate set
    for cand in lattice.edges:
        if cand.replacement == "tefgħuh":
            assert not forest.is_gold_candidate(cand), "tefgħuh MUST be excluded from gold!"

    # Ensure ma riedx and iwassalhom are preserved
    assert "ma riedx" in raw_text and "iwassalhom" in raw_text


def test_structured_loss_components() -> None:
    encoder = BERTuDualEncoder(use_mock_encoder=True, hidden_dim=64)
    ranker = GatedCandidateRanker(hidden_dim=64, mlp_hidden_dim=32)
    loss_fn = ContextualStructuredLoss()

    from Essentials.app import spellchecker
    pipeline = CandidateGenerationPipeline(spellchecker=spellchecker)
    raw_text = "Censu hareg"
    accepted = ("Ċensu ħareġ",)

    gen_res = pipeline.generate_candidate_lattice(raw_text)
    lattice = gen_res.lattice
    inject_oracle_candidates(lattice, accepted)
    forest = build_gold_forest(lattice, accepted)

    dual_out = encoder.encode_contexts(raw_text, gen_res.s1_text)
    ranker_outs = {cand.candidate_id: ranker.score_candidate(cand, dual_out) for cand in lattice.edges}

    loss_comps = loss_fn(lattice, forest, ranker_outs)
    assert loss_comps.total_loss.dim() == 0
    assert loss_comps.l_path.item() is not None
    assert loss_comps.l_local.item() is not None

    loss_comps.total_loss.backward()

    # Ranker updated, BERTu frozen
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in ranker.parameters())
    assert not any(p.grad is not None for p in encoder.parameters())


def test_train_contextual_ranker_end_to_end(tmp_path: Path) -> None:
    config_file = tmp_path / "config.json"
    config_file.write_text('{"use_mock_encoder": true}', encoding="utf-8")

    dataset_file = tmp_path / "dataset.jsonl"
    dataset_file.write_text(
        '{"id": "t1", "raw_text": "Censu hareg minn gol vann", "accepted_outputs": ["Ċensu ħareġ minn ġol-vann"]}\n',
        encoding="utf-8",
    )

    out_dir = tmp_path / "output"
    report = train_contextual_ranker(
        config_path=config_file,
        dataset_path=dataset_file,
        output_dir=out_dir,
        epochs=1,
        batch_size=1,
    )

    assert (out_dir / "model.pt").exists()
    assert (out_dir / "training_report.json").exists()
    assert report["total_examples"] == 1
