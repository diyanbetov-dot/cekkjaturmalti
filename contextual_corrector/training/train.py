from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import random
import time
import torch
from torch.optim import AdamW

from ..models.dual_encoder import BERTuDualEncoder
from ..models.gated_ranker import GatedCandidateRanker
from ..pipeline import CandidateGenerationPipeline
from .corruption import corrupt_stage1_output
from .gold_forest import build_gold_forest, inject_oracle_candidates
from .loss import ContextualStructuredLoss
from .schema import ContextualTrainingExample, TrainingMetadata


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_contextual_ranker(
    config_path: Path | str,
    dataset_path: Path | str,
    output_dir: Path | str,
    *,
    spellchecker=None,
    epochs: int = 2,
    batch_size: int = 4,
    learning_rate: float = 1e-3,
    seed: int = 42,
) -> dict:
    set_seed(seed)
    config_path = Path(config_path)
    dataset_path = Path(dataset_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = {}
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))

    if spellchecker is None:
        from Essentials.app import spellchecker as production_spellchecker
        spellchecker = production_spellchecker

    # Initialize frozen BERTu dual encoder and gated candidate ranker
    use_mock = config.get("use_mock_encoder", True)
    dual_encoder = BERTuDualEncoder(use_mock_encoder=use_mock)
    assert dual_encoder.verify_frozen(), "BERTu dual encoder MUST be frozen!"

    ranker = GatedCandidateRanker(hidden_dim=dual_encoder.hidden_dim)
    optimizer = AdamW(ranker.parameters(), lr=learning_rate, weight_decay=0.01)
    loss_fn = ContextualStructuredLoss()

    # Load dataset lines
    raw_data = []
    if dataset_path.exists():
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    raw_data.append(json.loads(line))
    else:
        # Default sample dataset if path not found
        raw_data = [
            {
                "id": "ex_1",
                "raw_text": "Censu hareg minn gol vann, tefghaw barra ghax ma riedx iwassalhom.",
                "accepted_outputs": ["Ċensu ħareġ minn ġol-vann, tefgħu barra għax ma riedx iwassalhom."],
            }
        ]

    pipeline = CandidateGenerationPipeline(spellchecker=spellchecker)
    examples: list[ContextualTrainingExample] = []

    oracle_diagnostics = {
        "total_examples": 0,
        "oracle_injections": 0,
    }

    for idx, row in enumerate(raw_data):
        raw_text = row["raw_text"]
        accepted = tuple(row["accepted_outputs"])

        # Candidate generation
        gen_res = pipeline.generate_candidate_lattice(raw_text)
        lattice = gen_res.lattice
        s1_text = gen_res.s1_text

        # Corruption simulation (training mode)
        corr_res = corrupt_stage1_output(s1_text, corruption_rate=0.30, seed=seed + idx)

        # Oracle candidate injection
        inj_stats = inject_oracle_candidates(lattice, accepted)
        if inj_stats["oracle_injected"]:
            oracle_diagnostics["oracle_injections"] += 1
        oracle_diagnostics["total_examples"] += 1

        forest = build_gold_forest(lattice, accepted)
        meta = TrainingMetadata(
            parent_id=row.get("id", f"ex_{idx}"),
            corruption_family=corr_res.corruption_family,
            clean_identity=(raw_text == accepted[0]),
        )

        ex = ContextualTrainingExample(
            example_id=row.get("id", f"ex_{idx}"),
            raw_text=raw_text,
            accepted_outputs=accepted,
            s1_text=corr_res.corrupted_s1_text,
            s1_out_of_fold=True,
            lattice=lattice,
            gold_forest=forest,
            metadata=meta,
        )
        examples.append(ex)

    # Training loop
    history = []
    start_time = time.time()

    for epoch in range(epochs):
        ranker.train()
        total_epoch_loss = 0.0
        optimizer.zero_grad()

        for step, ex in enumerate(examples):
            dual_out = dual_encoder.encode_contexts(ex.raw_text, ex.s1_text)

            ranker_outputs = {}
            for cand in ex.lattice.edges:
                ranker_outputs[cand.candidate_id] = ranker.score_candidate(cand, dual_out)

            loss_comps = loss_fn(ex.lattice, ex.gold_forest, ranker_outputs)
            loss = loss_comps.total_loss / batch_size
            loss.backward()

            total_epoch_loss += loss.item() * batch_size

            if (step + 1) % batch_size == 0 or (step + 1) == len(examples):
                optimizer.step()
                optimizer.zero_grad()

        avg_loss = total_epoch_loss / max(1, len(examples))
        history.append({"epoch": epoch + 1, "loss": avg_loss})

    # Save checkpoint
    model_path = output_dir / "model.pt"
    torch.save(
        {
            "ranker_state_dict": ranker.state_dict(),
            "config": config,
            "hidden_dim": dual_encoder.hidden_dim,
            "epoch": epochs,
        },
        model_path,
    )

    report = {
        "model_version": "contextual-ranker-v1.0",
        "training_seconds": round(time.time() - start_time, 2),
        "total_examples": len(examples),
        "history": history,
        "oracle_diagnostics": oracle_diagnostics,
        "model_bytes": model_path.stat().st_size if model_path.exists() else 0,
    }

    report_path = output_dir / "training_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return report
