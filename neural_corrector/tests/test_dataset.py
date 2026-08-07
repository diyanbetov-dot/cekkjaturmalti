from __future__ import annotations

import json
from pathlib import Path

from neural_corrector.dataset.analyze_pairs import read_jsonl
from neural_corrector.models.alignment import apply_actions, derive_actions

ROOT = Path(__file__).resolve().parents[2]
NEURAL_ROOT = ROOT / "neural_corrector"


def test_parser_keeps_every_documented_block() -> None:
    report = json.loads(
        (NEURAL_ROOT / "data/reports/parse_report.json").read_text(encoding="utf-8")
    )
    rows = read_jsonl(NEURAL_ROOT / "data/processed/all_pairs.jsonl")
    assert report["total_blocks"] == 212
    assert report["parsed_examples"] == 212
    assert report["malformed_blocks"] == 0
    assert len(rows) == 212
    assert all(row["raw_block"] for row in rows)
    assert all(row["raw_input"] == row["noisy"] for row in rows)
    assert all(row["raw_output"] == row["clean"] for row in rows)


def test_every_pair_has_an_exact_edit_encoding() -> None:
    rows = read_jsonl(NEURAL_ROOT / "data/processed/all_pairs.jsonl")
    for row in rows:
        actions = derive_actions(row["noisy"], row["clean"])
        assert len(actions) == len(row["noisy"])
        assert apply_actions(row["noisy"], actions) == row["clean"], row["id"]


def test_locked_splits_are_disjoint_and_complete() -> None:
    payload = json.loads(
        (NEURAL_ROOT / "data/splits/LOCKED_SPLITS.json").read_text(encoding="utf-8")
    )
    split_sets = {
        name: set(example_ids) for name, example_ids in payload["splits"].items()
    }
    names = list(split_sets)
    for index, left_name in enumerate(names):
        for right_name in names[index + 1 :]:
            assert split_sets[left_name].isdisjoint(split_sets[right_name])
    expected_ids = {
        row["id"]
        for row in read_jsonl(NEURAL_ROOT / "data/processed/all_pairs.jsonl")
    }
    assert set().union(*split_sets.values()) == expected_ids


def test_synthetic_rows_come_only_from_training_sources() -> None:
    split_payload = json.loads(
        (NEURAL_ROOT / "data/splits/LOCKED_SPLITS.json").read_text(encoding="utf-8")
    )
    training_ids = set(split_payload["splits"]["train"])
    rows = read_jsonl(NEURAL_ROOT / "data/processed/synthetic_train.jsonl")
    assert rows
    assert all(row["clean_source_id"] in training_ids for row in rows)
    assert all("*" not in row["clean"] for row in rows)
    assert all(row["source"] == "synthetic" for row in rows)
