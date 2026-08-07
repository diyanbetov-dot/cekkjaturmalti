from __future__ import annotations

import json
from pathlib import Path

from neural_corrector.dataset.analyze_pairs import read_jsonl
from neural_corrector.dataset.merge_examples import merge
from neural_corrector.dataset.split_groups import build_groups, signature
from neural_corrector.models.alignment import apply_actions, derive_actions

ROOT = Path(__file__).resolve().parents[2]
NEURAL_ROOT = ROOT / "neural_corrector"


def test_parser_keeps_every_documented_block() -> None:
    report = json.loads(
        (NEURAL_ROOT / "data/reports/parse_report.json").read_text(encoding="utf-8")
    )
    merge_report = json.loads(
        (NEURAL_ROOT / "data/reports/merge_report.json").read_text(encoding="utf-8")
    )
    rows = read_jsonl(NEURAL_ROOT / "data/processed/all_pairs.jsonl")
    assert report["total_blocks"] == merge_report["final_examples"] == 1068
    assert report["parsed_examples"] == merge_report["final_examples"]
    assert report["malformed_blocks"] == 0
    assert report["exact_duplicate_pairs"] == 0
    assert not report["conflicting_duplicate_inputs"]
    assert report["unchanged_examples"] == 215
    assert len(rows) == merge_report["final_examples"]
    assert all(row["raw_block"] for row in rows)
    assert all(row["raw_input"] == row["noisy"] for row in rows)
    assert all(row["raw_output"] == row["clean"] for row in rows)


def test_every_pair_has_an_exact_edit_encoding() -> None:
    rows = read_jsonl(NEURAL_ROOT / "data/processed/all_pairs.jsonl")
    for row in rows:
        actions = derive_actions(row["noisy"], row["clean"])
        assert len(actions) == len(row["noisy"])
        assert apply_actions(row["noisy"], actions) == row["clean"], row["id"]


def test_incoming_merge_is_recoverable_and_idempotent() -> None:
    current_path = ROOT / "AI corrections.txt"
    incoming_path = (
        NEURAL_ROOT / "data/raw/incoming_examples_2026-07-30.txt"
    )
    current = current_path.read_text(encoding="utf-8")
    merged, report, _ = merge(
        current,
        incoming_path.read_text(encoding="utf-8"),
        identity_count=200,
    )
    assert report["incoming_examples_recovered"] == 656
    assert report["exact_duplicates_excluded"] == 656
    assert report["conflicting_inputs_excluded"] == 0
    assert report["new_noisy_clean_examples"] == 0
    assert report["selected_identity_examples"] == 0
    assert report["final_examples"] == 1068
    assert merged == current


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


def test_normalized_groups_do_not_cross_dataset_partitions() -> None:
    split_payload = json.loads(
        (NEURAL_ROOT / "data/splits/LOCKED_SPLITS.json").read_text(encoding="utf-8")
    )
    rows = read_jsonl(NEURAL_ROOT / "data/processed/all_pairs.jsonl")
    split_for_id = {
        example_id: split_name
        for split_name, example_ids in split_payload["splits"].items()
        for example_id in example_ids
    }

    def partition(split_name: str) -> str:
        return "test" if split_name.startswith("test_") else split_name

    for group in build_groups(rows):
        partitions = {
            partition(split_for_id[row["id"]])
            for row in group
        }
        assert len(partitions) == 1

    signature_partitions: dict[str, set[str]] = {}
    for row in rows:
        row_partition = partition(split_for_id[row["id"]])
        for value in {signature(row["noisy"]), signature(row["clean"])} - {""}:
            signature_partitions.setdefault(value, set()).add(row_partition)
    assert all(
        len(partitions) == 1 for partitions in signature_partitions.values()
    )


def test_identity_examples_are_distributed_without_contaminating_changed_tests() -> None:
    split_payload = json.loads(
        (NEURAL_ROOT / "data/splits/LOCKED_SPLITS.json").read_text(encoding="utf-8")
    )
    rows_by_id = {
        row["id"]: row
        for row in read_jsonl(NEURAL_ROOT / "data/processed/all_pairs.jsonl")
    }
    split_rows = {
        split_name: [rows_by_id[example_id] for example_id in example_ids]
        for split_name, example_ids in split_payload["splits"].items()
    }
    assert any(row["is_unchanged"] for row in split_rows["train"])
    assert any(row["is_unchanged"] for row in split_rows["validation"])
    assert split_rows["test_clean"]
    assert all(row["is_unchanged"] for row in split_rows["test_clean"])
    assert all(not row["is_unchanged"] for row in split_rows["test_real"])
    assert all(not row["is_unchanged"] for row in split_rows["test_challenge"])


def test_expanded_dataset_audit_passes() -> None:
    report = json.loads(
        (
            NEURAL_ROOT / "data/reports/expanded_dataset_audit.json"
        ).read_text(encoding="utf-8")
    )
    assert report["status"] == "passed"
    assert all(report["checks"].values())


def test_expanded_neural_only_run_cannot_load_stale_synthetic_rows() -> None:
    config = json.loads(
        (NEURAL_ROOT / "configs/baseline.json").read_text(encoding="utf-8")
    )
    assert config["use_synthetic_training"] is False
