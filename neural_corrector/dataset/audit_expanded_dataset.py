from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

from .analyze_pairs import read_jsonl
from .split_groups import build_groups, signature

TEST_SPLITS = {"test_real", "test_clean", "test_challenge"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _partition(split_name: str) -> str:
    return "test" if split_name in TEST_SPLITS else split_name


def audit(
    rows: list[dict],
    split_payload: dict,
    parse_report: dict,
    merge_report: dict,
) -> dict:
    rows_by_id = {row["id"]: row for row in rows}
    split_for_id: dict[str, str] = {}
    duplicate_assignments: list[dict] = []
    for split_name, example_ids in split_payload["splits"].items():
        for example_id in example_ids:
            previous = split_for_id.setdefault(example_id, split_name)
            if previous != split_name:
                duplicate_assignments.append(
                    {
                        "id": example_id,
                        "first_split": previous,
                        "second_split": split_name,
                    }
                )

    expected_ids = set(rows_by_id)
    assigned_ids = set(split_for_id)
    missing_ids = sorted(expected_ids - assigned_ids)
    unknown_ids = sorted(assigned_ids - expected_ids)

    group_leaks: list[dict] = []
    for group_number, group in enumerate(build_groups(rows), start=1):
        partitions = {
            _partition(split_for_id[row["id"]])
            for row in group
            if row["id"] in split_for_id
        }
        if len(partitions) > 1:
            group_leaks.append(
                {
                    "group": group_number,
                    "ids": [row["id"] for row in group],
                    "partitions": sorted(partitions),
                }
            )

    signature_partitions: dict[str, set[str]] = {}
    signature_ids: dict[str, list[str]] = {}
    for row in rows:
        if row["id"] not in split_for_id:
            continue
        partition = _partition(split_for_id[row["id"]])
        for value in {signature(row["noisy"]), signature(row["clean"])} - {""}:
            signature_partitions.setdefault(value, set()).add(partition)
            signature_ids.setdefault(value, []).append(row["id"])
    signature_leaks = [
        {
            "signature": value,
            "partitions": sorted(partitions),
            "ids": sorted(set(signature_ids[value])),
        }
        for value, partitions in signature_partitions.items()
        if len(partitions) > 1
    ]

    split_counts: dict[str, dict] = {}
    for split_name, example_ids in split_payload["splits"].items():
        split_rows = [rows_by_id[example_id] for example_id in example_ids]
        split_counts[split_name] = {
            "examples": len(split_rows),
            "identity": sum(row["is_unchanged"] for row in split_rows),
            "changed": sum(not row["is_unchanged"] for row in split_rows),
        }

    normalized_inputs: dict[str, set[str]] = {}
    exact_pairs = Counter()
    for row in rows:
        noisy_signature = signature(row["noisy"])
        clean_signature = signature(row["clean"])
        normalized_inputs.setdefault(noisy_signature, set()).add(clean_signature)
        exact_pairs[(row["noisy"], row["clean"])] += 1
    conflicts = [
        {"normalized_input": noisy, "normalized_outputs": sorted(outputs)}
        for noisy, outputs in normalized_inputs.items()
        if len(outputs) > 1
    ]
    duplicate_pairs = sum(count - 1 for count in exact_pairs.values() if count > 1)

    checks = {
        "all_examples_assigned_once": not (
            duplicate_assignments or missing_ids or unknown_ids
        ),
        "no_group_partition_leakage": not group_leaks,
        "no_exact_signature_partition_leakage": not signature_leaks,
        "no_exact_duplicate_pairs": duplicate_pairs == 0,
        "no_conflicting_normalized_inputs": not conflicts,
        "train_has_identity_examples": split_counts["train"]["identity"] > 0,
        "validation_has_identity_examples": (
            split_counts["validation"]["identity"] > 0
        ),
        "test_clean_contains_only_identity_examples": (
            split_counts["test_clean"]["changed"] == 0
        ),
        "changed_test_sets_contain_no_identity_examples": (
            split_counts["test_real"]["identity"] == 0
            and split_counts["test_challenge"]["identity"] == 0
        ),
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "dataset": {
            "examples": len(rows),
            "identity_examples": sum(row["is_unchanged"] for row in rows),
            "changed_examples": sum(not row["is_unchanged"] for row in rows),
            "exact_duplicate_pairs": duplicate_pairs,
            "conflicting_normalized_inputs": conflicts,
            "suspicious_examples": parse_report["suspicious_examples"],
        },
        "merge": {
            "incoming_examples_recovered": merge_report[
                "incoming_examples_recovered"
            ],
            "new_noisy_clean_examples": merge_report[
                "new_noisy_clean_examples"
            ],
            "selected_identity_examples": merge_report[
                "selected_identity_examples"
            ],
            "structural_repairs": merge_report["incoming_structural_repairs"],
        },
        "split_algorithm": split_payload["algorithm"],
        "split_manifest_sha256": split_payload["manifest_sha256"],
        "split_counts": split_counts,
        "leakage": {
            "duplicate_assignments": duplicate_assignments,
            "missing_ids": missing_ids,
            "unknown_ids": unknown_ids,
            "group_partition_leaks": group_leaks,
            "exact_signature_partition_leaks": signature_leaks,
        },
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pairs",
        type=Path,
        default=Path("neural_corrector/data/processed/all_pairs.jsonl"),
    )
    parser.add_argument(
        "--splits",
        type=Path,
        default=Path("neural_corrector/data/splits/LOCKED_SPLITS.json"),
    )
    parser.add_argument(
        "--parse-report",
        type=Path,
        default=Path("neural_corrector/data/reports/parse_report.json"),
    )
    parser.add_argument(
        "--merge-report",
        type=Path,
        default=Path("neural_corrector/data/reports/merge_report.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "neural_corrector/data/reports/expanded_dataset_audit.json"
        ),
    )
    args = parser.parse_args()

    report = audit(
        read_jsonl(args.pairs),
        json.loads(args.splits.read_text(encoding="utf-8")),
        json.loads(args.parse_report.read_text(encoding="utf-8")),
        json.loads(args.merge_report.read_text(encoding="utf-8")),
    )
    report["artifacts"] = {
        "pairs_sha256": _sha256(args.pairs),
        "splits_sha256": _sha256(args.splits),
        "parse_report_sha256": _sha256(args.parse_report),
        "merge_report_sha256": _sha256(args.merge_report),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
