from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

from .analyze_pairs import read_jsonl

LOCK_FILENAME = "LOCKED_SPLITS.json"


def signature(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.casefold())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.replace("għ", "gh")
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def build_groups(rows: list[dict], threshold: float = 0.9) -> list[list[dict]]:
    parents = list(range(len(rows)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    row_signatures = [
        {signature(row["noisy"]), signature(row["clean"])} - {""} for row in rows
    ]
    exact_owner: dict[str, int] = {}
    for index, signatures in enumerate(row_signatures):
        for value in signatures:
            if value in exact_owner:
                union(index, exact_owner[value])
            else:
                exact_owner[value] = index

    representatives: list[tuple[int, str]] = []
    for index, signatures in enumerate(row_signatures):
        primary = min(signatures, key=len, default="")
        if not primary:
            continue
        for other_index, other in representatives:
            if max(len(primary), len(other)) <= 280 and SequenceMatcher(
                None, primary, other, autojunk=False
            ).ratio() >= threshold:
                union(index, other_index)
                break
        representatives.append((index, primary))

    grouped: dict[int, list[dict]] = {}
    for index, row in enumerate(rows):
        grouped.setdefault(find(index), []).append(row)
    return list(grouped.values())


def _group_score(group: list[dict]) -> int:
    return max(len(row["error_tags"]) for row in group) + max(
        2 if "\n" in row["noisy"] else 0 for row in group
    )


def create_splits(rows: list[dict], seed: int = 1701) -> dict[str, list[str]]:
    rng = random.Random(seed)
    groups = build_groups(rows)
    challenge_groups = sorted(groups, key=_group_score, reverse=True)[
        : max(1, round(len(groups) * 0.08))
    ]
    remaining = [group for group in groups if group not in challenge_groups]
    rng.shuffle(remaining)

    validation_count = max(1, round(len(groups) * 0.1))
    test_real_count = max(1, round(len(groups) * 0.1))
    validation_groups = remaining[:validation_count]
    test_real_groups = remaining[validation_count : validation_count + test_real_count]
    train_groups = remaining[validation_count + test_real_count :]

    result: dict[str, list[str]] = {
        "train": sorted(
            row["id"] for group in train_groups for row in group
        ),
        "validation": sorted(
            row["id"] for group in validation_groups for row in group
        ),
        "test_real": sorted(
            row["id"]
            for group in test_real_groups
            for row in group
            if not row["is_unchanged"]
        ),
        "test_clean": sorted(
            row["id"]
            for group in test_real_groups + challenge_groups
            for row in group
            if row["is_unchanged"]
        ),
        "test_challenge": sorted(
            row["id"]
            for group in challenge_groups
            for row in group
            if not row["is_unchanged"]
        ),
    }
    if not result["test_clean"]:
        raise ValueError("No clean identity examples reached the locked test partition.")
    seen: set[str] = set()
    for split_name, ids in result.items():
        overlap = seen.intersection(ids)
        if overlap:
            raise ValueError(f"Split leakage detected in {split_name}: {sorted(overlap)}")
        seen.update(ids)
    if seen != {row["id"] for row in rows}:
        missing = {row["id"] for row in rows} - seen
        raise ValueError(f"Examples missing from splits: {sorted(missing)}")
    return result


def write_splits(
    rows: list[dict], output_dir: Path, seed: int = 1701, force: bool = False
) -> dict:
    lock_path = output_dir / LOCK_FILENAME
    if lock_path.exists() and not force:
        return json.loads(lock_path.read_text(encoding="utf-8"))
    splits = create_splits(rows, seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, ids in splits.items():
        (output_dir / f"{name}.json").write_text(
            json.dumps({"split": name, "ids": ids}, indent=2) + "\n",
            encoding="utf-8",
        )
    payload = {
        "locked": True,
        "seed": seed,
        "algorithm": "normalized-noisy-clean-near-duplicate-groups-v2",
        "splits": splits,
        "manifest_sha256": hashlib.sha256(
            json.dumps(splits, sort_keys=True).encode("utf-8")
        ).hexdigest(),
    }
    lock_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


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
        "--output-dir",
        type=Path,
        default=Path("neural_corrector/data/splits"),
    )
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    payload = write_splits(
        read_jsonl(args.pairs), args.output_dir, args.seed, args.force
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
