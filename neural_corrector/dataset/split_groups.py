from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import unicodedata
from collections import defaultdict
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
    groups: list[list[dict]] = []
    signatures: list[str] = []
    for row in rows:
        sig = signature(row["noisy"])
        chosen = None
        for index, representative in enumerate(signatures):
            if sig == representative:
                chosen = index
                break
            if max(len(sig), len(representative)) <= 280:
                ratio = SequenceMatcher(None, sig, representative, autojunk=False).ratio()
                if ratio >= threshold:
                    chosen = index
                    break
        if chosen is None:
            signatures.append(sig)
            groups.append([row])
        else:
            groups[chosen].append(row)
    return groups


def _group_score(group: list[dict]) -> int:
    return max(len(row["error_tags"]) for row in group) + max(
        2 if "\n" in row["noisy"] else 0 for row in group
    )


def create_splits(rows: list[dict], seed: int = 1701) -> dict[str, list[str]]:
    rng = random.Random(seed)
    groups = build_groups(rows)
    clean_groups = [group for group in groups if all(row["is_unchanged"] for row in group)]
    remaining = [group for group in groups if group not in clean_groups]
    challenge_groups = sorted(remaining, key=_group_score, reverse=True)[
        : max(1, round(len(groups) * 0.08))
    ]
    remaining = [group for group in remaining if group not in challenge_groups]
    rng.shuffle(remaining)

    validation_count = max(1, round(len(groups) * 0.1))
    test_real_count = max(1, round(len(groups) * 0.1))
    validation_groups = remaining[:validation_count]
    test_real_groups = remaining[validation_count : validation_count + test_real_count]
    train_groups = remaining[validation_count + test_real_count :]

    if not clean_groups:
        clean_groups = train_groups[-1:]
        train_groups = train_groups[:-1]

    split_groups = {
        "train": train_groups,
        "validation": validation_groups,
        "test_real": test_real_groups,
        "test_clean": clean_groups,
        "test_challenge": challenge_groups,
    }
    result: dict[str, list[str]] = {}
    seen: set[str] = set()
    for split_name, grouped in split_groups.items():
        ids = [row["id"] for group in grouped for row in group]
        overlap = seen.intersection(ids)
        if overlap:
            raise ValueError(f"Split leakage detected in {split_name}: {sorted(overlap)}")
        seen.update(ids)
        result[split_name] = sorted(ids)
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
        "algorithm": "normalized-near-duplicate-groups-v1",
        "splits": splits,
        "manifest_sha256": hashlib.sha256(
            json.dumps(splits, sort_keys=True).encode("utf-8")
        ).hexdigest(),
    }
    lock_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
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

