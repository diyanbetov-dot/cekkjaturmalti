from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Any


@dataclass(frozen=True, slots=True)
class DatasetSplit:
    name: str
    examples: tuple[dict[str, Any], ...]

    @property
    def count(self) -> int:
        return len(self.examples)


class GroupedSplitter:
    """Splits a dataset into train, validation, and test sets using group hashing to prevent data leakage."""

    def __init__(self, val_ratio: float = 0.15, test_ratio: float = 0.15, seed: int = 42) -> None:
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed

    def get_group_key(self, example: dict[str, Any]) -> str:
        group_id = example.get("group_id") or example.get("id") or example.get("raw_text", "")
        return str(group_id).strip().lower()

    def assign_split(self, example: dict[str, Any]) -> str:
        group_key = self.get_group_key(example)
        hash_input = f"{self.seed}:{group_key}".encode("utf-8")
        val = int(hashlib.sha256(hash_input).hexdigest(), 16) / float(1 << 256)
        if val < self.test_ratio:
            return "test"
        elif val < (self.test_ratio + self.val_ratio):
            return "validation"
        else:
            return "train"

    def split_examples(self, examples: Sequence[dict[str, Any]]) -> dict[str, DatasetSplit]:
        train_list: list[dict[str, Any]] = []
        val_list: list[dict[str, Any]] = []
        test_list: list[dict[str, Any]] = []

        for ex in examples:
            split_name = self.assign_split(ex)
            if split_name == "test":
                test_list.append(ex)
            elif split_name == "validation":
                val_list.append(ex)
            else:
                train_list.append(ex)

        return {
            "train": DatasetSplit("train", tuple(train_list)),
            "validation": DatasetSplit("validation", tuple(val_list)),
            "test": DatasetSplit("test", tuple(test_list)),
        }

    def save_splits(self, splits: dict[str, DatasetSplit], output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        for name, split in splits.items():
            path = output_dir / f"{name}.jsonl"
            with open(path, "w", encoding="utf-8") as f:
                for ex in split.examples:
                    f.write(json.dumps(ex, ensure_ascii=False) + "\n")
