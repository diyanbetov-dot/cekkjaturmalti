from __future__ import annotations

import argparse
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path


def surface(key: str) -> str:
    return key.split("\t", 1)[0].strip().casefold()


def count_from_log(value: float) -> int:
    return max(1, round(math.expm1(float(value))))


def log_count(value: int) -> float:
    return round(math.log1p(value), 4)


def rebuild(corpus_dir: Path) -> dict[str, int]:
    unigram_path = corpus_dir / "unigrams.json.gz"
    bigram_path = corpus_dir / "bigrams.json.gz"
    meta_path = corpus_dir / "meta.json"

    with gzip.open(unigram_path, "rt", encoding="utf-8") as stream:
        legacy_unigrams = json.load(stream)
    with gzip.open(bigram_path, "rt", encoding="utf-8") as stream:
        legacy_bigrams = json.load(stream)

    unigram_counts: dict[str, int] = defaultdict(int)
    for key, value in legacy_unigrams.items():
        token = surface(key)
        if token:
            unigram_counts[token] += count_from_log(value)

    bigram_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for left_key, following in legacy_bigrams.items():
        left = surface(left_key)
        if not left:
            continue
        for right_key, value in following.items():
            right = surface(right_key)
            if right:
                bigram_counts[left][right] += count_from_log(value)

    repaired_unigrams = {
        key: log_count(value)
        for key, value in sorted(unigram_counts.items())
    }
    repaired_bigrams = {
        left: {
            right: log_count(value)
            for right, value in sorted(following.items())
        }
        for left, following in sorted(bigram_counts.items())
    }

    with gzip.open(unigram_path, "wt", encoding="utf-8") as stream:
        json.dump(repaired_unigrams, stream, ensure_ascii=False, separators=(",", ":"))
    with gzip.open(bigram_path, "wt", encoding="utf-8") as stream:
        json.dump(repaired_bigrams, stream, ensure_ascii=False, separators=(",", ":"))

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["preprocessing_version"] = "hope-surface-repair-1.0"
    meta["repaired_from_tagged_keys"] = True
    stats = meta.setdefault("stats", {})
    stats["vocab_size"] = len(repaired_unigrams)
    stats["bigram_count"] = sum(len(values) for values in repaired_bigrams.values())
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "unigrams": len(repaired_unigrams),
        "bigram_roots": len(repaired_bigrams),
        "bigrams": sum(len(values) for values in repaired_bigrams.values()),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus_dir", nargs="?", default="corpus", type=Path)
    args = parser.parse_args()
    print(json.dumps(rebuild(args.corpus_dir), ensure_ascii=False, indent=2))
