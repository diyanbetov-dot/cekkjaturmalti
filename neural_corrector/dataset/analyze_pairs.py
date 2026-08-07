from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def approximate_edit_distance(source: str, target: str) -> int:
    matcher = SequenceMatcher(None, source, target, autojunk=False)
    distance = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "equal":
            distance += max(i2 - i1, j2 - j1)
    return distance


def token_distance(source: str, target: str) -> int:
    return approximate_edit_distance(source.split(), target.split())


def summarize(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": round(statistics.fmean(values), 3),
        "median": statistics.median(values),
        "p90": ordered[min(len(ordered) - 1, math.floor(len(ordered) * 0.9))],
    }


def analyze(rows: list[dict]) -> dict:
    tag_counts = Counter(tag for row in rows for tag in row["error_tags"])
    char_distances = [
        approximate_edit_distance(row["noisy"], row["clean"]) for row in rows
    ]
    word_distances = [token_distance(row["noisy"], row["clean"]) for row in rows]
    chars = Counter("".join(row["noisy"] + row["clean"] for row in rows))
    rare_chars = [
        {"character": char, "codepoint": f"U+{ord(char):04X}", "count": count}
        for char, count in sorted(chars.items(), key=lambda item: (item[1], item[0]))
        if count <= 2 and not char.isspace()
    ]
    normalized_inputs = Counter(
        re.sub(r"\s+", " ", row["noisy"]).strip().casefold() for row in rows
    )
    return {
        "total_examples": len(rows),
        "single_line_examples": sum(
            "\n" not in row["noisy"] and "\n" not in row["clean"] for row in rows
        ),
        "multiline_examples": sum(
            "\n" in row["noisy"] or "\n" in row["clean"] for row in rows
        ),
        "unchanged_examples": sum(row["is_unchanged"] for row in rows),
        "duplicate_input_groups": sum(count > 1 for count in normalized_inputs.values()),
        "error_category_distribution": dict(sorted(tag_counts.items())),
        "input_character_lengths": summarize([len(row["noisy"]) for row in rows]),
        "output_character_lengths": summarize([len(row["clean"]) for row in rows]),
        "character_edit_distance": summarize(char_distances),
        "word_edit_distance": summarize(word_distances),
        "rare_characters": rare_chars,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pairs",
        type=Path,
        default=Path("neural_corrector/data/processed/all_pairs.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("neural_corrector/data/reports/dataset_analysis.json"),
    )
    args = parser.parse_args()
    report = analyze(read_jsonl(args.pairs))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

