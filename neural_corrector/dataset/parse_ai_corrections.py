from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .tags import infer_error_tags, normalize_for_analysis, suspicious_reasons

BLOCK_SEPARATOR_RE = re.compile(r"(?m)^===\s*$")
PAIR_RE = re.compile(
    r"\AINPUT:\s*\n(?P<input>.*?)\n\s*\nOUTPUT:\s*\n(?P<output>.*)\Z",
    re.DOTALL,
)


@dataclass(frozen=True)
class ParseResult:
    examples: list[dict]
    malformed: list[dict]
    report: dict
    suspicious: list[dict]


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _normalized_key(text: str) -> str:
    return re.sub(r"\s+", " ", normalize_for_analysis(text)).strip().casefold()


def parse_text(text: str, source_name: str = "AI corrections.txt") -> ParseResult:
    normalized_document = text.replace("\r\n", "\n").replace("\r", "\n")
    raw_blocks = BLOCK_SEPARATOR_RE.split(normalized_document)
    examples: list[dict] = []
    malformed: list[dict] = []

    for block_number, raw_block in enumerate(raw_blocks, start=1):
        if not raw_block.strip():
            continue
        block = raw_block.strip("\n")
        match = PAIR_RE.match(block)
        if not match:
            malformed.append(
                {
                    "block_number": block_number,
                    "raw_block": raw_block,
                    "reason": "block_does_not_match_input_output_format",
                }
            )
            continue
        noisy = match.group("input")
        clean = match.group("output")
        example_id = f"ai-corrections-{len(examples) + 1:06d}"
        examples.append(
            {
                "id": example_id,
                "noisy": noisy,
                "clean": clean,
                "source": source_name,
                "source_group": "manual",
                "raw_block": raw_block,
                "raw_input": noisy,
                "raw_output": clean,
                "normalized_noisy": normalize_for_analysis(noisy),
                "normalized_clean": normalize_for_analysis(clean),
                "error_tags": infer_error_tags(noisy, clean),
                "is_unchanged": noisy == clean,
                "review_status": "unreviewed",
            }
        )

    by_input: dict[str, list[dict]] = defaultdict(list)
    exact_counter: Counter[tuple[str, str]] = Counter()
    for example in examples:
        by_input[_normalized_key(example["noisy"])].append(example)
        exact_counter[(example["noisy"], example["clean"])] += 1

    conflicting_ids: set[str] = set()
    conflicts: list[dict] = []
    for key, grouped in by_input.items():
        outputs = {_normalized_key(item["clean"]) for item in grouped}
        if len(outputs) > 1:
            conflicting_ids.update(item["id"] for item in grouped)
            conflicts.append(
                {
                    "normalized_input": key,
                    "example_ids": [item["id"] for item in grouped],
                    "outputs": [item["clean"] for item in grouped],
                }
            )

    suspicious: list[dict] = []
    for example in examples:
        reasons = suspicious_reasons(example["noisy"], example["clean"])
        if example["id"] in conflicting_ids:
            reasons.append("conflicting_duplicate_input")
        if reasons:
            suspicious.append(
                {
                    "id": example["id"],
                    "noisy": example["noisy"],
                    "clean": example["clean"],
                    "reasons": sorted(set(reasons)),
                }
            )

    report = {
        "source": source_name,
        "total_blocks": len([block for block in raw_blocks if block.strip()]),
        "parsed_examples": len(examples),
        "malformed_blocks": len(malformed),
        "unchanged_examples": sum(item["is_unchanged"] for item in examples),
        "multiline_examples": sum(
            "\n" in item["noisy"] or "\n" in item["clean"] for item in examples
        ),
        "unique_inputs": len({_normalized_key(item["noisy"]) for item in examples}),
        "unique_outputs": len({_normalized_key(item["clean"]) for item in examples}),
        "exact_duplicate_pairs": sum(count - 1 for count in exact_counter.values() if count > 1),
        "conflicting_duplicate_inputs": conflicts,
        "suspicious_examples": len(suspicious),
    }
    return ParseResult(examples, malformed, report, suspicious)


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def parse_file(source: Path, output_root: Path) -> ParseResult:
    raw_bytes = source.read_bytes()
    text = raw_bytes.decode("utf-8-sig")
    result = parse_text(text, source.name)
    processed = output_root / "processed" / "all_pairs.jsonl"
    reports = output_root / "reports"
    write_jsonl(processed, result.examples)
    write_jsonl(reports / "suspicious_pairs.jsonl", result.suspicious)
    write_jsonl(reports / "malformed_blocks.jsonl", result.malformed)
    report = dict(result.report)
    report.update(
        {
            "source_path": str(source.resolve()),
            "source_sha256": _sha256_bytes(raw_bytes),
            "processed_path": str(processed.resolve()),
        }
    )
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "parse_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return ParseResult(result.examples, result.malformed, report, result.suspicious)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("AI corrections.txt"))
    parser.add_argument(
        "--output-root", type=Path, default=Path("neural_corrector/data")
    )
    args = parser.parse_args()
    result = parse_file(args.source, args.output_root)
    print(json.dumps(result.report, ensure_ascii=False, indent=2))
    if result.malformed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
