from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from .tags import normalize_for_analysis

MARKER_RE = re.compile(r"(?m)^(INPUT|OUTPUT):?\s*$")
SEPARATOR_RE = re.compile(r"(?m)^===\S*\s*$")


@dataclass(frozen=True)
class LoosePair:
    noisy: str
    clean: str
    raw_block: str


def normalized_key(text: str) -> str:
    return re.sub(r"\s+", " ", normalize_for_analysis(text)).strip().casefold()


def parse_loose_pairs(text: str) -> tuple[list[LoosePair], list[dict]]:
    document = text.replace("\r\n", "\n").replace("\r", "\n")
    markers = list(MARKER_RE.finditer(document))
    pairs: list[LoosePair] = []
    problems: list[dict] = []
    cursor = 0
    while cursor < len(markers):
        input_marker = markers[cursor]
        if input_marker.group(1) != "INPUT":
            problems.append(
                {
                    "offset": input_marker.start(),
                    "reason": "output_marker_without_preceding_input",
                }
            )
            cursor += 1
            continue
        if cursor + 1 >= len(markers) or markers[cursor + 1].group(1) != "OUTPUT":
            problems.append(
                {
                    "offset": input_marker.start(),
                    "reason": "input_marker_without_following_output",
                }
            )
            cursor += 1
            continue
        output_marker = markers[cursor + 1]
        next_input = (
            markers[cursor + 2]
            if cursor + 2 < len(markers)
            and markers[cursor + 2].group(1) == "INPUT"
            else None
        )
        block_end = next_input.start() if next_input else len(document)
        noisy = document[input_marker.end() : output_marker.start()].strip()
        raw_clean = document[output_marker.end() : block_end]
        clean = SEPARATOR_RE.sub("", raw_clean).strip()
        if not noisy or not clean:
            problems.append(
                {
                    "offset": input_marker.start(),
                    "reason": "empty_input_or_output",
                }
            )
        else:
            pairs.append(
                LoosePair(
                    noisy=noisy,
                    clean=clean,
                    raw_block=document[input_marker.start() : block_end],
                )
            )
        cursor += 2
    return pairs, problems


def canonical_block(noisy: str, clean: str) -> str:
    return f"INPUT:\n{noisy}\n\nOUTPUT:\n{clean}"


def select_identities(
    pairs: list[LoosePair],
    existing_pairs: list[LoosePair],
    count: int,
) -> list[LoosePair]:
    all_pairs = existing_pairs + pairs
    noisy_to_outputs: dict[str, set[str]] = {}
    for pair in all_pairs:
        noisy_to_outputs.setdefault(normalized_key(pair.noisy), set()).add(
            normalized_key(pair.clean)
        )

    candidates: dict[str, str] = {}
    for pair in pairs:
        key = normalized_key(pair.clean)
        if not key or "\n" in pair.clean or len(pair.clean) > 320:
            continue
        if key in noisy_to_outputs and noisy_to_outputs[key] != {key}:
            continue
        candidates.setdefault(key, pair.clean)

    buckets: dict[str, list[tuple[str, str]]] = {
        "short": [],
        "medium": [],
        "long": [],
    }
    for key, clean in candidates.items():
        length = len(clean)
        bucket = "short" if length <= 45 else "medium" if length <= 130 else "long"
        digest = hashlib.sha256(clean.encode("utf-8")).hexdigest()
        buckets[bucket].append((digest, clean))
    for values in buckets.values():
        values.sort()

    targets = {
        "short": round(count * 0.35),
        "medium": round(count * 0.45),
    }
    targets["long"] = count - targets["short"] - targets["medium"]
    selected: list[str] = []
    leftovers: list[tuple[str, str]] = []
    for bucket, values in buckets.items():
        take = min(targets[bucket], len(values))
        selected.extend(clean for _, clean in values[:take])
        leftovers.extend(values[take:])
    if len(selected) < count:
        leftovers.sort()
        selected.extend(clean for _, clean in leftovers[: count - len(selected)])

    return [
        LoosePair(clean, clean, canonical_block(clean, clean))
        for clean in selected[:count]
    ]


def merge(
    current_text: str,
    incoming_text: str,
    identity_count: int,
) -> tuple[str, dict, str]:
    current_pairs, current_problems = parse_loose_pairs(current_text)
    incoming_pairs, incoming_problems = parse_loose_pairs(incoming_text)
    if current_problems:
        raise ValueError(f"Current file is malformed: {current_problems}")
    if incoming_problems:
        raise ValueError(f"Incoming file has unrecoverable pairs: {incoming_problems}")

    accepted: list[LoosePair] = []
    duplicates: list[dict] = []
    conflicts: list[dict] = []
    known_pairs = {
        (normalized_key(pair.noisy), normalized_key(pair.clean))
        for pair in current_pairs
    }
    outputs_by_input: dict[str, set[str]] = {}
    for pair in current_pairs:
        outputs_by_input.setdefault(normalized_key(pair.noisy), set()).add(
            normalized_key(pair.clean)
        )

    for pair in incoming_pairs:
        input_key = normalized_key(pair.noisy)
        output_key = normalized_key(pair.clean)
        if (input_key, output_key) in known_pairs:
            duplicates.append({"noisy": pair.noisy, "clean": pair.clean})
            continue
        existing_outputs = outputs_by_input.get(input_key, set())
        if existing_outputs and output_key not in existing_outputs:
            conflicts.append(
                {
                    "noisy": pair.noisy,
                    "incoming_clean": pair.clean,
                    "existing_clean_keys": sorted(existing_outputs),
                    "decision": "excluded_incoming_conflict",
                }
            )
            continue
        accepted.append(pair)
        known_pairs.add((input_key, output_key))
        outputs_by_input.setdefault(input_key, set()).add(output_key)

    identities = select_identities(accepted, current_pairs, identity_count)
    accepted_identities: list[LoosePair] = []
    for identity in identities:
        key = normalized_key(identity.clean)
        if (key, key) in known_pairs:
            continue
        accepted_identities.append(identity)
        known_pairs.add((key, key))

    appended_blocks = accepted + accepted_identities
    if appended_blocks:
        addition = "\n\n===\n\n".join(
            canonical_block(pair.noisy, pair.clean) for pair in appended_blocks
        )
        merged_text = current_text.rstrip() + "\n\n===\n\n" + addition + "\n"
    else:
        merged_text = current_text
    report = {
        "current_examples": len(current_pairs),
        "incoming_markers": {
            "input": len(re.findall(r"(?m)^INPUT:?\s*$", incoming_text)),
            "output": len(re.findall(r"(?m)^OUTPUT:?\s*$", incoming_text)),
        },
        "incoming_examples_recovered": len(incoming_pairs),
        "incoming_structural_repairs": {
            "missing_input_colons": len(
                re.findall(r"(?m)^INPUT\s*$", incoming_text)
            ),
            "noncanonical_separators": len(
                re.findall(r"(?m)^===\S+\s*$", incoming_text)
            ),
            "missing_separators": max(
                0,
                len(incoming_pairs)
                - 1
                - len(re.findall(r"(?m)^===\s*$", incoming_text))
                - len(re.findall(r"(?m)^===\S+\s*$", incoming_text)),
            ),
        },
        "exact_duplicates_excluded": len(duplicates),
        "conflicting_inputs_excluded": len(conflicts),
        "new_noisy_clean_examples": len(accepted),
        "selected_identity_examples": len(accepted_identities),
        "final_examples": len(current_pairs)
        + len(accepted)
        + len(accepted_identities),
        "duplicates": duplicates,
        "conflicts": conflicts,
        "incoming_sha256": hashlib.sha256(
            incoming_text.encode("utf-8")
        ).hexdigest(),
    }
    canonical_incoming = (
        "\n\n===\n\n".join(
            canonical_block(pair.noisy, pair.clean) for pair in incoming_pairs
        )
        + "\n"
    )
    return merged_text, report, canonical_incoming


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current", type=Path, default=Path("AI corrections.txt"))
    parser.add_argument("--incoming", type=Path, required=True)
    parser.add_argument("--identity-count", type=int, default=200)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("neural_corrector/data/reports/merge_report.json"),
    )
    parser.add_argument(
        "--raw-copy",
        type=Path,
        default=Path("neural_corrector/data/raw/incoming_examples_2026-07-30.txt"),
    )
    args = parser.parse_args()
    current_text = args.current.read_text(encoding="utf-8")
    incoming_text = args.incoming.read_text(encoding="utf-8")
    merged_text, report, canonical_incoming = merge(
        current_text, incoming_text, args.identity_count
    )
    if args.apply:
        args.current.write_text(merged_text, encoding="utf-8")
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        args.raw_copy.parent.mkdir(parents=True, exist_ok=True)
        args.raw_copy.write_text(incoming_text, encoding="utf-8")
        canonical_path = args.raw_copy.with_name(
            args.raw_copy.stem + "_canonical.txt"
        )
        canonical_path.write_text(canonical_incoming, encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
