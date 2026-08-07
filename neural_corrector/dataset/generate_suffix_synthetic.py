from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path

from Essentials.helpers.suffix_rules import ALL_SUFFIXES, MalteseSuffixRules
from Essentials.helpers.verb_form_index import MalteseVerbFormIndex
from neural_corrector.dataset.build_suffix_bloom import OrthographyAdapter

ASCII_DIACRITICS = str.maketrans(
    {
        "ċ": "c",
        "ġ": "g",
        "ħ": "h",
        "ż": "z",
        "Ċ": "C",
        "Ġ": "G",
        "Ħ": "H",
        "Ż": "Z",
    }
)


def corruption_variants(clean: str) -> list[tuple[str, str]]:
    variants: list[tuple[str, str]] = []

    def add(value: str, operation: str) -> None:
        if value != clean and value and (value, operation) not in variants:
            variants.append((value, operation))

    add(clean.translate(ASCII_DIACRITICS), "remove_diacritics")
    for match in re.finditer("għ", clean):
        add(
            clean[: match.start()] + "g" + clean[match.end() :],
            "reduce_gh_to_g",
        )
        add(
            clean[: match.start()] + clean[match.end() :],
            "remove_gh",
        )
    for match in re.finditer(r"([bcċdfgġhħjklmnpqrstvwxyzż])\1", clean):
        add(
            clean[: match.start()]
            + match.group(1)
            + clean[match.end() :],
            "remove_doubled_consonant",
        )
    for marker, replacement in (
        ("hiel", "il"),
        ("hie", "i"),
        ("ie", "i"),
        ("lhom", "lom"),
        ("lha", "la"),
        ("lh", "l"),
    ):
        start = clean.find(marker)
        if start >= 0:
            add(
                clean[:start]
                + replacement
                + clean[start + len(marker) :],
                f"reduce_{marker}_to_{replacement}",
            )
    return variants


def compound_corruption_variants(
    clean: str,
    *,
    max_results: int = 120,
) -> list[tuple[str, str]]:
    """Generate two-step compound corruptions by chaining single corruptions.

    Covers colloquial double-suffix omissions such as::

        agħmilhulu -> amilulu  (remove_gh + reduce_lh_to_l)
        ibgħathielu -> ibatilu  (remove_gh + reduce_hie_to_i)

    Only returns pairs that differ from ``clean`` and are non-empty.
    """
    seen: set[tuple[str, str]] = set()
    compound: list[tuple[str, str]] = []
    for intermediate, op1 in corruption_variants(clean):
        for result, op2 in corruption_variants(intermediate):
            key = (result, f"{op1}+{op2}")
            if result != clean and result and key not in seen:
                seen.add(key)
                compound.append(key)
                if len(compound) >= max_results:
                    return compound
    return compound



def generate(
    verb_files: list[Path],
    *,
    per_suffix_family: int,
    compound_per_suffix_family: int = 100,
    seed: int,
) -> tuple[list[dict], dict]:
    adapter = OrthographyAdapter()
    verb_index = MalteseVerbFormIndex(
        verb_files,
        normalizer=adapter._normalize_word,
        grapheme_splitter=adapter._graphemes,
    )
    rules = MalteseSuffixRules(
        spellchecker=adapter,
        verb_index=verb_index,
    )
    records = [
        record
        for grouped_records in verb_index.by_word.values()
        for record in grouped_records
    ]
    rng = random.Random(seed)
    proposed = []
    family_counts = {}

    for family_index, spec in enumerate(ALL_SUFFIXES):
        selected = {}
        compound_selected: dict[str, dict] = {}
        start = rng.randrange(len(records))
        step = 7919 + family_index * 2
        while math.gcd(step, len(records)) != 1:
            step += 2
        for offset in range(len(records)):
            record = records[(start + offset * step) % len(records)]
            for candidate in rules.generate_for_record_and_spec(record, spec):
                clean = candidate.surface
                if not (4 <= len(clean) <= 28) or clean in selected:
                    continue
                variants = corruption_variants(clean)
                if not variants:
                    continue
                noisy, operation = rng.choice(variants)
                selected[clean] = {
                    "noisy": noisy,
                    "clean": clean,
                    "suffix_family": spec.label,
                    "suffix_kind": spec.kind,
                    "corruption_operations": [operation],
                    "base": candidate.base,
                    "raw_tag": candidate.raw_tag,
                }
                # Compound corruption: pick one two-step variant when available
                if len(compound_selected) < compound_per_suffix_family and clean not in compound_selected:
                    comp_variants = compound_corruption_variants(clean)
                    if comp_variants:
                        comp_noisy, comp_ops = rng.choice(comp_variants)
                        compound_selected[clean] = {
                            "noisy": comp_noisy,
                            "clean": clean,
                            "suffix_family": spec.label,
                            "suffix_kind": spec.kind,
                            "corruption_operations": comp_ops.split("+"),
                            "base": candidate.base,
                            "raw_tag": candidate.raw_tag,
                        }
                if len(selected) >= per_suffix_family and len(compound_selected) >= compound_per_suffix_family:
                    break
            if len(selected) >= per_suffix_family and len(compound_selected) >= compound_per_suffix_family:
                break
        family_counts[spec.label] = len(selected)
        proposed.extend(selected.values())
        proposed.extend(compound_selected.values())

    targets_by_noisy = defaultdict(set)
    for row in proposed:
        targets_by_noisy[row["noisy"]].add(row["clean"])
    conflicts = {
        noisy: sorted(targets)
        for noisy, targets in targets_by_noisy.items()
        if len(targets) > 1
    }

    rows = []
    for row in proposed:
        if row["noisy"] in conflicts:
            continue
        rows.append(
            {
                "id": f"suffix-synthetic-{len(rows) + 1:06d}",
                **row,
                "source": "suffix_synthetic",
                "source_group": f"suffix-{row['suffix_family']}",
                "generator_version": "1.0.0",
            }
        )

    identity_target = max(1, len(rows) // 5)
    identity_candidates = list(dict.fromkeys(row["clean"] for row in rows))
    rng.shuffle(identity_candidates)
    for clean in identity_candidates[:identity_target]:
        rows.append(
            {
                "id": f"suffix-identity-{clean.encode('utf-8').hex()[:20]}",
                "noisy": clean,
                "clean": clean,
                "source": "suffix_synthetic_identity",
                "source_group": "suffix-identity",
                "suffix_family": "IDENTITY",
                "suffix_kind": "IDENTITY",
                "corruption_operations": [],
                "generator_version": "1.0.0",
            }
        )
    rng.shuffle(rows)
    report = {
        "rows": len(rows),
        "changed_rows": sum(row["noisy"] != row["clean"] for row in rows),
        "identity_rows": sum(row["noisy"] == row["clean"] for row in rows),
        "conflicting_noisy_forms_removed": len(conflicts),
        "per_suffix_family_target": per_suffix_family,
        "family_counts": family_counts,
        "seed": seed,
    }
    return rows, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--verb-file",
        action="append",
        type=Path,
        dest="verb_files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "neural_corrector/data/processed/synthetic_suffix_train.jsonl"
        ),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(
            "neural_corrector/data/reports/synthetic_suffix_report.json"
        ),
    )
    parser.add_argument("--per-family", type=int, default=250)
    parser.add_argument("--seed", type=int, default=1701)
    args = parser.parse_args()
    verb_files = args.verb_files or [
        Path("Essentials/finaldics/verbmt_semitic.dic"),
        Path("Essentials/finaldics/verbmt_nonsemitic.dic"),
    ]
    rows, report = generate(
        verb_files,
        per_suffix_family=args.per_family,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
