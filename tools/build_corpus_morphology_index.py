from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path


def build(tagged_unigrams: Path, output: Path) -> dict[str, int]:
    with gzip.open(tagged_unigrams, "rt", encoding="utf-8") as stream:
        tagged = json.load(stream)

    rows: dict[str, dict[tuple[str, str | None, str | None], float]] = defaultdict(dict)
    for key, frequency in tagged.items():
        parts = key.split("\t")
        if len(parts) < 2:
            continue
        surface = parts[0].strip().casefold()
        pos = parts[1].strip().casefold()
        lemma = parts[2].strip().casefold() if len(parts) > 2 else ""
        root = parts[3].strip().casefold() if len(parts) > 3 else ""
        lemma_value = None if lemma in {"", "null"} else lemma
        root_value = None if root in {"", "null"} else root
        if not surface or not pos:
            continue
        analysis = (pos, lemma_value, root_value)
        rows[surface][analysis] = max(rows[surface].get(analysis, 0.0), float(frequency))

    payload = {
        surface: [
            {
                "pos": pos,
                "lemma": lemma,
                "root": root,
                "frequency": frequency,
            }
            for (pos, lemma, root), frequency in sorted(
                analyses.items(),
                key=lambda item: (
                    item[0][0],
                    item[0][1] or "",
                    item[0][2] or "",
                ),
            )
        ]
        for surface, analyses in sorted(rows.items())
    }
    with gzip.open(output, "wt", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, separators=(",", ":"))
    return {
        "surfaces": len(payload),
        "analyses": sum(len(values) for values in payload.values()),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("tagged_unigrams", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.tagged_unigrams, args.output), ensure_ascii=False, indent=2))
