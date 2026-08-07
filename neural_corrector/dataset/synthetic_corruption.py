from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from .analyze_pairs import read_jsonl
from .parse_ai_corrections import write_jsonl

REMOVE_DIACRITICS = str.maketrans(
    {"ċ": "c", "ġ": "g", "ħ": "h", "ż": "z", "Ċ": "C", "Ġ": "G", "Ħ": "H", "Ż": "Z"}
)
CONSONANTS = "bcċdfgġgħhħjklmnpqrstvwxyzż"


def _remove_diacritic(text: str, rng: random.Random) -> str:
    indexes = [index for index, char in enumerate(text) if char in "ċġħżĊĠĦŻ"]
    if not indexes:
        return text
    index = rng.choice(indexes)
    return text[:index] + text[index].translate(REMOVE_DIACRITICS) + text[index + 1 :]


def _remove_gh(text: str, rng: random.Random) -> str:
    matches = list(re.finditer("għ", text, re.IGNORECASE))
    if not matches:
        return text
    match = rng.choice(matches)
    replacement = "g" if rng.random() < 0.5 else ""
    return text[: match.start()] + replacement + text[match.end() :]


def _remove_separator(text: str, rng: random.Random) -> str:
    indexes = [index for index, char in enumerate(text) if char in "'’-"]
    if not indexes:
        return text
    index = rng.choice(indexes)
    return text[:index] + (" " if text[index] == "-" and rng.random() < 0.5 else "") + text[index + 1 :]


def _change_double(text: str, rng: random.Random) -> str:
    doubles = list(re.finditer(r"([bcċdfgġhħjklmnpqrstvwxyzż])\1", text, re.IGNORECASE))
    if doubles:
        match = rng.choice(doubles)
        return text[: match.start()] + match.group(1) + text[match.end() :]
    indexes = [
        index
        for index, char in enumerate(text)
        if char.casefold() in CONSONANTS and char.isalpha()
    ]
    if not indexes:
        return text
    index = rng.choice(indexes)
    return text[:index] + text[index] + text[index:]


def _merge_space(text: str, rng: random.Random) -> str:
    indexes = [index for index, char in enumerate(text) if char == " "]
    if not indexes:
        return text
    index = rng.choice(indexes)
    return text[:index] + text[index + 1 :]


def _change_case(text: str, rng: random.Random) -> str:
    indexes = [index for index, char in enumerate(text) if char.isalpha()]
    if not indexes:
        return text
    index = indexes[0]
    replacement = text[index].lower() if text[index].isupper() else text[index].upper()
    return text[:index] + replacement + text[index + 1 :]


def _remove_terminal_punctuation(text: str, rng: random.Random) -> str:
    return re.sub(r"([.?!])\s*$", "", text)


OPERATIONS = {
    "remove_diacritic": _remove_diacritic,
    "remove_gh": _remove_gh,
    "remove_apostrophe_or_hyphen": _remove_separator,
    "change_consonant_doubling": _change_double,
    "merge_space": _merge_space,
    "change_capitalization": _change_case,
    "remove_terminal_punctuation": _remove_terminal_punctuation,
}


def chunks(text: str, maximum: int = 260) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [part.strip() for part in parts if 2 <= len(part.strip()) <= maximum]


def generate(
    rows: list[dict],
    train_ids: set[str],
    variants_per_example: int,
    seed: int,
) -> list[dict]:
    rng = random.Random(seed)
    generated: list[dict] = []
    operation_names = list(OPERATIONS)
    for row in rows:
        if row["id"] not in train_ids:
            continue
        if "*" in row["clean"] or re.search(r"\b\w+/\w+\b", row["clean"]):
            continue
        for clean_chunk_index, clean in enumerate(chunks(row["clean"])):
            for variant in range(variants_per_example):
                noisy = clean
                applied: list[str] = []
                for operation_name in rng.sample(
                    operation_names, k=rng.randint(1, min(3, len(operation_names)))
                ):
                    changed = OPERATIONS[operation_name](noisy, rng)
                    if changed != noisy:
                        noisy = changed
                        applied.append(operation_name)
                if noisy == clean:
                    continue
                generated.append(
                    {
                        "id": f"synthetic-{row['id']}-{clean_chunk_index:03d}-{variant:02d}",
                        "noisy": noisy,
                        "clean": clean,
                        "source": "synthetic",
                        "source_group": row["id"],
                        "clean_source_id": row["id"],
                        "corruption_operations": applied,
                        "generator_version": "1.0.0",
                    }
                )
    return generated


def main() -> None:
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
        "--output",
        type=Path,
        default=Path("neural_corrector/data/processed/synthetic_train.jsonl"),
    )
    parser.add_argument("--variants", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1701)
    args = parser.parse_args()
    rows = read_jsonl(args.pairs)
    split_payload = json.loads(args.splits.read_text(encoding="utf-8"))
    train_ids = set(split_payload["splits"]["train"])
    generated = generate(rows, train_ids, args.variants, args.seed)
    write_jsonl(args.output, generated)
    print(json.dumps({"generated": len(generated), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
