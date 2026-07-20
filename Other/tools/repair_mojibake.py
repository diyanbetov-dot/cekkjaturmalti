# -*- coding: utf-8 -*-
"""Repair UTF-8 text that was decoded as Windows-1252/CP1252.

This is intended for benchmark fixtures and pasted test text, not as part of
the live spellchecker pipeline. It fixes recoverable mojibake such as
``gÄ§andek`` -> ``għandek`` while preserving already-correct Maltese text.
"""

from __future__ import annotations

import argparse
from pathlib import Path


MOJIBAKE_STARTERS = frozenset("ÃÄÅÂâ")


def _decode_cp1252_utf8(chunk: str) -> str | None:
    try:
        repaired = chunk.encode("cp1252").decode("utf-8")
    except UnicodeError:
        return None
    return repaired if repaired != chunk else None


def repair_mojibake_text(text: str) -> str:
    """Return text with recoverable CP1252/UTF-8 mojibake repaired.

    A whole-string roundtrip is preferred when possible. If the string already
    contains real Maltese characters, the whole roundtrip can fail, so the
    scanner repairs only known mojibake-looking byte fragments.
    """

    whole = _decode_cp1252_utf8(text)
    if whole is not None:
        return whole

    repaired: list[str] = []
    index = 0
    while index < len(text):
        char = text[index]
        if char in MOJIBAKE_STARTERS:
            replacement = None
            replacement_len = 0
            for width in (4, 3, 2):
                chunk = text[index : index + width]
                candidate = _decode_cp1252_utf8(chunk)
                if candidate is not None:
                    replacement = candidate
                    replacement_len = width
                    break
            if replacement is not None:
                repaired.append(replacement)
                index += replacement_len
                continue

        repaired.append(char)
        index += 1

    return "".join(repaired)


def repair_file(input_path: Path, output_path: Path | None = None) -> Path:
    source = input_path.read_text(encoding="utf-8")
    repaired = repair_mojibake_text(source)
    target = output_path or input_path
    target.write_text(repaired, encoding="utf-8", newline="")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Repair recoverable CP1252/UTF-8 mojibake in a text file."
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite the input file after repair.",
    )
    args = parser.parse_args()

    if args.output and args.in_place:
        parser.error("Use either --output or --in-place, not both.")
    if not args.output and not args.in_place:
        parser.error("Choose --output PATH or --in-place.")

    target = repair_file(args.input, None if args.in_place else args.output)
    print(f"Repaired mojibake text written to {target}")


if __name__ == "__main__":
    main()
