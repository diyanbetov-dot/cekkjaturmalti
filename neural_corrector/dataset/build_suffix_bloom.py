from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
import time
import unicodedata
from pathlib import Path

from Essentials.helpers.suffix_rules import ALL_SUFFIXES, MalteseSuffixRules
from Essentials.helpers.verb_form_index import MalteseVerbFormIndex

MAGIC = b"CMSFXBL1"
HEADER = struct.Struct("<8sQIQ")
DEFAULT_BITS = 1 << 28
DEFAULT_HASHES = 7


class OrthographyAdapter:
    VOWELS = set("aeiouàèìòù")

    @staticmethod
    def _normalize_word(word: str) -> str:
        return (
            unicodedata.normalize("NFC", str(word).strip().lower())
            .replace("\u2019", "'")
            .replace("\u02bc", "'")
        )

    def _graphemes(self, word: str) -> list[str]:
        word = self._normalize_word(word)
        graphemes = []
        index = 0
        while index < len(word):
            if word.startswith("għ", index):
                graphemes.append("għ")
                index += 2
            else:
                graphemes.append(word[index])
                index += 1
        return graphemes

    @staticmethod
    def _from_graphemes(graphemes) -> str:
        return "".join(graphemes)


def bloom_positions(value: str, bit_count: int, hash_count: int):
    digest = hashlib.blake2b(
        value.encode("utf-8"), digest_size=16, person=b"cmsuffix"
    ).digest()
    first, second = struct.unpack("<QQ", digest)
    second |= 1
    mask = bit_count - 1
    for index in range(hash_count):
        yield (first + index * second) & mask


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_suffix_bloom(
    verb_files: list[Path],
    output: Path,
    report_path: Path,
    *,
    bit_count: int = DEFAULT_BITS,
    hash_count: int = DEFAULT_HASHES,
) -> dict:
    if bit_count <= 0 or bit_count & (bit_count - 1):
        raise ValueError("bit_count must be a positive power of two")
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
    bits = bytearray(bit_count // 8)
    generated_count = 0
    started = time.perf_counter()

    for records in verb_index.by_word.values():
        for record in records:
            for spec in ALL_SUFFIXES:
                for candidate in rules.generate_for_record_and_spec(record, spec):
                    value = adapter._normalize_word(candidate.surface)
                    if not value:
                        continue
                    for position in bloom_positions(
                        value, bit_count, hash_count
                    ):
                        bits[position >> 3] |= 1 << (position & 7)
                    generated_count += 1

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        handle.write(
            HEADER.pack(
                MAGIC,
                bit_count,
                hash_count,
                generated_count,
            )
        )
        handle.write(bits)

    report = {
        "format_version": 1,
        "index_path": str(output.resolve()),
        "index_sha256": sha256(output),
        "index_bytes": output.stat().st_size,
        "bit_count": bit_count,
        "hash_count": hash_count,
        "generated_insertions": generated_count,
        "verb_records": verb_index.record_count(),
        "suffix_specs": len(ALL_SUFFIXES),
        "build_seconds": round(time.perf_counter() - started, 3),
        "sources": [
            {
                "path": str(path.resolve()),
                "sha256": sha256(path),
            }
            for path in verb_files
        ],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
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
            "neural_corrector/data/indexes/maltese_suffix_forms.bloom"
        ),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(
            "neural_corrector/data/reports/suffix_bloom_report.json"
        ),
    )
    parser.add_argument("--bits", type=int, default=DEFAULT_BITS)
    parser.add_argument("--hashes", type=int, default=DEFAULT_HASHES)
    args = parser.parse_args()
    verb_files = args.verb_files or [
        Path("Essentials/finaldics/verbmt_semitic.dic"),
        Path("Essentials/finaldics/verbmt_nonsemitic.dic"),
    ]
    report = build_suffix_bloom(
        verb_files,
        args.output,
        args.report,
        bit_count=args.bits,
        hash_count=args.hashes,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
