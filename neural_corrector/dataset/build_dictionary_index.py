from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
import unicodedata
from pathlib import Path

TAG_BITS = {
    "noun": 1 << 0,
    "verb": 1 << 1,
    "adjective": 1 << 2,
    "adverb": 1 << 3,
    "pronoun": 1 << 4,
    "article": 1 << 5,
    "preposition": 1 << 6,
    "participle": 1 << 7,
    "name": 1 << 8,
    "place": 1 << 9,
    "usage": 1 << 10,
    "other": 1 << 11,
    "plural_noun": 1 << 12,
    "cardinal_number": 1 << 13,
    "short_attributive_number": 1 << 14,
    "long_attributive_number": 1 << 15,
}

SOURCE_TAGS = {
    "fixednouns.dic": "noun",
    "nopossessionnouns.dic": "noun",
    "verbmt_semitic.dic": "verb",
    "verbmt_nonsemitic.dic": "verb",
    "maltese_adjectives.dic": "adjective",
    "maltese_adverbs.dic": "adverb",
    "maltese_pronouns.dic": "pronoun",
    "maltese_articles.dic": "article",
    "prepositions.dic": "preposition",
    "participles.dic": "participle",
    "names.dic": "name",
    "places.dic": "place",
    "eu_countries.dic": "place",
    "Maltese_usage.dic": "usage",
    "dev_extra.dic": "other",
}


def normalize_key(text: str) -> str:
    return " ".join(
        unicodedata.normalize("NFC", text)
        .replace("’", "'")
        .replace("ʼ", "'")
        .casefold()
        .split()
    )


def fuzzy_key(text: str) -> str:
    folded = normalize_key(text).translate(
        str.maketrans({"ċ": "c", "ġ": "g", "ħ": "h", "ż": "z"})
    )
    return "".join(character for character in folded if character.isalpha())


def descriptor_bits(descriptor: str, fallback: str) -> int:
    upper = descriptor.upper()
    bits = TAG_BITS[fallback]
    if "NOUN" in upper:
        bits |= TAG_BITS["noun"]
    if "PLUNOUN" in upper or "COLLNOUN" in upper:
        bits |= TAG_BITS["plural_noun"]
    if "/T-" in f"/{upper}" or upper.startswith("T-"):
        bits |= TAG_BITS["verb"]
    if "ADJ" in upper:
        bits |= TAG_BITS["adjective"]
    if "ADVERB" in upper:
        bits |= TAG_BITS["adverb"]
    if "PRON" in upper:
        bits |= TAG_BITS["pronoun"]
    if "ART" in upper:
        bits |= TAG_BITS["article"]
    if "PREP" in upper:
        bits |= TAG_BITS["preposition"]
    if "PART" in upper:
        bits |= TAG_BITS["participle"]
    if "NAME" in upper or "SNAME" in upper or "PROPN" in upper:
        bits |= TAG_BITS["name"]
    if "PLACE" in upper or "DNYM" in upper:
        bits |= TAG_BITS["place"]
    if "MALTI" in upper:
        bits |= TAG_BITS["usage"]
    if "CARDNUM" in upper:
        bits |= TAG_BITS["cardinal_number"]
    if "SHORTATTNUM" in upper:
        bits |= TAG_BITS["short_attributive_number"]
    if "LONGATTNUM" in upper:
        bits |= TAG_BITS["long_attributive_number"]
    return bits


def dictionary_rows(path: Path, fallback_tag: str):
    with path.open(encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#") or line.isdigit():
                continue
            surface, separator, descriptor = line.partition("/")
            surface = unicodedata.normalize("NFC", surface.strip())
            if not surface:
                continue
            yield (
                normalize_key(surface),
                surface,
                descriptor_bits(descriptor if separator else "", fallback_tag),
                line_number,
            )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_index(dictionary_dir: Path, output: Path, report_path: Path) -> dict:
    sources = [
        path
        for path in sorted(dictionary_dir.glob("*.dic"))
        if path.name in SOURCE_TAGS
    ]
    if not sources:
        raise ValueError(f"No supported dictionaries found in {dictionary_dir}")

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()
    connection = sqlite3.connect(output)
    try:
        connection.executescript(
            """
            PRAGMA journal_mode=OFF;
            PRAGMA synchronous=OFF;
            PRAGMA temp_store=MEMORY;
            CREATE TABLE words (
                word_key TEXT PRIMARY KEY,
                canonical TEXT NOT NULL,
                tag_bits INTEGER NOT NULL,
                source_bits INTEGER NOT NULL,
                fuzzy_key TEXT NOT NULL,
                fuzzy_head TEXT NOT NULL,
                fuzzy_tail TEXT NOT NULL,
                fuzzy_length INTEGER NOT NULL
            ) WITHOUT ROWID;
            CREATE INDEX words_fuzzy_head_length
                ON words(fuzzy_head, fuzzy_length);
            CREATE INDEX words_fuzzy_tail_length
                ON words(fuzzy_tail, fuzzy_length);
            CREATE TABLE metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            ) WITHOUT ROWID;
            """
        )
        source_reports = []
        parsed_rows = 0
        with connection:
            for source_index, path in enumerate(sources):
                source_bit = 1 << source_index
                source_rows = 0
                for word_key, surface, tag_bits, _ in dictionary_rows(
                    path, SOURCE_TAGS[path.name]
                ):
                    search_key = fuzzy_key(surface)
                    connection.execute(
                        """
                        INSERT INTO words(
                            word_key, canonical, tag_bits, source_bits,
                            fuzzy_key, fuzzy_head, fuzzy_tail, fuzzy_length
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(word_key) DO UPDATE SET
                            tag_bits = words.tag_bits | excluded.tag_bits,
                            source_bits = words.source_bits | excluded.source_bits
                        """,
                        (
                            word_key,
                            surface,
                            tag_bits,
                            source_bit,
                            search_key,
                            search_key[:1],
                            search_key[-2:],
                            len(search_key),
                        ),
                    )
                    source_rows += 1
                parsed_rows += source_rows
                source_reports.append(
                    {
                        "name": path.name,
                        "rows": source_rows,
                        "sha256": sha256(path),
                        "tag": SOURCE_TAGS[path.name],
                        "source_bit": source_bit,
                    }
                )
            unique_words = connection.execute(
                "SELECT COUNT(*) FROM words"
            ).fetchone()[0]
            metadata = {
                "format_version": "2",
                "normalization": "NFC + apostrophe folding + casefold + whitespace",
                "source_count": str(len(sources)),
                "parsed_rows": str(parsed_rows),
                "unique_words": str(unique_words),
            }
            connection.executemany(
                "INSERT INTO metadata(key, value) VALUES (?, ?)",
                metadata.items(),
            )
        connection.execute("VACUUM")
    finally:
        connection.close()

    report = {
        "format_version": 2,
        "dictionary_directory": str(dictionary_dir.resolve()),
        "index_path": str(output.resolve()),
        "index_sha256": sha256(output),
        "index_bytes": output.stat().st_size,
        "parsed_rows": parsed_rows,
        "unique_words": unique_words,
        "tag_bits": TAG_BITS,
        "sources": source_reports,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dictionary-dir",
        type=Path,
        default=Path("Essentials/finaldics"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "neural_corrector/data/indexes/maltese_dictionary.sqlite3"
        ),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(
            "neural_corrector/data/reports/dictionary_index_report.json"
        ),
    )
    args = parser.parse_args()
    print(
        json.dumps(
            build_index(args.dictionary_dir, args.output, args.report),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
