from __future__ import annotations

import argparse
import gzip
import json
import math
import re
import subprocess
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


TOKEN_LINE_RE = re.compile(r"^(.+?)\t([^\t]+)\t([^\t]*)\t([^\t]*)$")


def normalize(value: str) -> str:
    return unicodedata.normalize("NFC", value or "").casefold().strip()


def log_count(value: int) -> float:
    return round(math.log1p(value), 4)


def list_archive_members(seven_zip: Path, archive: Path) -> list[str]:
    command = [str(seven_zip), "l", "-slt", str(archive)]
    result = subprocess.run(command, check=True, capture_output=True, text=True, encoding="utf-8", errors="replace")
    members: list[str] = []
    for line in result.stdout.splitlines():
        if not line.startswith("Path = "):
            continue
        member = line.removeprefix("Path = ").strip()
        if member.endswith(".txt"):
            members.append(member)
    return members


def iter_member_lines(seven_zip: Path, archive: Path, member: str):
    command = [str(seven_zip), "e", "-so", str(archive), member]
    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1024 * 1024,
    )
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            yield line.rstrip("\n\r")
    finally:
        proc.stdout.close()
        code = proc.wait()
        if code not in (0,):
            raise RuntimeError(f"7-Zip failed while reading {member}: exit code {code}")


def flush_sentence(
    sentence: list[str],
    bigrams: dict[str, Counter[str]],
    sentence_counter: list[int],
) -> None:
    if not sentence:
        return
    sentence_counter[0] += 1
    for left, right in zip(sentence, sentence[1:]):
        bigrams[left][right] += 1
    sentence.clear()


def build(
    *,
    archive: Path,
    output_dir: Path,
    seven_zip: Path,
    min_freq: int = 1,
    min_bigram_freq: int = 1,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    members = list_archive_members(seven_zip, archive)
    if not members:
        raise RuntimeError(f"No .txt members found in {archive}")

    unigrams: Counter[str] = Counter()
    tagged_unigrams: Counter[str] = Counter()
    bigrams: dict[str, Counter[str]] = defaultdict(Counter)
    malformed_rows = 0
    valid_rows = 0
    sentence_counter = [0]

    for member_index, member in enumerate(members, 1):
        sentence: list[str] = []
        for line in iter_member_lines(seven_zip, archive, member):
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("</s"):
                flush_sentence(sentence, bigrams, sentence_counter)
                continue
            if stripped.startswith("<"):
                continue

            match = TOKEN_LINE_RE.match(stripped)
            if not match:
                malformed_rows += 1
                continue

            surface, pos, lemma, root = (normalize(part) for part in match.groups())
            if not surface or not pos:
                malformed_rows += 1
                continue

            unigrams[surface] += 1
            tagged_unigrams[f"{surface}\t{pos}\t{lemma or 'null'}\t{root or 'null'}"] += 1
            sentence.append(surface)
            valid_rows += 1

        flush_sentence(sentence, bigrams, sentence_counter)
        print(
            json.dumps(
                {
                    "member": member,
                    "index": member_index,
                    "of": len(members),
                    "tokens": valid_rows,
                    "vocab": len(unigrams),
                    "bigram_roots": len(bigrams),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    filtered_unigrams = {
        word: log_count(count)
        for word, count in sorted(unigrams.items())
        if count >= min_freq
    }
    allowed_words = set(filtered_unigrams)
    filtered_bigrams = {
        left: {
            right: log_count(count)
            for right, count in sorted(row.items())
            if count >= min_bigram_freq and right in allowed_words
        }
        for left, row in sorted(bigrams.items())
        if left in allowed_words
    }
    filtered_bigrams = {left: row for left, row in filtered_bigrams.items() if row}
    filtered_tagged = {
        key: log_count(count)
        for key, count in sorted(tagged_unigrams.items())
        if key.split("\t", 1)[0] in allowed_words
    }

    morphology: dict[str, dict[tuple[str, str | None, str | None], float]] = defaultdict(dict)
    for key, frequency in filtered_tagged.items():
        parts = key.split("\t")
        if len(parts) < 2:
            continue
        surface = parts[0]
        pos = parts[1]
        lemma = None if len(parts) < 3 or parts[2] in {"", "null"} else parts[2]
        root = None if len(parts) < 4 or parts[3] in {"", "null"} else parts[3]
        morphology[surface][(pos, lemma, root)] = max(
            morphology[surface].get((pos, lemma, root), 0.0),
            float(frequency),
        )
    morphology_payload = {
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
        for surface, analyses in sorted(morphology.items())
    }

    with gzip.open(output_dir / "unigrams.json.gz", "wt", encoding="utf-8") as stream:
        json.dump(filtered_unigrams, stream, ensure_ascii=False, separators=(",", ":"))
    with gzip.open(output_dir / "bigrams.json.gz", "wt", encoding="utf-8") as stream:
        json.dump(filtered_bigrams, stream, ensure_ascii=False, separators=(",", ":"))
    with gzip.open(output_dir / "tagged_unigrams.json.gz", "wt", encoding="utf-8") as stream:
        json.dump(filtered_tagged, stream, ensure_ascii=False, separators=(",", ":"))
    with gzip.open(output_dir / "morphology.json.gz", "wt", encoding="utf-8") as stream:
        json.dump(morphology_payload, stream, ensure_ascii=False, separators=(",", ":"))

    meta = {
        "corpus_name": "Korpus Malti",
        "corpus_source": "MLRS",
        "corpus_revision": "full-local-archive-rebuild",
        "corpus_version": "4.2",
        "selected_section": "All Sections",
        "source_url": archive.resolve().as_uri(),
        "build_date": datetime.now(timezone.utc).isoformat(),
        "index_format_version": "2.0",
        "preprocessing_version": "hope-full-corpus-rebuild-1.0",
        "min_freq": min_freq,
        "min_bigram_freq": min_bigram_freq,
        "stats": {
            "total_tokens": valid_rows,
            "total_sentences": sentence_counter[0],
            "vocab_size": len(filtered_unigrams),
            "tagged_unigram_count": len(filtered_tagged),
            "bigram_count": sum(len(row) for row in filtered_bigrams.values()),
            "bigram_roots": len(filtered_bigrams),
            "morphology_surfaces": len(morphology_payload),
            "morphology_analyses": sum(len(rows) for rows in morphology_payload.values()),
            "malformed_rows": malformed_rows,
            "valid_rows": valid_rows,
            "archive_members": len(members),
        },
        "attribution": "Korpus Malti v4.2 provided by MLRS (Maltese Language Resource Server), University of Malta.",
    }
    (output_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=Path(r"C:\Users\diyan\Downloads\All.zip"))
    parser.add_argument("--output", type=Path, default=Path("corpus"))
    parser.add_argument("--seven-zip", type=Path, default=Path(r"C:\Program Files\7-Zip\7z.exe"))
    parser.add_argument("--min-freq", type=int, default=1)
    parser.add_argument("--min-bigram-freq", type=int, default=1)
    args = parser.parse_args()
    print(
        json.dumps(
            build(
                archive=args.archive,
                output_dir=args.output,
                seven_zip=args.seven_zip,
                min_freq=args.min_freq,
                min_bigram_freq=args.min_bigram_freq,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )
