# -*- coding: utf-8 -*-
"""
Setup script for downloading and preprocessing Korpus Malti v4.0.

Usage:
    python tools/setup_korpus_malti.py [--force] [--token HF_TOKEN] [--out-dir DIR]

Requirements:
    Requires huggingface_hub to download MLRS/korpus_malti.
    Must be run as an explicit offline step before launching the corpus-enabled spellchecker.
"""

import argparse
import gzip
import hashlib
import json
import logging
import math
import os
import re
import shutil
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("setup_korpus_malti")

REPO_ID = "MLRS/korpus_malti"
DEFAULT_REVISION = "main"
PREPROCESSING_VERSION = "1.1.0"
DEFAULT_MLRS_URL = "https://mlrs.research.um.edu.mt/CQPweb/malti04/"
DEFAULT_MLRS_SECTION = "wiki"

# Target output directory
BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = BASE_DIR / "Essentials" / "corpus"
RAW_CACHE_DIR = BASE_DIR / ".corpus_cache"

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MARKUP_PATTERN = re.compile(r"<[^>]+>")
PUNCT_PATTERN = re.compile(r"^[^\w\s]+$", re.UNICODE)


def calculate_file_checksum(filepath: Path) -> str:
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    return sha256.hexdigest()


def normalize_token(text: str) -> str:
    return text.strip().lower()


def is_valid_token(token: str) -> bool:
    if not token or len(token) > 80:
        return False
    if URL_PATTERN.search(token) or MARKUP_PATTERN.search(token):
        return False
    if PUNCT_PATTERN.match(token):
        return False
    if any(char.isdigit() for char in token):
        # Allow numbers only if mixed or pure numeric tokens are ignored
        return False
    return True


def download_with_retries(url: str, dest_path: Path, timeout: int = 30, retries: int = 3) -> Path:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(dest_path.suffix + ".part")
    if dest_path.exists() and dest_path.is_file() and not temp_path.exists():
        logger.info(f"Using existing archive: {dest_path}")
        return dest_path

    for attempt in range(1, retries + 1):
        try:
            logger.info(f"Downloading {url} (attempt {attempt}/{retries})")
            with urllib.request.urlopen(url, timeout=timeout) as response, open(temp_path, "wb") as fh:
                if response.getcode() >= 400:
                    raise RuntimeError(f"HTTP {response.getcode()}")
                shutil.copyfileobj(response, fh, length=1024 * 1024)
            os.replace(temp_path, dest_path)
            logger.info(f"Archive saved to {dest_path}")
            return dest_path
        except Exception as exc:
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)
            if attempt < retries:
                logger.warning(f"Download attempt {attempt} failed: {exc}. Retrying...")
                continue
            raise RuntimeError(f"Failed to download {url}: {exc}") from exc

    raise RuntimeError(f"Failed to download {url}")


def download_corpus(token: str | None = None, revision: str = DEFAULT_REVISION, source: str = "auto", section: str = DEFAULT_MLRS_SECTION, url: Optional[str] = None) -> Path:
    RAW_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if source in {"mlrs", "auto"}:
        mlrs_url = url or DEFAULT_MLRS_URL
        archive_path = RAW_CACHE_DIR / "korpus_malti_mlrs.tar.gz"
        try:
            archive_path = download_with_retries(mlrs_url, archive_path)
            logger.info(f"MLRS archive downloaded to {archive_path}")
            return archive_path
        except Exception as exc:
            logger.warning(f"MLRS direct download failed: {exc}")
            if source == "mlrs":
                raise

    if source in {"huggingface", "auto"}:
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            logger.error("`huggingface_hub` is not installed for Hugging Face fallback.")
            if source == "huggingface":
                raise
        hf_token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        logger.info(f"Downloading/verifying dataset {REPO_ID} (revision: {revision})...")
        try:
            local_dir = snapshot_download(
                repo_id=REPO_ID,
                repo_type="dataset",
                revision=revision,
                token=hf_token,
                local_dir=str(RAW_CACHE_DIR / "korpus_malti_hf"),
            )
            logger.info(f"Corpus raw files ready at: {local_dir}")
            return Path(local_dir)
        except Exception as exc:
            logger.warning(f"Hugging Face fallback failed: {exc}")
            if source == "huggingface":
                raise

    if source == "local":
        raise FileNotFoundError("Local source requested but no local corpus path was provided.")

    raise RuntimeError("No corpus source could be downloaded or located.")


def _extract_archive(archive_path: Path, dest_dir: Path) -> List[Path]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(dest_dir)
            return [Path(p) for p in zf.namelist() if not p.endswith("/")]
    if tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path, "r:*") as tf:
            safe_members = []
            for member in tf.getmembers():
                if member.name.startswith("/") or ".." in Path(member.name).parts:
                    continue
                safe_members.append(member)
            tf.extractall(dest_dir, members=safe_members)
            return [Path(member.name) for member in safe_members if not member.isdir()]
    raise RuntimeError(f"Unsupported archive format: {archive_path}")


def discover_corpus_files(corpus_dir: Path) -> List[Path]:
    if not corpus_dir.exists():
        return []
    candidates = []
    for path in sorted(corpus_dir.rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue
        lower = path.suffix.lower()
        if lower in {".txt", ".vert", ".tsv", ".conllu", ".csv", ".xml", ".gz", ".bz2", ".xz"}:
            candidates.append(path)
    return candidates


def parse_vertical_row(line: str) -> Optional[Tuple[str, str]]:
    if not line or line.startswith("#"):
        return None
    # Korpus Malti v4 vertical files are tab-separated. Older fixtures and
    # exports use pipes, so accept both without indexing a complete annotated
    # row as one surface token.
    delimiter = "\t" if "\t" in line else "|"
    parts = [part.strip() for part in line.split(delimiter)]
    if not parts:
        return None
    token_surface = parts[0]
    if not token_surface:
        return None
    if URL_PATTERN.search(token_surface) or MARKUP_PATTERN.search(token_surface):
        return None
    if PUNCT_PATTERN.match(token_surface):
        return None
    if any(char.isdigit() for char in token_surface):
        return None
    return token_surface, parts[0]


def process_vertical_or_text_files(corpus_dir: Path, min_freq: int = 2) -> Dict:
    logger.info("Preprocessing Korpus Malti text / vertical files...")

    unigram_counts: Dict[str, int] = {}
    bigram_counts: Dict[Tuple[str, str], int] = {}
    trigram_counts: Dict[Tuple[str, str, str], int] = {}
    capitalization_counts: Dict[str, Dict[str, int]] = {}

    total_tokens = 0
    total_sentences = 0
    malformed_rows = 0
    valid_rows = 0

    corpus_files = discover_corpus_files(corpus_dir)
    if not corpus_files:
        logger.warning(f"No supported corpus files found under {corpus_dir}")
        return {
            "unigrams": {},
            "bigrams": {},
            "trigrams": {},
            "capitalization": {},
            "stats": {
                "total_tokens": 0,
                "total_sentences": 0,
                "vocab_size": 0,
                "bigram_count": 0,
                "trigram_count": 0,
                "malformed_rows": 0,
                "valid_rows": 0,
            },
        }

    logger.info(f"Found {len(corpus_files)} files in corpus directory.")

    for filepath in corpus_files:
        logger.info(f"Processing {filepath.relative_to(corpus_dir)}...")
        try:
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                prev_w1 = None
                prev_w2 = None
                for line in f:
                    line_str = line.strip()
                    if not line_str or line_str.startswith("#") or line_str.startswith("<"):
                        if not line_str or line_str.startswith("</s>") or line_str.startswith("</s"):
                            total_sentences += 1
                        prev_w1 = None
                        prev_w2 = None
                        continue

                    parsed = parse_vertical_row(line_str)
                    if parsed is None:
                        malformed_rows += 1
                        continue

                    token_surface, _ = parsed
                    if not is_valid_token(token_surface):
                        continue

                    norm = normalize_token(token_surface)
                    if not norm:
                        continue

                    valid_rows += 1
                    total_tokens += 1
                    unigram_counts[norm] = unigram_counts.get(norm, 0) + 1

                    if norm not in capitalization_counts:
                        capitalization_counts[norm] = {}
                    capitalization_counts[norm][token_surface] = capitalization_counts[norm].get(token_surface, 0) + 1

                    if prev_w1 is not None:
                        bg = (prev_w1, norm)
                        bigram_counts[bg] = bigram_counts.get(bg, 0) + 1

                        if prev_w2 is not None:
                            tg = (prev_w2, prev_w1, norm)
                            trigram_counts[tg] = trigram_counts.get(tg, 0) + 1

                    prev_w2 = prev_w1
                    prev_w1 = norm

        except Exception as exc:
            logger.warning(f"Error reading {filepath}: {exc}")

    logger.info(f"Raw stats: {total_tokens} tokens, {len(unigram_counts)} unique unigrams, {len(bigram_counts)} bigrams.")

    filtered_unigrams = {k: v for k, v in unigram_counts.items() if v >= min_freq}
    filtered_bigrams = {k: v for k, v in bigram_counts.items() if v >= min_freq and k[0] in filtered_unigrams and k[1] in filtered_unigrams}
    filtered_trigrams = {k: v for k, v in trigram_counts.items() if v >= min_freq and all(w in filtered_unigrams for w in k)}

    unigram_log_freq = {k: round(math.log(v), 4) for k, v in filtered_unigrams.items()}
    bigram_dict: Dict[str, Dict[str, float]] = {}
    for (w1, w2), cnt in filtered_bigrams.items():
        if w1 not in bigram_dict:
            bigram_dict[w1] = {}
        bigram_dict[w1][w2] = round(math.log(cnt), 4)

    trigram_dict: Dict[str, Dict[str, float]] = {}
    for (w1, w2, w3), cnt in filtered_trigrams.items():
        key = f"{w1} {w2}"
        if key not in trigram_dict:
            trigram_dict[key] = {}
        trigram_dict[key][w3] = round(math.log(cnt), 4)

    pref_cap = {}
    for norm, cap_map in capitalization_counts.items():
        if norm in filtered_unigrams:
            most_freq = max(cap_map.items(), key=lambda x: x[1])[0]
            if most_freq != norm:
                pref_cap[norm] = most_freq

    return {
        "unigrams": unigram_log_freq,
        "bigrams": bigram_dict,
        "trigrams": trigram_dict,
        "capitalization": pref_cap,
        "stats": {
            "total_tokens": total_tokens,
            "total_sentences": total_sentences,
            "vocab_size": len(unigram_log_freq),
            "bigram_count": len(filtered_bigrams),
            "trigram_count": len(filtered_trigrams),
            "malformed_rows": malformed_rows,
            "valid_rows": valid_rows,
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Download and build runtime indexes for Korpus Malti.")
    parser.add_argument("--force", action="store_true", help="Force rebuild even if index files exist.")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face access token.")
    parser.add_argument("--revision", type=str, default=DEFAULT_REVISION, help="Corpus git revision.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Directory to save index files.")
    parser.add_argument("--min-freq", type=int, default=2, help="Minimum frequency threshold.")
    parser.add_argument("--source", choices=["mlrs", "huggingface", "local", "auto"], default="mlrs", help="Corpus source to use.")
    parser.add_argument("--section", choices=["wiki", "all"], default=DEFAULT_MLRS_SECTION, help="MLRS corpus section to select for this build.")
    parser.add_argument("--mlrs-url", type=str, default=DEFAULT_MLRS_URL, help="MLRS direct download URL.")
    parser.add_argument("--local-dir", type=Path, default=None, help="Local archive or extracted corpus directory.")

    args = parser.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_file = out_dir / "meta.json"
    unigrams_file = out_dir / "unigrams.json.gz"
    bigrams_file = out_dir / "bigrams.json.gz"
    trigrams_file = out_dir / "trigrams.json.gz"

    if not args.force and meta_file.exists() and unigrams_file.exists() and bigrams_file.exists():
        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
            logger.info("Found existing valid Korpus Malti index:")
            logger.info(f"  Version: {meta.get('preprocessing_version')}")
            logger.info(f"  Build Date: {meta.get('build_date')}")
            logger.info(f"  Total Tokens: {meta.get('stats', {}).get('total_tokens')}")
            logger.info(f"  Vocab Size: {meta.get('stats', {}).get('vocab_size')}")
            logger.info("Use --force to force download and rebuild.")
            return
        except Exception:
            logger.warning("Existing index metadata corrupted. Rebuilding...")

    # 1. Download raw corpus
    if args.source == "local" and args.local_dir is None:
        raise SystemExit("--source local requires --local-dir")

    if args.source == "local" and args.local_dir is not None:
        local_path = args.local_dir.resolve()
        if local_path.is_file():
            archive_dir = RAW_CACHE_DIR / "local_corpus"
            _extract_archive(local_path, archive_dir)
            corpus_dir = archive_dir
        elif local_path.is_dir():
            corpus_dir = local_path
        else:
            raise SystemExit(f"Local corpus path not found: {local_path}")
    else:
        archive_or_dir = download_corpus(
            token=args.token,
            revision=args.revision,
            source=args.source,
            section=args.section,
            url=args.mlrs_url,
        )
        if archive_or_dir.is_file():
            extracted_dir = RAW_CACHE_DIR / "extracted_corpus"
            _extract_archive(archive_or_dir, extracted_dir)
            corpus_dir = extracted_dir
        else:
            corpus_dir = archive_or_dir

    # 2. Preprocess
    processed = process_vertical_or_text_files(corpus_dir, min_freq=args.min_freq)

    # 3. Save indexes
    logger.info(f"Saving compact gzip JSON indexes to {out_dir}...")

    with gzip.open(unigrams_file, "wt", encoding="utf-8") as f:
        json.dump(processed["unigrams"], f)

    with gzip.open(bigrams_file, "wt", encoding="utf-8") as f:
        json.dump(processed["bigrams"], f)

    with gzip.open(trigrams_file, "wt", encoding="utf-8") as f:
        json.dump(processed["trigrams"], f)

    uni_checksum = calculate_file_checksum(unigrams_file)
    bi_checksum = calculate_file_checksum(bigrams_file)
    tri_checksum = calculate_file_checksum(trigrams_file)

    meta_data = {
        "corpus_name": "Korpus Malti",
        "corpus_source": "MLRS" if args.source in {"mlrs", "auto"} else ("HuggingFace" if args.source == "huggingface" else "local"),
        "corpus_revision": args.revision,
        "corpus_version": "4.2",
        "selected_section": args.section,
        "source_url": args.mlrs_url if args.source in {"mlrs", "auto"} else None,
        "download_timestamp": datetime.now(timezone.utc).isoformat(),
        "preprocessing_version": PREPROCESSING_VERSION,
        "build_date": datetime.now(timezone.utc).isoformat(),
        "index_format_version": "1.0",
        "min_freq": args.min_freq,
        "stats": processed["stats"],
        "checksums": {
            "unigrams.json.gz": uni_checksum,
            "bigrams.json.gz": bi_checksum,
            "trigrams.json.gz": tri_checksum,
        },
        "attribution": "Korpus Malti v4.2 provided by MLRS (Maltese Language Resource Server), University of Malta."
    }

    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta_data, f, indent=2)

    logger.info("=" * 70)
    logger.info("SUCCESS: Korpus Malti runtime indexes built successfully!")
    logger.info(f"Index Location: {out_dir}")
    logger.info(f"Tokens Processed: {processed['stats']['total_tokens']}")
    logger.info(f"Vocabulary Size: {processed['stats']['vocab_size']}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
