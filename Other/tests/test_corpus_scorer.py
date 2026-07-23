# -*- coding: utf-8 -*-
"""
Unit tests for MalteseCorpusScorer.
"""

import gzip
import io
import json
import tempfile
import tarfile
import zipfile
from pathlib import Path
import pytest

from Essentials.helpers.corpus_scorer import MalteseCorpusScorer
from Essentials.helpers.candidate_evidence import CandidateEvidence, CandidateEvidencePool
from tools.setup_korpus_malti import parse_vertical_row, process_vertical_or_text_files, _extract_archive


def test_missing_index_fails_gracefully():
    with tempfile.TemporaryDirectory() as tmpdir:
        scorer = MalteseCorpusScorer(corpus_dir=Path(tmpdir), enabled=True)
        assert not scorer.is_available()
        assert scorer.score_candidate("ħobż") == 0.0


def test_mock_index_scoring():
    with tempfile.TemporaryDirectory() as tmpdir:
        dir_path = Path(tmpdir)
        meta_data = {"corpus_name": "mock", "preprocessing_version": "1.0"}
        unigrams = {"ħobż": 10.0, "il-ħobż": 8.0, "dar": 9.0}
        bigrams = {"nixtri": {"l-ħobż": 7.0, "ħobż": 8.5}}
        trigrams = {"jien nixtri": {"ħobż": 6.0}}

        with open(dir_path / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta_data, f)
        with gzip.open(dir_path / "unigrams.json.gz", "wt", encoding="utf-8") as f:
            json.dump(unigrams, f)
        with gzip.open(dir_path / "bigrams.json.gz", "wt", encoding="utf-8") as f:
            json.dump(bigrams, f)
        with gzip.open(dir_path / "trigrams.json.gz", "wt", encoding="utf-8") as f:
            json.dump(trigrams, f)

        scorer = MalteseCorpusScorer(
            corpus_dir=dir_path,
            enabled=True,
            unigram_enabled=True,
            bigram_enabled=True,
            trigram_enabled=True,
            max_score_contribution=0.25,
        )

        assert scorer.is_available()

        # Unigram score
        u_score = scorer.score_candidate("ħobż")
        assert u_score > 0.0
        assert u_score <= 0.25

        # Bigram score (nixtri -> ħobż)
        bi_score = scorer.score_candidate("ħobż", prev_word="nixtri")
        assert bi_score > u_score
        assert bi_score <= 0.25

        # Unattested word returns 0.0
        assert scorer.score_candidate("xyzqwerty") == 0.0


def test_shadow_mode_returns_zero():
    with tempfile.TemporaryDirectory() as tmpdir:
        dir_path = Path(tmpdir)
        with open(dir_path / "meta.json", "w", encoding="utf-8") as f:
            json.dump({}, f)
        with gzip.open(dir_path / "unigrams.json.gz", "wt", encoding="utf-8") as f:
            json.dump({"ħobż": 10.0}, f)
        with gzip.open(dir_path / "bigrams.json.gz", "wt", encoding="utf-8") as f:
            json.dump({}, f)

        scorer = MalteseCorpusScorer(
            corpus_dir=dir_path,
            enabled=True,
            shadow_mode=True,
        )

        assert scorer.is_available()
        assert scorer.score_candidate("ħobż") == 0.0


def test_status_reports_disabled_config():
    with tempfile.TemporaryDirectory() as tmpdir:
        scorer = MalteseCorpusScorer(corpus_dir=Path(tmpdir), enabled=False)
        assert scorer.status == "CORPUS_SCORER_DISABLED_BY_CONFIG"
        assert scorer.status_reason


def test_status_reports_missing_indexes():
    with tempfile.TemporaryDirectory() as tmpdir:
        scorer = MalteseCorpusScorer(corpus_dir=Path(tmpdir), enabled=True)
        assert scorer.status == "CORPUS_SCORER_MISSING_INDEXES"
        assert scorer.status_reason


def test_candidate_evidence_pool_annotation():
    with tempfile.TemporaryDirectory() as tmpdir:
        dir_path = Path(tmpdir)
        with open(dir_path / "meta.json", "w", encoding="utf-8") as f:
            json.dump({}, f)
        with gzip.open(dir_path / "unigrams.json.gz", "wt", encoding="utf-8") as f:
            json.dump({"mort": 8.0, "mrot": 1.0}, f)
        with gzip.open(dir_path / "bigrams.json.gz", "wt", encoding="utf-8") as f:
            json.dump({"jien": {"mort": 7.0}}, f)

        scorer = MalteseCorpusScorer(corpus_dir=dir_path, enabled=True)
        pool = CandidateEvidencePool(normalizer=lambda w: w.lower())
        pool.add("mort", "anchor")
        pool.add("mrot", "anchor")

        pool.annotate_corpus(scorer, prev_word="jien")

        by_word = pool.by_word()
        assert "corpus" in by_word["mort"].sources
        assert by_word["mort"].corpus_score > by_word["mrot"].corpus_score


def test_parse_vertical_row_handles_maltese_unicode_and_whitespace():
    parsed = parse_vertical_row("  ħobż | NOUN | ħobż | ħobż  ")
    assert parsed is not None
    assert parsed[0] == "ħobż"


def test_parse_vertical_row_rejects_malformed_and_noise_rows():
    assert parse_vertical_row("") is None
    assert parse_vertical_row("http://example.com") is None
    assert parse_vertical_row("!!!") is None


def test_process_vertical_or_text_files_skips_cross_boundary_bigrams_and_preserves_unicode(tmp_path):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    sample = "\n".join([
        "Ħobż | NOUN | ħobż | ħobż",
        "</s>",
        "ta' | PRP | ta' | ta'",
        "il- | PRP | il- | il-",
        "</s>",
        "jien | PRON | jien | jien",
        "nixtri | VERB | nixtri | nixtri",
        "ħobż | NOUN | ħobż | ħobż",
    ])
    (corpus_dir / "sample.vert").write_text(sample, encoding="utf-8")
    processed = process_vertical_or_text_files(corpus_dir, min_freq=1)
    assert processed["stats"]["valid_rows"] >= 3
    assert ("ħobż", "ta'") not in processed["bigrams"]
    assert processed["stats"]["malformed_rows"] == 0


def test_extract_archive_blocks_path_traversal(tmp_path):
    archive_path = tmp_path / "bad.tar.gz"
    with tarfile.open(archive_path, "w:gz") as tar:
        data = b"bad"
        info = tarfile.TarInfo("../escape.txt")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    destination = tmp_path / "out"
    extracted = _extract_archive(archive_path, destination)
    assert not (destination / "escape.txt").exists()
    assert extracted == []
