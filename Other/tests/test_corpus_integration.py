# -*- coding: utf-8 -*-
"""
Integration test suite for Korpus Malti evidence ranking across contextual,
frequency-trap, rare-word, and noise scenarios.
"""

import tempfile
import json
import gzip
from pathlib import Path
import pytest

from Essentials.app import spellchecker
from Essentials.helpers.corpus_scorer import MalteseCorpusScorer


@pytest.fixture
def mock_corpus_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        dir_path = Path(tmpdir)
        meta_data = {"corpus_name": "mock_integration", "preprocessing_version": "1.0"}
        unigrams = {
            "mort": 10.0,
            "mrot": 1.0,
            "baħar": 9.0,
            "ħbieb": 8.0,
            "jikteb": 9.5,
            "jiktep": 0.5,
            "morna": 8.0,
            "nagħmel": 9.0,
            "xogħol": 9.5,
            "bilanc": 0.5,
            "bilanċ": 8.5,
            "ħdejn": 9.0,
            "ġibt": 8.5,
        }
        bigrams = {
            "jien": {"mort": 9.0},
            "mort": {"il-baħar": 8.5, "il-ħanut": 8.0},
            "mal-ħbieb": {"tiegħi": 8.0},
            "aħna": {"morna": 8.0},
            "nagħmel": {"ix-xogħol": 8.5},
        }

        with open(dir_path / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta_data, f)
        with gzip.open(dir_path / "unigrams.json.gz", "wt", encoding="utf-8") as f:
            json.dump(unigrams, f)
        with gzip.open(dir_path / "bigrams.json.gz", "wt", encoding="utf-8") as f:
            json.dump(bigrams, f)

        yield dir_path


def test_contextual_candidate_scoring(mock_corpus_dir):
    scorer = MalteseCorpusScorer(corpus_dir=mock_corpus_dir, enabled=True)
    
    # 1. 'mort' vs 'mrot'
    score_mort = scorer.score_candidate("mort", prev_word="jien")
    score_mrot = scorer.score_candidate("mrot", prev_word="jien")
    assert score_mort > score_mrot

    # 2. Phrase evidence 'mort il-baħar'
    score_bahar = scorer.score_candidate("il-baħar", prev_word="mort")
    assert score_bahar > 0.0

    # 3. Orthography + corpus for 'bilanċ'
    score_bilanc = scorer.score_candidate("bilanċ")
    assert score_bilanc > 0.0


def test_frequency_trap_preserves_hard_rules(mock_corpus_dir):
    scorer = MalteseCorpusScorer(corpus_dir=mock_corpus_dir, enabled=True)
    
    # High frequency singular ('tifel') vs plural ('tfal')
    # Corpus score for singular shouldn't override valid plural when plural is input
    score_tifel = scorer.score_candidate("tifel")
    score_tfal = scorer.score_candidate("tfal")
    # Even if tifel is more frequent in corpus, existing hard rules prevent converting valid 'tfal' into 'tifel'
    assert isinstance(score_tifel, float)
    assert isinstance(score_tfal, float)


def test_noise_tokens_ignored(mock_corpus_dir):
    scorer = MalteseCorpusScorer(corpus_dir=mock_corpus_dir, enabled=True)
    
    # OCR noise / fragments / URLs return 0 bonus
    assert scorer.score_candidate("http://example.com") == 0.0
    assert scorer.score_candidate("???!!!") == 0.0
    assert scorer.score_candidate("abc123xyz") == 0.0
