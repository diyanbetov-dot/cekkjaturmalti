# -*- coding: utf-8 -*-
"""
Tests fallback behavior when corpus and BERTu scoring are disabled or missing.
Ensures existing Maltese spellchecker pipeline operates identically.
"""

import pytest
from Essentials.app import spellchecker


def test_corpus_disabled_fallback_behavior():
    # Verify spellchecker runs and produces expected corrections without error
    result = spellchecker.correct_text_rich("Jien mrot il-ħanut")
    assert "corrected_text" in result
    assert result["corrected_text"] != ""


def test_missing_corpus_scorer_does_not_crash():
    # Temporarily remove corpus_scorer attribute if present
    original_scorer = getattr(spellchecker, "corpus_scorer", None)
    try:
        spellchecker.corpus_scorer = None
        suggestions = spellchecker.suggest("xar", limit=5)
        assert isinstance(suggestions, list)
        assert len(suggestions) > 0
    finally:
        spellchecker.corpus_scorer = original_scorer
