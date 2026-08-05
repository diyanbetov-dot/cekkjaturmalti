from __future__ import annotations

import unicodedata

import pytest

from contextual_corrector.schema import TokenKind
from contextual_corrector.text import (
    boundary_span,
    grapheme_edit_distance,
    normalize_for_lattice,
    span_for_token_range,
    tokenize_lattice,
)


def test_nfc_internal_text_and_ui_offsets_round_trip() -> None:
    composed = "Ċensu ħareġ"
    original = unicodedata.normalize("NFD", composed)
    text = normalize_for_lattice(original)

    assert text.original == original
    assert text.normalized == composed
    tokens = tokenize_lattice(text)
    hareg = tokens[1]
    original_start, original_end = text.original_span(hareg.char_start, hareg.char_end)

    assert unicodedata.normalize("NFC", original[original_start:original_end]) == hareg.text
    assert text.normalized_span(original_start, original_end) == (
        hareg.char_start,
        hareg.char_end,
    )


def test_only_nfc_is_normalized_before_candidate_generation() -> None:
    original = "x'għandek x’għandek il-lejla \"test\""
    text = normalize_for_lattice(original)

    assert text.normalized == original
    assert "'" in text.normalized
    assert "’" in text.normalized
    assert "-" in text.normalized
    tokens = tokenize_lattice(text)
    punctuation = [token.text for token in tokens if token.kind == TokenKind.PUNCTUATION]
    assert "'" in punctuation
    assert "’" in punctuation
    assert "-" in punctuation
    assert '"' in punctuation


def test_grapheme_distance_is_separate_from_codepoint_offsets() -> None:
    decomposed = unicodedata.normalize("NFD", "ċ")

    assert len(decomposed) > 1
    assert grapheme_edit_distance(decomposed, "ċ") == 0
    assert grapheme_edit_distance("c", "ċ") == 1


def test_non_boundary_spans_round_trip_against_normalized_text() -> None:
    text = normalize_for_lattice("ma   hawnx")
    tokens = tokenize_lattice(text)
    span = span_for_token_range(text.normalized, tokens, 0, 2)

    assert span.text == "ma   hawnx"
    assert text.normalized[span.char_start : span.char_end] == span.text


def test_boundary_span_is_zero_length_and_sits_after_previous_token() -> None:
    text = normalize_for_lattice("kelma   oħra")
    tokens = tokenize_lattice(text)
    boundary = boundary_span(text.normalized, tokens, 1)

    assert boundary.char_start == boundary.char_end == len("kelma")
    assert boundary.token_start == boundary.token_end == 1
    assert boundary.boundary_index == 1
    assert boundary.text == ""


def test_invalid_non_boundary_range_is_rejected() -> None:
    text = normalize_for_lattice("kelma")
    tokens = tokenize_lattice(text)
    with pytest.raises(ValueError):
        span_for_token_range(text.normalized, tokens, 0, 0)
