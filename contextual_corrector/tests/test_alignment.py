from __future__ import annotations

from contextual_corrector.alignment import align_texts
from contextual_corrector.schema import AlignmentOperation
from contextual_corrector.text import normalize_for_lattice, span_for_token_range, tokenize_lattice


def test_equal_alignment_has_exact_spans() -> None:
    raw = normalize_for_lattice("Ċensu ħareġ")
    alignment = align_texts(raw, normalize_for_lattice("Ċensu ħareġ"))

    assert len(alignment.records) == 1
    record = alignment.records[0]
    assert record.operation == AlignmentOperation.EQUAL
    assert record.confidence == 1.0
    assert not record.ambiguous
    assert record.raw_span.text == "Ċensu ħareġ"
    assert record.s1_spans[0].text == "Ċensu ħareġ"


def test_merge_alignment_is_many_to_many_and_marked_ambiguous() -> None:
    raw = normalize_for_lattice("ma hawnx")
    alignment = align_texts(raw, normalize_for_lattice("m'hawnx"))
    raw_tokens = tokenize_lattice(raw)
    phrase = span_for_token_range(raw.normalized, raw_tokens, 0, 2)
    record = alignment.alignment_for_raw_span(phrase)

    assert record is not None
    assert record.operation == AlignmentOperation.MANY_TO_MANY
    assert record.ambiguous
    assert record.raw_span.text == "ma hawnx"
    assert "".join(span.text for span in record.s1_spans) == "m'hawnx"


def test_inserted_s1_material_uses_explicit_raw_boundary() -> None:
    raw = normalize_for_lattice("kelma")
    alignment = align_texts(raw, normalize_for_lattice("kelma!"))

    inserted = alignment.records[-1]
    assert inserted.operation == AlignmentOperation.INSERT
    assert inserted.raw_span.is_boundary
    assert inserted.raw_span.boundary_index == 1
    assert inserted.s1_spans[0].text == "!"


def test_deleted_raw_material_has_no_s1_span() -> None:
    raw = normalize_for_lattice("kelma !")
    alignment = align_texts(raw, normalize_for_lattice("kelma"))

    deleted = alignment.records[-1]
    assert deleted.operation == AlignmentOperation.DELETE
    assert deleted.raw_span.text == "!"
    assert deleted.s1_spans == ()


def test_candidate_span_can_collect_multiple_uncertain_alignment_records() -> None:
    raw = normalize_for_lattice("il lejla tal-festa")
    alignment = align_texts(raw, normalize_for_lattice("illejla tal-festa"))
    tokens = tokenize_lattice(raw)
    phrase = span_for_token_range(raw.normalized, tokens, 0, 2)
    mapped = alignment.alignment_for_raw_span(phrase)

    assert mapped is not None
    assert mapped.raw_span == phrase
    assert mapped.ambiguous
    assert mapped.s1_spans
