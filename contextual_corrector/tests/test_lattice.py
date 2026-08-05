from __future__ import annotations

from dataclasses import asdict

import pytest

from contextual_corrector.lattice import CandidateLattice, LatticeLimits
from contextual_corrector.schema import (
    CandidateOperation,
    DictionaryAnalysis,
    SourceEvidence,
    SuffixAnalysis,
)
from contextual_corrector.text import normalize_for_lattice


def make_lattice(text: str, *, limits: LatticeLimits | None = None) -> CandidateLattice:
    return CandidateLattice(
        sentence_id="sentence-001",
        raw=normalize_for_lattice(text),
        limits=limits,
    )


def add_stub(
    lattice: CandidateLattice,
    token_start: int,
    token_end: int,
    replacement: str,
    source: str,
    *,
    deterministic: bool = False,
    confidence: float = 0.8,
    operation: CandidateOperation = CandidateOperation.REPLACE,
) -> None:
    span = lattice.span(token_start, token_end)
    lattice.add(
        lattice.make_candidate(
            span=span,
            replacement=replacement,
            operation=operation,
            sources={
                source: (
                    SourceEvidence(
                        source=source,
                        rule_id=f"{source}-rule",
                        raw_score=confidence,
                        calibrated_confidence=confidence,
                        deterministic=deterministic,
                    ),
                )
            },
        )
    )


@pytest.mark.parametrize(
    "text",
    [
        "",
        "\n\n",
        "kelma",
        "kelma   oħra",
        "?!...",
        "🙂",
        "kelma 🙂 mhuxmagħrufa!\n\n oħra",
    ],
)
def test_complete_keep_path_survives_all_supported_input_shapes(text: str) -> None:
    lattice = make_lattice(text)
    edges = lattice.finalize()

    assert lattice.has_complete_keep_path()
    assert lattice.diagnostics().complete_keep_path
    assert sum(candidate.keep for candidate in edges) == len(lattice.tokens)


def test_stable_candidate_id_does_not_depend_on_source_or_order() -> None:
    first = make_lattice("xandek")
    second = make_lattice("xandek")
    span1 = first.span(0, 1)
    span2 = second.span(0, 1)
    a = first.make_candidate(
        span=span1,
        replacement="x'għandek",
        operation=CandidateOperation.REPLACE,
        sources={"stage1": (SourceEvidence(source="stage1"),)},
    )
    b = second.make_candidate(
        span=span2,
        replacement="x'għandek",
        operation=CandidateOperation.REPLACE,
        sources={"bigru": (SourceEvidence(source="bigru"),)},
    )

    assert a.candidate_id == b.candidate_id


def test_deduplication_preserves_multiple_records_from_the_same_source() -> None:
    lattice = make_lattice("xandek")
    span = lattice.span(0, 1)
    for rule_id in ("apostrophe", "restore-gh"):
        lattice.add(
            lattice.make_candidate(
                span=span,
                replacement="x'għandek",
                operation=CandidateOperation.REPLACE,
                sources={
                    "stage1": (
                        SourceEvidence(source="stage1", rule_id=rule_id, deterministic=True),
                    )
                },
                dictionary_evidence=(
                    DictionaryAnalysis(
                        entry="x'għandek", tags=("PRON",), dictionary="test", exact=True
                    ),
                ),
            )
        )

    candidate = next(edge for edge in lattice.finalize() if edge.replacement == "x'għandek")
    assert [record.rule_id for record in candidate.sources["stage1"]] == [
        "apostrophe",
        "restore-gh",
    ]
    assert len(candidate.dictionary_evidence) == 1


def test_typed_suffix_evidence_survives_deduplication() -> None:
    lattice = make_lattice("tefghaw")
    span = lattice.span(0, 1)
    suffix = SuffixAnalysis(
        lemma="tefa'",
        paradigm="F1",
        tense_or_mood="PERF",
        subject_person="3",
        subject_number="P",
        direct_object=None,
        indirect_object=None,
        surface_valid=True,
        confidence=0.94,
    )
    candidate = lattice.make_candidate(
        span=span,
        replacement="tefgħu",
        operation=CandidateOperation.REPLACE,
        sources={"suffix": (SourceEvidence(source="suffix", raw_score=0.94),)},
        suffix_evidence=(suffix,),
    )
    lattice.add(candidate)

    selected = next(edge for edge in lattice.finalize() if edge.replacement == "tefgħu")
    assert selected.suffix_evidence == (suffix,)
    assert selected.suffix_evidence[0].direct_object is None


def test_boundary_candidate_does_not_overlap_singleton_keep_edges() -> None:
    lattice = make_lattice("kelma oħra")
    boundary = lattice.boundary(1)
    lattice.add(
        lattice.make_candidate(
            span=boundary,
            replacement=",",
            operation=CandidateOperation.BOUNDARY,
            sources={"punctuation": (SourceEvidence(source="punctuation"),)},
        )
    )
    edges = lattice.finalize()

    inserted = next(edge for edge in edges if edge.operation == CandidateOperation.BOUNDARY)
    assert inserted.raw_span.char_start == inserted.raw_span.char_end
    assert inserted.raw_span.token_start == inserted.raw_span.token_end == 1
    assert lattice.has_complete_keep_path()


def test_pruning_obeys_reservations_and_never_removes_keep() -> None:
    limits = LatticeLimits(candidates_per_span=12)
    lattice = make_lattice("kelma", limits=limits)
    for index in range(8):
        add_stub(lattice, 0, 1, f"exact{index}", "dictionary", deterministic=True)
    for index in range(7):
        add_stub(lattice, 0, 1, f"suffix{index}", "suffix")
    for index in range(7):
        add_stub(lattice, 0, 1, f"neural{index}", "bigru")
    for index in range(5):
        add_stub(lattice, 0, 1, f"fuzzy{index}", "fuzzy")

    edges = lattice.finalize()
    sources = [next(iter(edge.sources)) for edge in edges if not edge.keep]

    assert len(edges) == 12
    assert sum(edge.keep for edge in edges) == 1
    assert sources.count("dictionary") == 4
    assert sources.count("suffix") == 3
    assert sources.count("bigru") == 3
    assert sources.count("fuzzy") == 1
    assert lattice.diagnostics().edges_removed_by_pruning > 0


def test_span_and_global_limits_are_deterministic() -> None:
    limits = LatticeLimits(
        candidates_per_span=3,
        spans_per_start_token=2,
        minimum_edge_limit=4,
        absolute_edge_limit=4,
        edges_per_token=1,
    )
    lattice = make_lattice("a b c", limits=limits)
    add_stub(lattice, 0, 2, "ab", "phrase", operation=CandidateOperation.MERGE)
    add_stub(lattice, 0, 3, "abc", "phrase", operation=CandidateOperation.MERGE)
    add_stub(lattice, 1, 3, "bc", "phrase", operation=CandidateOperation.MERGE)
    edges = lattice.finalize()

    assert lattice.has_complete_keep_path()
    assert len(edges) <= 4
    assert len({edge.raw_span.token_end for edge in edges if edge.raw_span.token_start == 0}) <= 2


def test_source_specific_candidate_limits_remain_hard_when_capacity_is_free() -> None:
    lattice = make_lattice("kelma")
    for index in range(8):
        add_stub(lattice, 0, 1, f"neural{index}", "bigru", confidence=0.9 - index / 100)
    for index in range(6):
        add_stub(lattice, 0, 1, f"fuzzy{index}", "fuzzy", confidence=0.8 - index / 100)

    edges = lattice.finalize()
    sources = [next(iter(edge.sources)) for edge in edges if not edge.keep]

    assert sources.count("bigru") == 3
    assert sources.count("fuzzy") == 2


def test_candidate_span_cannot_exceed_four_input_tokens() -> None:
    lattice = make_lattice("a b c d e")
    with pytest.raises(ValueError):
        lattice.span(0, 5)


def test_diagnostics_report_required_counts() -> None:
    lattice = make_lattice("ma hawnx")
    add_stub(
        lattice,
        0,
        2,
        "m'hawnx",
        "phrase",
        deterministic=True,
        operation=CandidateOperation.MERGE,
    )
    diagnostics = asdict(lattice.diagnostics())

    assert diagnostics["lattice_tokens"] == 2
    assert diagnostics["candidate_spans_generated"] == 3
    assert diagnostics["edges_before_deduplication"] == 3
    assert diagnostics["edges_after_deduplication"] == 3
    assert diagnostics["complete_keep_path"] is True
    assert ("phrase", 1) in diagnostics["candidate_counts_by_source"]


def test_non_boundary_span_must_match_normalized_raw_text() -> None:
    lattice = make_lattice("kelma")
    span = lattice.span(0, 1)
    broken = span.__class__(
        char_start=span.char_start,
        char_end=span.char_end,
        token_start=span.token_start,
        token_end=span.token_end,
        text="mhux-kelma",
    )
    with pytest.raises(AssertionError):
        lattice.make_candidate(
            span=broken,
            replacement="kelma",
            operation=CandidateOperation.REPLACE,
        )
