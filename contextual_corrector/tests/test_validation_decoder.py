from __future__ import annotations

from collections import Counter
from dataclasses import replace
from types import SimpleNamespace

import pytest

from contextual_corrector import (
    CandidateEligibility,
    CandidateLattice,
    CandidateOperation,
    CandidateValidator,
    DictionaryAnalysis,
    EditOperation,
    SourceEvidence,
    SoftEditBudget,
    SpanCandidate,
    SuffixAnalysis,
    candidate_edit_cost,
    decode_lattice,
    incremental_edit_cost,
    normalize_for_lattice,
    validate_lattice,
)
from contextual_corrector.adapters import edit_operations


LOCKED = "Censu hareg minn gol vann, tefghaw barra ghax ma riedx iwassalhom."


def lattice(text: str) -> CandidateLattice:
    return CandidateLattice(sentence_id="commit-3", raw=normalize_for_lattice(text))


def source(name: str = "test"):
    return {name: (SourceEvidence(source=name, rule_id="fixture"),)}


def add(
    graph: CandidateLattice,
    start: int,
    end: int,
    replacement: str,
    *,
    operation: CandidateOperation = CandidateOperation.REPLACE,
    suffix_evidence=(),
    dictionary_evidence=(),
    unsupported=False,
    source_name="test",
) -> SpanCandidate:
    span = graph.span(start, end)
    candidate = graph.make_candidate(
        span=span,
        replacement=replacement,
        operation=operation,
        sources=source(source_name),
        suffix_evidence=tuple(suffix_evidence),
        dictionary_evidence=tuple(dictionary_evidence),
        unsupported_clitic_insertion=unsupported,
        edit_operations=edit_operations(span.text, replacement),
    )
    graph.add(candidate)
    return candidate


def row(
    surface: str,
    *,
    base: str,
    root: str,
    form: str,
    tense: str,
    person: str,
    kind: str = "NONE",
    clitic_person: str = "",
):
    return SimpleNamespace(
        surface=surface,
        base=base,
        root=root,
        form_class=form,
        tense=tense,
        person=person,
        suffix_kind=kind,
        suffix_person=clitic_person,
        suffix_label=f"{kind}_{clitic_person}",
        suffix_display="",
        rule_id="fixture_suffix",
        raw_tag="fixture",
    )


def analysis(value, *, source="suffix_generator") -> SuffixAnalysis:
    direct = value.suffix_person if value.suffix_kind == "DO" else None
    indirect = value.suffix_person if value.suffix_kind == "IDO" else None
    return SuffixAnalysis(
        lemma=value.base,
        surface=value.surface,
        root_or_stem=value.root,
        paradigm=value.form_class,
        tense_or_mood=value.tense,
        subject_person=value.person[:1],
        subject_number=value.person[1:2],
        subject_gender=value.person[2:3],
        direct_object=direct,
        indirect_object=indirect,
        surface_valid=True,
        validity_source=source,
        rule_id=value.rule_id,
    )


class FakeVerbIndex:
    def __init__(self, rows):
        self.rows = rows

    def word_records(self, surface):
        return self.rows.get(surface, ())


class FakeGenerator:
    def __init__(self, rows=(), verb_rows=None):
        self.rows = {}
        for value in rows:
            self.rows.setdefault(value.surface, []).append(value)
        self.verb_index = FakeVerbIndex(verb_rows or {})

    def candidates_for_surface(self, surface, limit=64):
        return self.rows.get(surface, ())[:limit]


def test_punctuation_bearing_neural_candidate_is_hard_invalid_but_valid_alternatives_remain():
    graph = lattice(LOCKED)
    moved = add(graph, 3, 4, "ġol,", operation=CandidateOperation.SPLIT, source_name="bigru")
    lexical = add(graph, 3, 4, "ġol", source_name="bigru")
    phrase = add(graph, 3, 5, "ġol-vann", operation=CandidateOperation.MERGE, source_name="phrase")
    result = validate_lattice(graph)

    assert result.for_candidate(moved).eligibility == CandidateEligibility.HARD_INVALID
    assert "ILLEGAL_PUNCTUATION_RELOCATION" in result.for_candidate(moved).violations
    assert result.for_candidate(lexical).decodable
    assert result.for_candidate(phrase).decodable


def test_unsupported_neural_surface_remains_diagnostic_and_decodable():
    graph = lattice("tefghaw")
    candidate = add(graph, 0, 1, "tefgħau", source_name="bigru")
    record = validate_lattice(graph).for_candidate(candidate)

    assert record.eligibility == CandidateEligibility.SOFTLY_UNSUPPORTED
    assert record.support.lexical_validity is None
    assert record.support.morphological_validity is None
    assert record.support.unsupported_surface


def test_integrity_checks_detect_metadata_edit_and_suffix_surface_conflicts():
    graph = lattice("kelma")
    base = graph.make_candidate(
        span=graph.span(0, 1),
        replacement="oħra",
        operation=CandidateOperation.REPLACE,
        sources=source(),
        edit_operations=edit_operations("kelma", "oħra"),
    )
    validator = CandidateValidator()

    wrong_count = base.clone(output_token_count=7)
    assert "OUTPUT_TOKEN_COUNT_MISMATCH" in validator.validate_candidate(graph, wrong_count).violations

    wrong_operation = base.clone(operation=CandidateOperation.MERGE)
    assert "CONTRADICTORY_CANDIDATE_OPERATION" in validator.validate_candidate(graph, wrong_operation).violations

    wrong_edits = base.clone(
        edit_operations=(EditOperation("replace", 0, 1, "x", "o"),)
    )
    assert "EDIT_RECONSTRUCTION_FAILED" in validator.validate_candidate(graph, wrong_edits).violations

    suffix = SuffixAnalysis(lemma="oħra", surface="mhux-oħra", surface_valid=True)
    wrong_surface = base.clone(suffix_evidence=(suffix,))
    assert "SUFFIX_SURFACE_MISMATCH" in validator.validate_candidate(graph, wrong_surface).violations


def test_invalid_keep_mutation_is_hard_invalid():
    graph = lattice("kelma")
    keep = next(candidate for candidate in graph.edges if candidate.keep)
    mutated = keep.clone(replacement="oħra")
    record = CandidateValidator().validate_candidate(graph, mutated)
    assert record.eligibility == CandidateEligibility.HARD_INVALID
    assert "INVALID_KEEP_MUTATION" in record.violations


def test_suffix_roundtrip_failure_is_excluded_from_decoding():
    graph = lattice("iwassalhom")
    claimed = row(
        "wassagħalhom", base="wassal", root="wsl", form="F2", tense="PERF", person="3SM", kind="DO", clitic_person="3P"
    )
    candidate = add(graph, 0, 1, claimed.surface, suffix_evidence=(analysis(claimed),))
    result = validate_lattice(graph, suffix_generator=FakeGenerator())
    record = result.for_candidate(candidate)
    decoded = decode_lattice(graph, lambda edge: 100.0 if edge == candidate else 0.0, validation=result)

    assert "SUFFIX_ANALYSIS_ROUNDTRIP_FAILED" in record.violations
    assert candidate.candidate_id not in decoded.diagnostics.selected_candidate_ids
    assert decoded.rendered_text == "iwassalhom"


def test_suffix_roundtrip_and_feature_deltas_are_typed():
    raw_row = row(
        "iwassalhom", base="wassal", root="wsl", form="F2", tense="IMPF", person="3SM", kind="DO", clitic_person="3P"
    )
    replacement_row = row(
        "wassalhom", base="wassal", root="wsl", form="F2", tense="PERF", person="3SM", kind="DO", clitic_person="3P"
    )
    graph = lattice("iwassalhom")
    candidate = add(graph, 0, 1, "wassalhom", suffix_evidence=(analysis(replacement_row),))
    result = validate_lattice(
        graph, suffix_generator=FakeGenerator((raw_row, replacement_row))
    )
    record = result.for_candidate(candidate)

    assert record.roundtrip_passed
    assert not record.roundtrip_failed
    assert ("tense_or_mood", "IMPF", "PERF") in record.feature_delta.changed
    assert "lemma:wassal" in record.feature_delta.preserved
    assert "stem:wsl" in record.feature_delta.preserved
    assert record.clitic_evidence.raw_has_clitic_evidence
    assert record.clitic_evidence.candidate_has_do


def test_tefghu_has_no_do_and_tefghuh_introduces_softly_penalized_do():
    unsuffixed_row = row(
        "tefgħu", base="tefgħu", root="tfgħ", form="F1", tense="IMP", person="2P"
    )
    suffixed_row = row(
        "tefgħuh", base="tefgħu", root="tfgħ", form="F1", tense="IMP", person="2P", kind="DO", clitic_person="3SM"
    )
    graph = lattice("tefghaw")
    unsuffixed = add(graph, 0, 1, "tefgħu", suffix_evidence=(analysis(unsuffixed_row),))
    suffixed = add(
        graph,
        0,
        1,
        "tefgħuh",
        suffix_evidence=(analysis(suffixed_row),),
        unsupported=True,
    )
    result = validate_lattice(
        graph, suffix_generator=FakeGenerator((unsuffixed_row, suffixed_row))
    )
    plain_record = result.for_candidate(unsuffixed)
    clitic_record = result.for_candidate(suffixed)

    assert not plain_record.clitic_evidence.candidate_has_do
    assert not plain_record.clitic_evidence.introduced_do
    assert clitic_record.clitic_evidence.introduced_do == ("3SM",)
    assert clitic_record.clitic_evidence.unsupported_clitic_insertion
    assert candidate_edit_cost(suffixed, clitic_record).unsupported_clitic > 0


@pytest.mark.parametrize(
    ("raw", "replacement", "code"),
    [
        ("ma riedx", "ma'riedx", "APOSTROPHIZED_MA_BEFORE_VERB"),
        ("ta' wieħed", "ta'-wieħed", "ILLEGAL_APOSTROPHE_HYPHEN_SEQUENCE"),
    ],
)
def test_locally_provable_phrase_constraints_are_hard(raw, replacement, code):
    graph = lattice(raw)
    candidate = add(graph, 0, len(graph.tokens), replacement, operation=CandidateOperation.MERGE)
    record = validate_lattice(graph).for_candidate(candidate)
    assert code in record.violations
    assert not record.decodable


def test_repeated_diacritic_restoration_costs_less_than_unrelated_substitutions():
    diacritics = lattice("ghax ghal")
    first = add(diacritics, 0, 1, "għax")
    second = add(diacritics, 1, 2, "għal")
    validated = validate_lattice(diacritics)
    prior = Counter()
    first_cost = candidate_edit_cost(first, validated.for_candidate(first))
    total_diacritic = incremental_edit_cost(first_cost, prior)
    prior.update(first_cost.coherent_signatures)
    second_cost = candidate_edit_cost(second, validated.for_candidate(second))
    total_diacritic += incremental_edit_cost(second_cost, prior)

    lexical = lattice("dar bieb")
    third = add(lexical, 0, 1, "siġra")
    fourth = add(lexical, 1, 2, "tieqa")
    lexical_validated = validate_lattice(lexical)
    total_lexical = sum(
        candidate_edit_cost(candidate, lexical_validated.for_candidate(candidate)).total
        for candidate in (third, fourth)
    )
    assert total_diacritic < total_lexical


def test_edit_budget_is_soft_and_sentence_length_aware():
    budget = SoftEditBudget()
    assert budget.allowance(100) > budget.allowance(1)
    assert budget.penalty(20.0, 4) > 0
    assert budget.penalty(20.0, 100) < budget.penalty(20.0, 4)


def test_decoder_supports_merge_split_and_never_selects_overlap():
    graph = lattice("a bc")
    merge = add(graph, 0, 2, "abc", operation=CandidateOperation.MERGE)
    overlap = add(graph, 1, 2, "b c", operation=CandidateOperation.SPLIT)
    result = validate_lattice(graph)
    decoded = decode_lattice(
        graph,
        lambda edge: 10.0 if edge.candidate_id == merge.candidate_id else (
            8.0 if edge.candidate_id == overlap.candidate_id else 0.0
        ),
        validation=result,
    )
    assert merge in decoded.candidates
    assert overlap not in decoded.candidates
    assert decoded.diagnostics.input_coverage == (0, 1)

    split_graph = lattice("abc")
    split = add(split_graph, 0, 1, "a bc", operation=CandidateOperation.SPLIT)
    split_result = validate_lattice(split_graph)
    assert decode_lattice(
        split_graph, lambda edge: 8.0 if edge == split else 0.0, validation=split_result
    ).rendered_text == "a bc"


def test_boundary_edges_are_single_use_and_cannot_loop():
    graph = lattice("kelma")
    span = graph.boundary(1)
    boundary = graph.make_candidate(
        span=span,
        replacement="!",
        operation=CandidateOperation.BOUNDARY,
        sources=source("punctuation"),
        edit_operations=(EditOperation("insert", 0, 0, "", "!"),),
    )
    graph.add(boundary)
    result = validate_lattice(graph)
    decoded = decode_lattice(
        graph, lambda edge: 5.0 if edge == boundary else 0.0, validation=result
    )
    assert decoded.rendered_text == "kelma!"
    assert decoded.candidates.count(boundary) == 1
    assert decoded.diagnostics.boundary_operations_selected == (boundary.candidate_id,)


def test_incompatible_boundary_insertions_cannot_both_be_selected():
    graph = lattice("kelma")
    candidates = []
    for mark in ("!", "?"):
        span = graph.boundary(1)
        candidate = graph.make_candidate(
            span=span,
            replacement=mark,
            operation=CandidateOperation.BOUNDARY,
            sources=source("punctuation"),
            edit_operations=(EditOperation("insert", 0, 0, "", mark),),
        )
        graph.add(candidate)
        candidates.append(candidate)
    result = validate_lattice(graph)
    decoded = decode_lattice(
        graph,
        lambda edge: 6.0 if edge.candidate_id in {row.candidate_id for row in candidates} else 0.0,
        validation=result,
    )
    assert sum(candidate in decoded.candidates for candidate in candidates) == 1


@pytest.mark.parametrize("text", ["", "   \n", "?!", "🙂", "a\n\n b"])
def test_decoder_always_returns_complete_keep_path_for_supported_shapes(text):
    graph = lattice(text)
    before = repr(graph.edges)
    result = validate_lattice(graph)
    decoded = decode_lattice(graph, lambda _edge: 0.0, validation=result)
    assert decoded.rendered_text == text
    assert decoded.diagnostics.input_coverage == tuple(range(len(graph.tokens)))
    assert repr(graph.edges) == before


def test_invalid_candidates_remain_visible_in_diagnostics():
    graph = lattice("gol vann,")
    invalid = add(graph, 0, 1, "ġol,", operation=CandidateOperation.SPLIT, source_name="bigru")
    result = validate_lattice(graph)
    decoded = decode_lattice(graph, lambda edge: 99.0 if edge == invalid else 0.0, validation=result)
    assert any(
        record.candidate_id == invalid.candidate_id
        for record in decoded.diagnostics.hard_invalid_candidates_excluded
    )


def test_locked_sentence_decodes_expected_fixture_path():
    graph = lattice(LOCKED)
    censu = add(graph, 0, 1, "Ċensu", dictionary_evidence=(DictionaryAnalysis(entry="Ċensu", exact=True),))
    hareg = add(graph, 1, 2, "ħareġ", dictionary_evidence=(DictionaryAnalysis(entry="ħareġ", exact=True),))
    moved = add(graph, 3, 4, "ġol,", operation=CandidateOperation.SPLIT, source_name="bigru")
    add(graph, 3, 4, "ġol", dictionary_evidence=(DictionaryAnalysis(entry="ġol", exact=True),))
    phrase = add(graph, 3, 5, "ġol-vann", operation=CandidateOperation.MERGE, source_name="phrase")
    unsuffixed_row = row("tefgħu", base="tefgħu", root="tfgħ", form="F1", tense="IMP", person="2P")
    suffixed_row = row("tefgħuh", base="tefgħu", root="tfgħ", form="F1", tense="IMP", person="2P", kind="DO", clitic_person="3SM")
    tefghu = add(graph, 6, 7, "tefgħu", suffix_evidence=(analysis(unsuffixed_row),))
    tefghuh = add(graph, 6, 7, "tefgħuh", suffix_evidence=(analysis(suffixed_row),), unsupported=True)
    ghax = add(graph, 8, 9, "għax", dictionary_evidence=(DictionaryAnalysis(entry="għax", exact=True),))
    ma_bad = add(graph, 9, 11, "ma'riedx", operation=CandidateOperation.MERGE)
    bad_iwassal = row("wassagħalhom", base="wassal", root="wsl", form="F2", tense="PERF", person="3SM", kind="DO", clitic_person="3P")
    odd = add(graph, 11, 12, bad_iwassal.surface, suffix_evidence=(analysis(bad_iwassal),))
    generator = FakeGenerator((unsuffixed_row, suffixed_row))
    result = validate_lattice(graph, suffix_generator=generator)
    preferred = {candidate.candidate_id for candidate in (censu, hareg, phrase, tefghu, ghax)}
    decoded = decode_lattice(
        graph,
        lambda edge: 8.0 if edge.candidate_id in preferred else 0.0,
        validation=result,
    )

    assert decoded.rendered_text == "Ċensu ħareġ minn ġol-vann, tefgħu barra għax ma riedx iwassalhom."
    assert "OUTPUT:" in decoded.format()
    assert tefghuh.candidate_id not in decoded.diagnostics.selected_candidate_ids
    assert result.for_candidate(tefghuh).clitic_evidence.introduced_do == ("3SM",)
    assert "ILLEGAL_PUNCTUATION_RELOCATION" in result.for_candidate(moved).violations
    assert "APOSTROPHIZED_MA_BEFORE_VERB" in result.for_candidate(ma_bad).violations
    assert "SUFFIX_ANALYSIS_ROUNDTRIP_FAILED" in result.for_candidate(odd).violations
    assert all(
        any(
            edge.keep and edge.raw_span.token_start == index
            for edge in graph.edges
        )
        for index in range(len(graph.tokens))
    )
