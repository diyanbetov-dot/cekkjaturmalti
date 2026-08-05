from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Iterable

from .lattice import CandidateLattice
from .schema import (
    CandidateEligibility,
    CandidateOperation,
    CandidateSupport,
    CandidateValidation,
    CliticEvidence,
    FeatureDelta,
    SpanCandidate,
    SuffixAnalysis,
    TokenKind,
)
from .text import grapheme_edit_distance, tokenize_lattice


MALTESE_STRUCTURAL_PUNCTUATION = frozenset({"'", "’", "-"})


@dataclass(frozen=True, slots=True)
class ValidationResult:
    lattice: CandidateLattice
    records: tuple[CandidateValidation, ...]

    def for_candidate(self, candidate: SpanCandidate | str) -> CandidateValidation:
        candidate_id = candidate if isinstance(candidate, str) else candidate.candidate_id
        return next(record for record in self.records if record.candidate_id == candidate_id)

    @property
    def eligible_candidates(self) -> tuple[SpanCandidate, ...]:
        by_id = {record.candidate_id: record for record in self.records}
        return tuple(edge for edge in self.lattice.edges if by_id[edge.candidate_id].decodable)

    @property
    def hard_invalid(self) -> tuple[CandidateValidation, ...]:
        return tuple(
            record
            for record in self.records
            if record.eligibility == CandidateEligibility.HARD_INVALID
        )


def _valid_unicode(value: str) -> bool:
    try:
        value.encode("utf-8", "strict")
    except UnicodeEncodeError:
        return False
    return unicodedata.normalize("NFC", value) == value


def _reconstruct_edits(source: str, candidate: SpanCandidate) -> str | None:
    if not candidate.edit_operations:
        return source if source == candidate.replacement else None
    cursor = 0
    output: list[str] = []
    for edit in sorted(candidate.edit_operations, key=lambda row: (row.start, row.end)):
        if edit.start < cursor or edit.end < edit.start or edit.end > len(source):
            return None
        if source[edit.start : edit.end] != edit.input_text:
            return None
        output.append(source[cursor : edit.start])
        output.append(edit.output_text)
        cursor = edit.end
    output.append(source[cursor:])
    return "".join(output)


def _expected_operation(candidate: SpanCandidate) -> CandidateOperation:
    span = candidate.raw_span
    if span.is_boundary:
        return CandidateOperation.BOUNDARY
    if candidate.keep:
        return CandidateOperation.KEEP
    if any(character.isspace() for character in span.text) and not any(
        character.isspace() for character in candidate.replacement
    ):
        return CandidateOperation.MERGE
    if not any(character.isspace() for character in span.text) and any(
        character.isspace() for character in candidate.replacement
    ):
        return CandidateOperation.SPLIT
    input_count = span.token_end - span.token_start
    output_count = len(tokenize_lattice(candidate.replacement))
    if input_count > 1 and output_count <= 1:
        return CandidateOperation.MERGE
    if input_count <= 1 and output_count > 1:
        return CandidateOperation.SPLIT
    return CandidateOperation.REPLACE


def _punctuation(value: str) -> tuple[str, ...]:
    return tuple(
        character
        for character in value
        if unicodedata.category(character).startswith("P")
    )


def _illegal_punctuation_relocation(lattice: CandidateLattice, candidate: SpanCandidate) -> bool:
    if candidate.operation == CandidateOperation.BOUNDARY:
        return False
    raw_punctuation = Counter(_punctuation(candidate.raw_span.text))
    replacement_punctuation = Counter(_punctuation(candidate.replacement))
    introduced = replacement_punctuation - raw_punctuation
    introduced_nonstructural = Counter(
        {
            mark: count
            for mark, count in introduced.items()
            if mark not in MALTESE_STRUCTURAL_PUNCTUATION
        }
    )
    if not introduced_nonstructural:
        raw_marks = tuple(
            mark
            for mark in _punctuation(candidate.raw_span.text)
            if mark not in MALTESE_STRUCTURAL_PUNCTUATION
        )
        replacement_marks = tuple(
            mark
            for mark in _punctuation(candidate.replacement)
            if mark not in MALTESE_STRUCTURAL_PUNCTUATION
        )
        if raw_marks != replacement_marks:
            return bool(raw_marks or replacement_marks)
        for mark in set(raw_marks):
            if (
                candidate.raw_span.text.startswith(mark)
                != candidate.replacement.startswith(mark)
                or candidate.raw_span.text.endswith(mark)
                != candidate.replacement.endswith(mark)
            ):
                return True
        return False
    following = lattice.raw.normalized[candidate.raw_span.char_end :]
    return any(mark in following for mark in introduced_nonstructural)


def _analysis_features(analysis: SuffixAnalysis | None) -> dict[str, str]:
    if analysis is None:
        return {}
    values = {
        "lemma": analysis.lemma,
        "stem": analysis.root_or_stem,
        "paradigm": analysis.paradigm,
        "tense_or_mood": analysis.tense_or_mood,
        "subject_person": analysis.subject_person,
        "subject_number": analysis.subject_number,
        "subject_gender": analysis.subject_gender,
        "DO": analysis.direct_object,
        "IDO": analysis.indirect_object,
        "negative": "true" if analysis.negative else "false",
    }
    return {name: value for name, value in values.items() if value is not None}


def _best_analysis(analyses: Iterable[SuffixAnalysis]) -> SuffixAnalysis | None:
    rows = tuple(analyses)
    if not rows:
        return None
    return max(
        rows,
        key=lambda row: (
            bool(row.surface_valid),
            row.confidence if row.confidence is not None else 0.0,
            bool(row.direct_object or row.indirect_object),
        ),
    )


def _feature_delta(raw: SuffixAnalysis | None, candidate: SuffixAnalysis | None) -> FeatureDelta:
    raw_features = _analysis_features(raw)
    candidate_features = _analysis_features(candidate)
    introduced = tuple(
        f"{name}:{candidate_features[name]}"
        for name in sorted(candidate_features.keys() - raw_features.keys())
    )
    removed = tuple(
        f"{name}:{raw_features[name]}"
        for name in sorted(raw_features.keys() - candidate_features.keys())
    )
    changed = tuple(
        (name, raw_features[name], candidate_features[name])
        for name in sorted(raw_features.keys() & candidate_features.keys())
        if raw_features[name] != candidate_features[name]
    )
    preserved = tuple(
        f"{name}:{raw_features[name]}"
        for name in sorted(raw_features.keys() & candidate_features.keys())
        if raw_features[name] == candidate_features[name]
    )
    return FeatureDelta(
        introduced=introduced,
        removed=removed,
        changed=changed,
        preserved=preserved,
    )


def _person(row, kind: str) -> str | None:
    suffix_kind = getattr(row, "suffix_kind", None)
    suffix_person = getattr(row, "suffix_person", "") or ""
    if suffix_kind == kind:
        return suffix_person or None
    if suffix_kind == "DO_IDO":
        direct, _, indirect = suffix_person.partition("+")
        return direct if kind == "DO" else indirect
    return None


def _row_matches_analysis(row, analysis: SuffixAnalysis, surface: str) -> bool:
    row_surface = getattr(row, "surface", None)
    if row_surface is None or unicodedata.normalize("NFC", row_surface).casefold() != surface.casefold():
        return False
    comparisons = (
        (analysis.lemma, getattr(row, "base", None)),
        (analysis.root_or_stem, getattr(row, "root", None)),
        (analysis.paradigm, getattr(row, "form_class", None)),
        (analysis.tense_or_mood, getattr(row, "tense", None)),
        (analysis.direct_object, _person(row, "DO")),
        (analysis.indirect_object, _person(row, "IDO")),
    )
    if any(expected is not None and expected != actual for expected, actual in comparisons):
        return False
    row_subject = getattr(row, "person", None) or ""
    if analysis.subject_person and not row_subject.startswith(analysis.subject_person):
        return False
    if analysis.subject_number and len(row_subject) > 1 and row_subject[1:2] != analysis.subject_number:
        return False
    if analysis.subject_gender and len(row_subject) > 2 and row_subject[2:3] != analysis.subject_gender:
        return False
    return True


def _roundtrip_suffix(
    generator,
    candidate: SpanCandidate,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if not candidate.suffix_evidence:
        return (), ()
    passed: list[str] = []
    failed: list[str] = []
    generated_rows = tuple(generator.candidates_for_surface(candidate.replacement, limit=64))
    verb_index = getattr(generator, "verb_index", None)
    verb_rows = (
        tuple(verb_index.word_records(candidate.replacement))
        if verb_index is not None
        else ()
    )
    for index, analysis in enumerate(candidate.suffix_evidence):
        label = f"{index}:{analysis.lemma}:{analysis.rule_id or analysis.validity_source or 'typed'}"
        if analysis.validity_source == "verb_form_index":
            matching = any(
                (analysis.root_or_stem is None or analysis.root_or_stem == getattr(row, "root", None))
                and (analysis.paradigm is None or analysis.paradigm == getattr(row, "form_class", None))
                and (analysis.tense_or_mood is None or analysis.tense_or_mood == getattr(row, "tense", None))
                for row in verb_rows
            )
        else:
            matching = any(
                _row_matches_analysis(row, analysis, candidate.replacement)
                for row in generated_rows
            )
        (passed if matching else failed).append(label)
    return tuple(passed), tuple(failed)


def _raw_suffix_analyses(generator, surface: str) -> tuple[SuffixAnalysis, ...]:
    from .adapters import (
        _generated_suffix_analysis,
        _split_suffix_person,
        _verb_record_analysis,
    )

    rows = tuple(generator.candidates_for_surface(surface, limit=64))
    exact_matches = getattr(generator, "exact_suffix_matches", lambda _surface: ())
    rows += tuple(exact_matches(surface))
    analyses = [_generated_suffix_analysis(row, negative=surface.casefold().endswith("x")) for row in rows]
    verb_index = getattr(generator, "verb_index", None)
    if verb_index is not None:
        analyses.extend(
            _verb_record_analysis(row, surface)
            for row in verb_index.word_records(surface)
        )
        parse_suffixes = getattr(generator, "parse_possible_suffixes", lambda _surface: ())
        for parsed in parse_suffixes(surface):
            direct, indirect = _split_suffix_person(parsed.spec.kind, parsed.spec.person)
            stem_variants = [parsed.typed_stem]
            if parsed.typed_stem.casefold().startswith("i") and len(parsed.typed_stem) > 1:
                stem_variants.extend(
                    ("j" + parsed.typed_stem[1:], parsed.typed_stem[1:])
                )
            records = tuple(
                record
                for stem in dict.fromkeys(stem_variants)
                for record in verb_index.word_records(stem)
            )
            for record in records:
                analyses.append(
                    SuffixAnalysis(
                        lemma=record.word,
                        surface=surface,
                        root_or_stem=record.root,
                        paradigm=record.form_class,
                        tense_or_mood=record.tense,
                        subject_person=record.person[:1] if record.person else None,
                        subject_number=record.person[1:2] if len(record.person) > 1 else None,
                        subject_gender=record.person[2:3] if len(record.person) > 2 else None,
                        direct_object=direct,
                        indirect_object=indirect,
                        has_direct_and_indirect_object=parsed.spec.kind == "DO_IDO",
                        negative=surface.casefold().endswith("x"),
                        surface_valid=True,
                        validity_source="suffix_parse+verb_form_index",
                        rule_id=getattr(parsed.spec, "rule_id", parsed.spec.label),
                    )
                )
    return tuple(dict.fromkeys(analyses))


def _raw_clitic_features(generator, surface: str) -> tuple[set[str], set[str]]:
    from .adapters import _split_suffix_person

    direct: set[str] = set()
    indirect: set[str] = set()
    parse_suffixes = getattr(generator, "parse_possible_suffixes", lambda _surface: ())
    for parsed in parse_suffixes(surface):
        parsed_do, parsed_ido = _split_suffix_person(parsed.spec.kind, parsed.spec.person)
        if parsed_do:
            direct.add(parsed_do)
        if parsed_ido:
            indirect.add(parsed_ido)
    return direct, indirect


def _clitic_evidence(
    raw_analysis: SuffixAnalysis | None,
    candidate_analysis: SuffixAnalysis | None,
    candidate: SpanCandidate,
    *,
    raw_do_evidence: set[str] | None = None,
    raw_ido_evidence: set[str] | None = None,
) -> CliticEvidence:
    raw_do = {raw_analysis.direct_object} if raw_analysis and raw_analysis.direct_object else set()
    raw_ido = {raw_analysis.indirect_object} if raw_analysis and raw_analysis.indirect_object else set()
    raw_do.update(raw_do_evidence or ())
    raw_ido.update(raw_ido_evidence or ())
    candidate_do = (
        {candidate_analysis.direct_object}
        if candidate_analysis and candidate_analysis.direct_object
        else set()
    )
    candidate_ido = (
        {candidate_analysis.indirect_object}
        if candidate_analysis and candidate_analysis.indirect_object
        else set()
    )
    similarity = SequenceMatcher(
        None, candidate.raw_span.text.casefold(), candidate.replacement.casefold(), autojunk=False
    ).ratio()
    introduced_do = tuple(sorted(candidate_do - raw_do))
    introduced_ido = tuple(sorted(candidate_ido - raw_ido))
    unsupported = candidate.unsupported_clitic_insertion or bool(introduced_do or introduced_ido)
    antecedent = tuple(
        feature.label
        for feature in candidate.introduced_features
        if feature.input_evidence and feature.category in {"DO", "IDO"}
    )
    if antecedent:
        unsupported = False
    return CliticEvidence(
        raw_has_clitic_evidence=bool(raw_do or raw_ido),
        candidate_has_do=bool(candidate_do),
        candidate_has_ido=bool(candidate_ido),
        introduced_do=introduced_do,
        introduced_ido=introduced_ido,
        removed_do=tuple(sorted(raw_do - candidate_do)),
        removed_ido=tuple(sorted(raw_ido - candidate_ido)),
        clitic_surface_similarity=similarity,
        antecedent_evidence=antecedent,
        unsupported_clitic_insertion=unsupported,
    )


class CandidateValidator:
    def __init__(self, *, suffix_generator=None) -> None:
        self.suffix_generator = suffix_generator

    def validate(self, lattice: CandidateLattice) -> ValidationResult:
        records = tuple(self.validate_candidate(lattice, candidate) for candidate in lattice.edges)
        return ValidationResult(lattice=lattice, records=records)

    def validate_candidate(
        self, lattice: CandidateLattice, candidate: SpanCandidate
    ) -> CandidateValidation:
        violations: list[str] = list(candidate.hard_violations)
        warnings: list[str] = []
        span = candidate.raw_span
        try:
            expected_span = (
                lattice.boundary(span.boundary_index or 0)
                if span.is_boundary
                else lattice.span(span.token_start, span.token_end)
            )
            if span != expected_span:
                violations.append("INCONSISTENT_RAW_SPAN")
        except (AssertionError, ValueError):
            violations.append("MALFORMED_RAW_SPAN")
        if lattice.raw.normalized[span.char_start : span.char_end] != span.text:
            violations.append("RAW_OFFSET_MISMATCH")
        if not _valid_unicode(candidate.replacement):
            violations.append("INVALID_UNICODE_REPLACEMENT")
        if candidate.output_token_count != len(tokenize_lattice(candidate.replacement)):
            violations.append("OUTPUT_TOKEN_COUNT_MISMATCH")
        expected_operation = _expected_operation(candidate)
        if candidate.operation != expected_operation:
            violations.append("CONTRADICTORY_CANDIDATE_OPERATION")
        reconstructed = _reconstruct_edits(span.text, candidate)
        if reconstructed != candidate.replacement:
            violations.append("EDIT_RECONSTRUCTION_FAILED")
        if candidate.keep and (
            candidate.replacement != span.text
            or candidate.operation != CandidateOperation.KEEP
            or candidate.edit_operations
        ):
            violations.append("INVALID_KEEP_MUTATION")
        if any(analysis.valid is False for analysis in candidate.morphology):
            violations.append("MORPHOLOGY_CLAIM_INVALID")
        if any(
            analysis.surface is not None
            and unicodedata.normalize("NFC", analysis.surface).casefold()
            != candidate.replacement.casefold()
            for analysis in candidate.suffix_evidence
        ):
            violations.append("SUFFIX_SURFACE_MISMATCH")
        if span.is_boundary and candidate.operation != CandidateOperation.BOUNDARY:
            violations.append("ILLEGAL_BOUNDARY_OPERATION")
        if not span.is_boundary and candidate.operation == CandidateOperation.BOUNDARY:
            violations.append("ILLEGAL_BOUNDARY_OPERATION")
        if _illegal_punctuation_relocation(lattice, candidate):
            violations.append("ILLEGAL_PUNCTUATION_RELOCATION")
        if re.search(r"(?i)\bma'\s*[^\s]+x\b", candidate.replacement) and re.search(
            r"(?i)\bma\s+[^\s]+x\b", span.text
        ):
            violations.append("APOSTROPHIZED_MA_BEFORE_VERB")
        if re.search(r"(?i)\bta'-(?=\w)", candidate.replacement):
            violations.append("ILLEGAL_APOSTROPHE_HYPHEN_SEQUENCE")

        roundtrip_passed: tuple[str, ...] = ()
        roundtrip_failed: tuple[str, ...] = ()
        raw_analyses: tuple[SuffixAnalysis, ...] = ()
        raw_do_evidence: set[str] = set()
        raw_ido_evidence: set[str] = set()
        if self.suffix_generator is not None:
            if candidate.suffix_evidence:
                roundtrip_passed, roundtrip_failed = _roundtrip_suffix(
                    self.suffix_generator, candidate
                )
                if roundtrip_failed:
                    violations.append("SUFFIX_ANALYSIS_ROUNDTRIP_FAILED")
            if not span.is_boundary and span.token_end - span.token_start == 1:
                raw_analyses = _raw_suffix_analyses(self.suffix_generator, span.text)
                raw_do_evidence, raw_ido_evidence = _raw_clitic_features(
                    self.suffix_generator, span.text
                )
        elif candidate.suffix_evidence:
            warnings.append("SUFFIX_ROUNDTRIP_NOT_RUN")

        raw_analysis = _best_analysis(raw_analyses)
        candidate_analysis = _best_analysis(candidate.suffix_evidence)
        feature_delta = _feature_delta(raw_analysis, candidate_analysis)
        clitic = _clitic_evidence(
            raw_analysis,
            candidate_analysis,
            candidate,
            raw_do_evidence=raw_do_evidence,
            raw_ido_evidence=raw_ido_evidence,
        )
        exact_dictionary = any(row.exact for row in candidate.dictionary_evidence)
        lexical_validity = True if exact_dictionary or candidate.keep else None
        morphological_validity = (
            not roundtrip_failed if candidate.suffix_evidence and self.suffix_generator else None
        )
        unsupported_surface = (
            not candidate.keep
            and lexical_validity is not True
            and morphological_validity is not True
        )
        if unsupported_surface:
            warnings.append("UNSUPPORTED_SURFACE")
        if clitic.unsupported_clitic_insertion:
            warnings.append("UNSUPPORTED_CLITIC_INSERTION")

        if candidate.metadata.get("diagnostic_only"):
            eligibility = CandidateEligibility.DIAGNOSTIC_ONLY
        elif violations:
            eligibility = CandidateEligibility.HARD_INVALID
        elif warnings:
            eligibility = CandidateEligibility.SOFTLY_UNSUPPORTED
        else:
            eligibility = CandidateEligibility.ELIGIBLE
        return CandidateValidation(
            candidate_id=candidate.candidate_id,
            eligibility=eligibility,
            violations=tuple(dict.fromkeys(violations)),
            warnings=tuple(dict.fromkeys(warnings)),
            support=CandidateSupport(
                lexical_validity=lexical_validity,
                morphological_validity=morphological_validity,
                dictionary_exact=exact_dictionary,
                unsupported_surface=unsupported_surface,
            ),
            feature_delta=feature_delta,
            clitic_evidence=clitic,
            roundtrip_passed=roundtrip_passed,
            roundtrip_failed=roundtrip_failed,
        )


def validate_lattice(lattice: CandidateLattice, *, suffix_generator=None) -> ValidationResult:
    return CandidateValidator(suffix_generator=suffix_generator).validate(lattice)
