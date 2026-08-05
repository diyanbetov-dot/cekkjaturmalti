from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any, Mapping


class TokenKind(StrEnum):
    WORD = "word"
    NUMBER = "number"
    PUNCTUATION = "punctuation"
    SYMBOL = "symbol"


class CandidateOperation(StrEnum):
    KEEP = "keep"
    REPLACE = "replace"
    SPLIT = "split"
    MERGE = "merge"
    BOUNDARY = "boundary"


class AlignmentOperation(StrEnum):
    EQUAL = "equal"
    REPLACE = "replace"
    INSERT = "insert"
    DELETE = "delete"
    MANY_TO_MANY = "many_to_many"


@dataclass(frozen=True, slots=True)
class TextSpan:
    char_start: int
    char_end: int
    token_start: int
    token_end: int
    text: str
    boundary_index: int | None = None

    def __post_init__(self) -> None:
        if min(self.char_start, self.char_end, self.token_start, self.token_end) < 0:
            raise ValueError("Span offsets cannot be negative.")
        if self.char_start > self.char_end or self.token_start > self.token_end:
            raise ValueError("Span starts must not follow span ends.")
        if (self.char_start == self.char_end) != (self.token_start == self.token_end):
            raise ValueError("Character and token boundary status must agree.")
        if self.is_boundary:
            if self.text:
                raise ValueError("Boundary spans must have empty text.")
            if self.boundary_index is None:
                raise ValueError("Boundary spans require a boundary index.")
        elif self.boundary_index is not None:
            raise ValueError("Non-boundary spans cannot have a boundary index.")

    @property
    def is_boundary(self) -> bool:
        return self.char_start == self.char_end and self.token_start == self.token_end


@dataclass(frozen=True, slots=True)
class LatticeToken:
    index: int
    kind: TokenKind
    char_start: int
    char_end: int
    text: str

    def __post_init__(self) -> None:
        if self.index < 0 or self.char_start < 0 or self.char_end <= self.char_start:
            raise ValueError("Invalid lattice-token offsets.")


@dataclass(frozen=True, slots=True)
class SourceEvidence:
    source: str
    rule_id: str | None = None
    raw_score: float | None = None
    source_confidence: float | None = None
    calibrated_confidence: float | None = None
    calibrated: bool = False
    rank: int | None = None
    deterministic: bool = False
    details: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class MorphologyAnalysis:
    lemma: str | None = None
    part_of_speech: tuple[str, ...] = ()
    features: tuple[str, ...] = ()
    root: str | None = None
    valid: bool | None = None
    confidence: float | None = None
    analyzer: str | None = None


@dataclass(frozen=True, slots=True)
class DictionaryAnalysis:
    entry: str
    normalized_surface: str | None = None
    lemma: str | None = None
    tags: tuple[str, ...] = ()
    part_of_speech: tuple[str, ...] = ()
    inflectional_tags: tuple[str, ...] = ()
    dictionary: str | None = None
    exact: bool = False
    confidence: float | None = None


@dataclass(frozen=True, slots=True)
class SuffixAnalysis:
    lemma: str
    surface: str | None = None
    root_or_stem: str | None = None
    paradigm: str | None = None
    tense_or_mood: str | None = None
    subject_person: str | None = None
    subject_number: str | None = None
    subject_gender: str | None = None
    direct_object: str | None = None
    indirect_object: str | None = None
    has_direct_and_indirect_object: bool = False
    negative: bool = False
    surface_valid: bool = False
    validity_source: str | None = None
    confidence: float | None = None
    rule_id: str | None = None


@dataclass(frozen=True, slots=True)
class IntroducedFeature:
    category: str
    value: str
    input_evidence: bool
    confidence: float | None = None

    @property
    def label(self) -> str:
        return f"{self.category}:{self.value}"


@dataclass(frozen=True, slots=True)
class EditOperation:
    operation: str
    start: int
    end: int
    input_text: str
    output_text: str


@dataclass(frozen=True, slots=True)
class AlignmentRecord:
    raw_span: TextSpan
    s1_spans: tuple[TextSpan, ...]
    operation: AlignmentOperation
    confidence: float
    ambiguous: bool


@dataclass(slots=True)
class SpanCandidate:
    candidate_id: str
    raw_span: TextSpan
    s1_alignment: AlignmentRecord | None
    replacement: str
    output_token_count: int
    operation: CandidateOperation
    keep: bool
    sources: dict[str, tuple[SourceEvidence, ...]] = field(default_factory=dict)
    morphology: tuple[MorphologyAnalysis, ...] = ()
    edit_operations: tuple[EditOperation, ...] = ()
    dictionary_evidence: tuple[DictionaryAnalysis, ...] = ()
    suffix_evidence: tuple[SuffixAnalysis, ...] = ()
    introduced_features: tuple[IntroducedFeature, ...] = ()
    unsupported_clitic_insertion: bool = False
    hard_violations: tuple[str, ...] = ()
    ranker_score: float | None = None
    gold: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def evidence_records(self) -> tuple[SourceEvidence, ...]:
        return tuple(record for records in self.sources.values() for record in records)

    def clone(self, **changes: Any) -> "SpanCandidate":
        return replace(self, **changes)


@dataclass(frozen=True, slots=True)
class LatticeDiagnostics:
    lattice_tokens: int
    candidate_spans_generated: int
    edges_before_deduplication: int
    edges_after_deduplication: int
    edges_removed_by_pruning: int
    candidates_per_span: tuple[tuple[str, int], ...]
    candidate_counts_by_source: tuple[tuple[str, int], ...]
    complete_keep_path: bool
    target_edge_limit: int
    effective_edge_limit: int
