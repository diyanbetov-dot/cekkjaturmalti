from __future__ import annotations

import hashlib
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass

from .alignment import AlignmentMap
from .schema import (
    CandidateOperation,
    DictionaryAnalysis,
    LatticeDiagnostics,
    MorphologyAnalysis,
    SourceEvidence,
    SpanCandidate,
    SuffixAnalysis,
    TextSpan,
)
from .text import NormalizedText, boundary_span, span_for_token_range, tokenize_lattice


@dataclass(frozen=True, slots=True)
class LatticeLimits:
    candidates_per_span: int = 12
    spans_per_start_token: int = 6
    maximum_phrase_tokens: int = 4
    fuzzy_candidates_per_span: int = 2
    neural_candidates_per_span: int = 3
    suffix_candidates_per_span: int = 4
    absolute_edge_limit: int = 1536
    minimum_edge_limit: int = 256
    edges_per_token: int = 24


class CandidateLattice:
    def __init__(
        self,
        *,
        sentence_id: str,
        raw: NormalizedText,
        s1_alignment: AlignmentMap | None = None,
        limits: LatticeLimits | None = None,
    ) -> None:
        self.sentence_id = sentence_id
        self.raw = raw
        self.tokens = tokenize_lattice(raw)
        self.s1_alignment = s1_alignment
        self.limits = limits or LatticeLimits()
        self._proposals: list[SpanCandidate] = []
        self._edges: tuple[SpanCandidate, ...] = ()
        self._edges_after_deduplication = 0
        self._removed_by_pruning = 0
        self._add_keep_edges()

    @staticmethod
    def normalize_replacement(replacement: str) -> str:
        return unicodedata.normalize("NFC", replacement)

    def candidate_id(self, span: TextSpan, replacement: str) -> str:
        identity = "\0".join(
            (
                self.sentence_id,
                str(span.char_start),
                str(span.char_end),
                self.normalize_replacement(replacement),
            )
        )
        return "c_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]

    def span(self, token_start: int, token_end: int) -> TextSpan:
        if token_end - token_start > self.limits.maximum_phrase_tokens:
            raise ValueError("Candidate span exceeds the configured phrase length.")
        return span_for_token_range(self.raw.normalized, self.tokens, token_start, token_end)

    def boundary(self, boundary_index: int) -> TextSpan:
        return boundary_span(self.raw.normalized, self.tokens, boundary_index)

    def make_candidate(
        self,
        *,
        span: TextSpan,
        replacement: str,
        operation: CandidateOperation,
        keep: bool = False,
        sources: dict[str, tuple[SourceEvidence, ...]] | None = None,
        morphology: tuple[MorphologyAnalysis, ...] = (),
        dictionary_evidence: tuple[DictionaryAnalysis, ...] = (),
        suffix_evidence: tuple[SuffixAnalysis, ...] = (),
        **kwargs,
    ) -> SpanCandidate:
        if span.is_boundary:
            expected_span = self.boundary(span.boundary_index or 0)
        else:
            expected_span = self.span(span.token_start, span.token_end)
        if span != expected_span:
            raise AssertionError("RAW character and token spans are inconsistent.")
        if not span.is_boundary and self.raw.normalized[span.char_start : span.char_end] != span.text:
            raise AssertionError("RAW span does not round-trip against normalized input.")
        replacement = self.normalize_replacement(replacement)
        return SpanCandidate(
            candidate_id=self.candidate_id(span, replacement),
            raw_span=span,
            s1_alignment=(
                self.s1_alignment.alignment_for_raw_span(span)
                if self.s1_alignment is not None
                else None
            ),
            replacement=replacement,
            output_token_count=len(tokenize_lattice(replacement)),
            operation=operation,
            keep=keep,
            sources=sources or {},
            morphology=morphology,
            dictionary_evidence=dictionary_evidence,
            suffix_evidence=suffix_evidence,
            **kwargs,
        )

    def add(self, candidate: SpanCandidate) -> None:
        if candidate.candidate_id != self.candidate_id(candidate.raw_span, candidate.replacement):
            raise ValueError("Candidate ID does not match its stable identity fields.")
        self._proposals.append(candidate)
        self._edges = ()

    def _add_keep_edges(self) -> None:
        for token in self.tokens:
            span = self.span(token.index, token.index + 1)
            self.add(
                self.make_candidate(
                    span=span,
                    replacement=span.text,
                    operation=CandidateOperation.KEEP,
                    keep=True,
                    sources={
                        "KEEP": (
                            SourceEvidence(
                                source="KEEP",
                                rule_id="singleton_keep",
                                raw_score=1.0,
                                source_confidence=1.0,
                                calibrated=False,
                                deterministic=True,
                            ),
                        )
                    },
                )
            )

    @staticmethod
    def _merge_unique(left: tuple, right: tuple) -> tuple:
        result = list(left)
        for item in right:
            if item not in result:
                result.append(item)
        return tuple(result)

    def _merge_candidates(self, current: SpanCandidate, incoming: SpanCandidate) -> SpanCandidate:
        sources = dict(current.sources)
        for source, records in incoming.sources.items():
            sources[source] = self._merge_unique(sources.get(source, ()), records)
        return current.clone(
            sources=sources,
            morphology=self._merge_unique(current.morphology, incoming.morphology),
            edit_operations=self._merge_unique(current.edit_operations, incoming.edit_operations),
            dictionary_evidence=self._merge_unique(
                current.dictionary_evidence, incoming.dictionary_evidence
            ),
            suffix_evidence=self._merge_unique(current.suffix_evidence, incoming.suffix_evidence),
            introduced_features=self._merge_unique(
                current.introduced_features, incoming.introduced_features
            ),
            hard_violations=self._merge_unique(current.hard_violations, incoming.hard_violations),
            unsupported_clitic_insertion=(
                current.unsupported_clitic_insertion or incoming.unsupported_clitic_insertion
            ),
            gold=current.gold or incoming.gold,
            keep=current.keep or incoming.keep,
        )

    def _deduplicate(self) -> list[SpanCandidate]:
        merged: dict[tuple[int, int, str], SpanCandidate] = {}
        for candidate in self._proposals:
            key = (
                candidate.raw_span.char_start,
                candidate.raw_span.char_end,
                self.normalize_replacement(candidate.replacement),
            )
            if key in merged:
                merged[key] = self._merge_candidates(merged[key], candidate)
            else:
                merged[key] = candidate
        return list(merged.values())

    @staticmethod
    def _source_tier(candidate: SpanCandidate) -> int:
        if candidate.keep:
            return 0
        names = set(candidate.sources)
        if any(record.deterministic for record in candidate.evidence_records()) or names.intersection(
            {"dictionary", "stage1_exact", "orthographic"}
        ):
            return 1
        if candidate.suffix_evidence or names.intersection({"suffix", "phrase"}):
            return 2
        if names.intersection({"bigru", "neural"}):
            return 3
        if names.intersection({"fuzzy", "symspell"}):
            return 4
        return 5

    @staticmethod
    def _priority(candidate: SpanCandidate) -> tuple[int, float, str]:
        confidences = [
            record.calibrated_confidence
            if record.calibrated and record.calibrated_confidence is not None
            else (
                record.source_confidence
                if record.source_confidence is not None
                else record.raw_score
            )
            for record in candidate.evidence_records()
        ]
        confidence = max((value for value in confidences if value is not None), default=0.0)
        return (CandidateLattice._source_tier(candidate), -confidence, candidate.candidate_id)

    def _prune_per_span(self, candidates: list[SpanCandidate]) -> list[SpanCandidate]:
        grouped: dict[tuple[int, int], list[SpanCandidate]] = defaultdict(list)
        for candidate in candidates:
            grouped[(candidate.raw_span.char_start, candidate.raw_span.char_end)].append(candidate)
        retained: list[SpanCandidate] = []
        quotas = {1: 4, 2: 3, 3: 3, 4: 1}
        for group in grouped.values():
            keeps = [candidate for candidate in group if candidate.keep]
            selected = list(keeps)
            remaining = [candidate for candidate in group if not candidate.keep]
            # Source-specific generation limits remain hard even when another
            # reservation tier leaves spare capacity.
            neural = sorted(
                (candidate for candidate in remaining if self._source_tier(candidate) == 3),
                key=self._priority,
            )[: self.limits.neural_candidates_per_span]
            fuzzy = sorted(
                (candidate for candidate in remaining if self._source_tier(candidate) == 4),
                key=self._priority,
            )[: self.limits.fuzzy_candidates_per_span]
            suffix = sorted(
                (
                    candidate
                    for candidate in remaining
                    if self._source_tier(candidate) == 2
                    and (
                        candidate.suffix_evidence
                        or set(candidate.sources).intersection({"suffix", "suffix_generator"})
                    )
                ),
                key=self._priority,
            )[: self.limits.suffix_candidates_per_span]
            allowed_limited = {
                candidate.candidate_id for candidate in (*neural, *fuzzy, *suffix)
            }
            remaining = [
                candidate
                for candidate in remaining
                if self._source_tier(candidate) not in {3, 4}
                and not (
                    self._source_tier(candidate) == 2
                    and (
                        candidate.suffix_evidence
                        or set(candidate.sources).intersection({"suffix", "suffix_generator"})
                    )
                )
                or candidate.candidate_id in allowed_limited
            ]
            for tier, quota in quotas.items():
                tier_candidates = sorted(
                    (candidate for candidate in remaining if self._source_tier(candidate) == tier),
                    key=self._priority,
                )[:quota]
                selected.extend(tier_candidates)
                selected_ids = {candidate.candidate_id for candidate in tier_candidates}
                remaining = [candidate for candidate in remaining if candidate.candidate_id not in selected_ids]
            capacity = self.limits.candidates_per_span - len(selected)
            if capacity > 0:
                selected.extend(sorted(remaining, key=self._priority)[:capacity])
            retained.extend(selected[: self.limits.candidates_per_span])
        return retained

    def _prune_spans_per_start(self, candidates: list[SpanCandidate]) -> list[SpanCandidate]:
        by_start: dict[int, dict[tuple[int, int, int, int], list[SpanCandidate]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for candidate in candidates:
            span = candidate.raw_span
            key = (span.char_start, span.char_end, span.token_start, span.token_end)
            by_start[span.token_start][key].append(candidate)
        retained: list[SpanCandidate] = []
        for span_groups in by_start.values():
            groups = list(span_groups.values())
            protected = [group for group in groups if any(candidate.keep for candidate in group)]
            others = [group for group in groups if group not in protected]
            others.sort(key=lambda group: min(self._priority(candidate) for candidate in group))
            selected_groups = protected + others[: max(0, self.limits.spans_per_start_token - len(protected))]
            for group in selected_groups:
                retained.extend(group)
        return retained

    def _edge_limits(self) -> tuple[int, int]:
        target = min(
            self.limits.absolute_edge_limit,
            max(self.limits.minimum_edge_limit, self.limits.edges_per_token * len(self.tokens)),
        )
        return target, max(target, len(self.tokens))

    def finalize(self) -> tuple[SpanCandidate, ...]:
        deduplicated = self._deduplicate()
        self._edges_after_deduplication = len(deduplicated)
        candidates = self._prune_per_span(deduplicated)
        candidates = self._prune_spans_per_start(candidates)
        target_limit, effective_limit = self._edge_limits()
        keeps = [candidate for candidate in candidates if candidate.keep]
        nonkeeps = sorted(
            (candidate for candidate in candidates if not candidate.keep), key=self._priority
        )
        candidates = keeps + nonkeeps[: max(0, effective_limit - len(keeps))]
        self._edges = tuple(
            sorted(
                candidates,
                key=lambda candidate: (
                    candidate.raw_span.char_start,
                    candidate.raw_span.char_end,
                    candidate.keep is False,
                    candidate.candidate_id,
                ),
            )
        )
        self._removed_by_pruning = self._edges_after_deduplication - len(self._edges)
        if not self.has_complete_keep_path():
            raise AssertionError("Candidate pruning removed the complete KEEP path.")
        return self._edges

    @property
    def edges(self) -> tuple[SpanCandidate, ...]:
        return self._edges or self.finalize()

    def has_complete_keep_path(self) -> bool:
        if not self.tokens:
            return True
        keeps = {
            (candidate.raw_span.token_start, candidate.raw_span.token_end): candidate
            for candidate in self._edges
            if candidate.keep
        }
        return all(
            (token.index, token.index + 1) in keeps
            and keeps[(token.index, token.index + 1)].replacement == token.text
            for token in self.tokens
        )

    def diagnostics(self) -> LatticeDiagnostics:
        edges = self.edges
        span_counts = Counter(
            f"{candidate.raw_span.char_start}:{candidate.raw_span.char_end}" for candidate in edges
        )
        source_counts = Counter(
            source for candidate in edges for source in candidate.sources
        )
        target_limit, effective_limit = self._edge_limits()
        return LatticeDiagnostics(
            lattice_tokens=len(self.tokens),
            candidate_spans_generated=len(
                {
                    (
                        candidate.raw_span.char_start,
                        candidate.raw_span.char_end,
                        candidate.raw_span.token_start,
                        candidate.raw_span.token_end,
                    )
                    for candidate in self._proposals
                }
            ),
            edges_before_deduplication=len(self._proposals),
            edges_after_deduplication=self._edges_after_deduplication,
            edges_removed_by_pruning=self._removed_by_pruning,
            candidates_per_span=tuple(sorted(span_counts.items())),
            candidate_counts_by_source=tuple(sorted(source_counts.items())),
            complete_keep_path=self.has_complete_keep_path(),
            target_edge_limit=target_limit,
            effective_edge_limit=effective_limit,
        )

    def render(self) -> str:
        lines = [f"RAW: {self.raw.normalized!r}", "TOKENS:"]
        for token in self.tokens:
            lines.append(
                f"  t{token.index} {token.kind.value} [{token.char_start},{token.char_end}) {token.text!r}"
            )
        lines.append("EDGES:")
        for candidate in self.edges:
            span = candidate.raw_span
            lines.append(
                "  "
                f"{candidate.candidate_id} [{span.char_start},{span.char_end}) "
                f"tokens[{span.token_start}:{span.token_end}] {span.text!r} -> "
                f"{candidate.replacement!r} {candidate.operation.value} "
                f"sources={sorted(candidate.sources)}"
            )
        lines.append(f"DIAGNOSTICS: {self.diagnostics()}")
        return "\n".join(lines)
