from __future__ import annotations

import math
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Iterable, Sequence

from .alignment import AlignmentMap, align_texts
from .lattice import CandidateLattice
from .schema import (
    AlignmentOperation,
    CandidateOperation,
    DictionaryAnalysis,
    EditOperation,
    IntroducedFeature,
    LatticeToken,
    MorphologyAnalysis,
    SourceEvidence,
    SpanCandidate,
    SuffixAnalysis,
    TextSpan,
    TokenKind,
)
from .text import grapheme_edit_distance, normalize_for_lattice, span_for_token_range, tokenize_lattice


def edit_operations(source: str, replacement: str) -> tuple[EditOperation, ...]:
    return tuple(
        EditOperation(tag, i1, i2, source[i1:i2], replacement[j1:j2])
        for tag, i1, i2, j1, j2 in SequenceMatcher(
            None, source, replacement, autojunk=False
        ).get_opcodes()
        if tag != "equal"
    )


def _candidate_operation(span: TextSpan, replacement: str) -> CandidateOperation:
    if span.is_boundary:
        return CandidateOperation.BOUNDARY
    if any(character.isspace() for character in span.text) and not any(
        character.isspace() for character in replacement
    ):
        return CandidateOperation.MERGE
    if not any(character.isspace() for character in span.text) and any(
        character.isspace() for character in replacement
    ):
        return CandidateOperation.SPLIT
    output_count = len(tokenize_lattice(replacement))
    input_count = span.token_end - span.token_start
    if input_count > 1 and output_count <= 1:
        return CandidateOperation.MERGE
    if input_count <= 1 and output_count > 1:
        return CandidateOperation.SPLIT
    return CandidateOperation.REPLACE


def _source(
    name: str,
    *,
    rule_id: str,
    score: float | None = None,
    confidence: float | None = None,
    rank: int | None = None,
    deterministic: bool = False,
    details: Iterable[tuple[str, str]] = (),
) -> dict[str, tuple[SourceEvidence, ...]]:
    return {
        name: (
            SourceEvidence(
                source=name,
                rule_id=rule_id,
                raw_score=score,
                source_confidence=confidence,
                calibrated=False,
                rank=rank,
                deterministic=deterministic,
                details=tuple(details),
            ),
        )
    }


def _raw_span_for_char_range(
    raw_text: str,
    tokens: Sequence[LatticeToken],
    start: int,
    end: int,
) -> TextSpan | None:
    covered = [token for token in tokens if token.char_start < end and token.char_end > start]
    if not covered:
        return None
    first, last = covered[0], covered[-1]
    return span_for_token_range(raw_text, tuple(tokens), first.index, last.index + 1)


def _alignment_replacement(record) -> str:
    return "".join(span.text for span in record.s1_spans)


class KeepCandidateAdapter:
    """Explicit KEEP provider; CandidateLattice also protects these edges."""

    def generate_candidates(self, lattice: CandidateLattice) -> list[SpanCandidate]:
        return [candidate for candidate in lattice.edges if candidate.keep]


@dataclass(slots=True)
class Stage1CandidateResult:
    s1_text: str
    alignment: AlignmentMap
    candidates: list[SpanCandidate]
    baseline_candidate_ids: tuple[str, ...]
    production_result: dict


class Stage1CandidateAdapter:
    def __init__(self, spellchecker) -> None:
        self.spellchecker = spellchecker

    def _dictionary_evidence(self, surface: str) -> tuple[DictionaryAnalysis, ...]:
        normalized = self.spellchecker._normalize_word(surface)
        if normalized not in self.spellchecker.dictionary_set:
            return ()
        tags = tuple(sorted(self.spellchecker.word_tags.get(normalized, ())))
        return (
            DictionaryAnalysis(
                entry=surface,
                normalized_surface=normalized,
                tags=tags,
                part_of_speech=tuple(sorted({tag.split("-", 1)[0] for tag in tags})),
                inflectional_tags=tags,
                dictionary="stage1_dictionary_index",
                exact=True,
                confidence=1.0,
            ),
        )

    def _suffix_evidence(self, surface: str) -> tuple[SuffixAnalysis, ...]:
        generator = getattr(self.spellchecker, "suffix_generator", None)
        if generator is None:
            return ()
        return tuple(_generated_suffix_analysis(row, negative=surface.casefold().endswith("x"))
                     for row in generator.exact_suffix_matches(surface))

    def _add_alignment_path(
        self, lattice: CandidateLattice, alignment: AlignmentMap
    ) -> tuple[list[SpanCandidate], list[str]]:
        proposals: list[SpanCandidate] = []
        path_ids: list[str] = []
        for record in alignment.records:
            replacement = _alignment_replacement(record)
            if record.operation == AlignmentOperation.EQUAL:
                for token_index in range(
                    record.raw_span.token_start, record.raw_span.token_end
                ):
                    span = lattice.span(token_index, token_index + 1)
                    candidate = lattice.make_candidate(
                        span=span,
                        replacement=span.text,
                        operation=CandidateOperation.KEEP,
                        keep=True,
                        sources=_source(
                            "stage1", rule_id="stage1_preserved", confidence=record.confidence,
                            deterministic=True, details=(("preserved", "true"),),
                        ),
                        dictionary_evidence=self._dictionary_evidence(span.text),
                        suffix_evidence=self._suffix_evidence(span.text),
                    )
                    proposals.append(candidate)
                    path_ids.append(candidate.candidate_id)
                continue
            span = record.raw_span
            if span.token_end - span.token_start > lattice.limits.maximum_phrase_tokens:
                continue
            candidate = lattice.make_candidate(
                span=span,
                replacement=replacement,
                operation=_candidate_operation(span, replacement),
                sources=_source(
                    "stage1", rule_id=f"stage1_{record.operation.value}",
                    confidence=record.confidence, deterministic=not record.ambiguous,
                    details=(("alignment_ambiguous", str(record.ambiguous).lower()),),
                ),
                dictionary_evidence=self._dictionary_evidence(replacement),
                suffix_evidence=self._suffix_evidence(replacement),
                edit_operations=edit_operations(span.text, replacement),
            )
            proposals.append(candidate)
            path_ids.append(candidate.candidate_id)
        return proposals, path_ids

    def _token_choice_candidates(
        self, raw_text: str, production_tokens: Sequence[dict], lattice: CandidateLattice
    ) -> list[SpanCandidate]:
        proposals: list[SpanCandidate] = []
        cursor = 0
        for token in production_tokens:
            if not isinstance(token, dict) or token.get("type") not in {"word", "phrase"}:
                continue
            original = str(token.get("original", ""))
            if not original:
                continue
            start = raw_text.find(original, cursor)
            if start < 0:
                start = raw_text.find(original)
            if start < 0:
                continue
            end = start + len(original)
            cursor = end
            span = _raw_span_for_char_range(lattice.raw.normalized, lattice.tokens, start, end)
            if span is None or span.token_end - span.token_start > lattice.limits.maximum_phrase_tokens:
                continue
            choices = [token.get("corrected", "")]
            choices.extend(choice.get("word", "") for choice in token.get("choices", ()))
            for rank, replacement in enumerate(dict.fromkeys(str(value) for value in choices if value), 1):
                if replacement == span.text:
                    continue
                recognition = tuple(str(value) for value in token.get("recognition_sources", ()))
                proposals.append(
                    lattice.make_candidate(
                        span=span,
                        replacement=replacement,
                        operation=_candidate_operation(span, replacement),
                        sources=_source(
                            "stage1", rule_id="stage1_token_choice", rank=rank,
                            deterministic=bool(not token.get("ambiguous") and rank == 1),
                            details=tuple(("recognition_source", value) for value in recognition),
                        ),
                        dictionary_evidence=self._dictionary_evidence(replacement),
                        suffix_evidence=self._suffix_evidence(replacement),
                        edit_operations=edit_operations(span.text, replacement),
                        metadata={"unrecognized": bool(token.get("unrecognized")), "crucial": bool(token.get("crucial"))},
                    )
                )
            get_analysis = getattr(self.spellchecker, "_get_token_analysis", None)
            analysis = get_analysis(original) if get_analysis is not None else None
            if analysis is None:
                continue
            intermediate_groups = (
                ("phase_candidates", tuple(getattr(analysis, "candidates", ()))),
                ("basic_candidates", tuple(getattr(analysis, "basic_candidates", ()))),
                ("complex_candidates", tuple(getattr(analysis, "complex_candidates", ()))),
                ("x_candidates", tuple(getattr(analysis, "x_candidates", ()))),
            )
            seen_intermediate: set[tuple[str, str]] = set()
            for group_name, values in intermediate_groups:
                for rank, replacement in enumerate(values, 1):
                    replacement = str(replacement)
                    key = (group_name, replacement)
                    if not replacement or key in seen_intermediate:
                        continue
                    seen_intermediate.add(key)
                    proposals.append(lattice.make_candidate(
                        span=span,
                        replacement=replacement,
                        operation=_candidate_operation(span, replacement),
                        sources=_source(
                            "stage1",
                            rule_id=f"stage1_{getattr(analysis, 'phase', 'unknown')}_{group_name}",
                            rank=rank,
                            deterministic=bool(getattr(analysis, "is_deterministic", False)),
                            details=(("intermediate_proposal", "true"),),
                        ),
                        dictionary_evidence=self._dictionary_evidence(replacement),
                        suffix_evidence=self._suffix_evidence(replacement),
                        edit_operations=edit_operations(span.text, replacement),
                    ))
        return proposals

    def generate_candidates(
        self,
        raw_text: str,
        s1_text: str | None = None,
        alignment: AlignmentMap | None = None,
        *,
        lattice: CandidateLattice | None = None,
        production_result: dict | None = None,
    ) -> Stage1CandidateResult:
        raw = normalize_for_lattice(raw_text)
        result = production_result or self.spellchecker.correct_text_rich(raw_text)
        actual_s1 = s1_text if s1_text is not None else str(result["corrected_text"])
        alignment = alignment or align_texts(raw, normalize_for_lattice(actual_s1))
        lattice = lattice or CandidateLattice(sentence_id=f"raw:{raw.normalized}", raw=raw, s1_alignment=alignment)
        proposals, path_ids = self._add_alignment_path(lattice, alignment)
        proposals.extend(self._token_choice_candidates(raw.normalized, result.get("tokens", ()), lattice))
        return Stage1CandidateResult(actual_s1, alignment, proposals, tuple(path_ids), result)


class DictionaryCandidateAdapter:
    def __init__(self, spellchecker=None, dictionary_index=None, *, include_fuzzy: bool = True) -> None:
        self.spellchecker = spellchecker
        self.dictionary_index = dictionary_index
        self.include_fuzzy = include_fuzzy

    def _lookup(self, surface: str):
        if self.dictionary_index is not None:
            return self.dictionary_index.lookup(surface)
        normalized = self.spellchecker._normalize_word(surface)
        if normalized not in self.spellchecker.dictionary_set:
            return None
        return normalized

    def _analysis(self, surface: str, match, *, exact: bool) -> DictionaryAnalysis:
        if self.dictionary_index is not None:
            tags = (f"tag_bits:{match.tag_bits}",)
            return DictionaryAnalysis(
                entry=match.canonical, normalized_surface=match.canonical.casefold(),
                tags=tags, inflectional_tags=tags,
                dictionary=f"sqlite_source_bits:{match.source_bits}", exact=exact,
                confidence=1.0 if exact else None,
            )
        normalized = self.spellchecker._normalize_word(surface)
        tags = tuple(sorted(self.spellchecker.word_tags.get(normalized, ())))
        return DictionaryAnalysis(
            entry=surface, normalized_surface=normalized, tags=tags,
            part_of_speech=tuple(sorted({tag.split("-", 1)[0] for tag in tags})),
            inflectional_tags=tags, dictionary="spellchecker_dictionary_index",
            exact=exact, confidence=1.0 if exact else None,
        )

    def generate_candidates(self, text: str, lattice: CandidateLattice) -> list[SpanCandidate]:
        proposals: list[SpanCandidate] = []
        for token in lattice.tokens:
            if token.kind != TokenKind.WORD:
                continue
            span = lattice.span(token.index, token.index + 1)
            exact = self._lookup(token.text)
            if exact is not None:
                replacement = exact.canonical if self.dictionary_index is not None else token.text
                proposals.append(lattice.make_candidate(
                    span=span, replacement=replacement, operation=CandidateOperation.KEEP,
                    keep=True, sources=_source("dictionary", rule_id="exact_lexical_match", deterministic=True),
                    dictionary_evidence=(self._analysis(replacement, exact, exact=True),),
                ))
            if not self.include_fuzzy:
                continue
            nearby = ()
            if self.dictionary_index is not None:
                nearby = self.dictionary_index.nearby(token.text, max_distance=2, limit=2)
            elif hasattr(self.spellchecker, "_symspell_candidates"):
                nearby = tuple((word, None) for word in self.spellchecker._symspell_candidates(token.text, limit=2))
            for rank, (entry, distance) in enumerate(nearby[:2], 1):
                replacement = entry.canonical if self.dictionary_index is not None else entry
                if replacement.casefold() == token.text.casefold():
                    continue
                lookup = entry if self.dictionary_index is not None else self._lookup(replacement)
                proposals.append(lattice.make_candidate(
                    span=span, replacement=replacement, operation=CandidateOperation.REPLACE,
                    sources=_source("fuzzy", rule_id="bounded_dictionary_retrieval", rank=rank,
                                    score=None if distance is None else -float(distance)),
                    dictionary_evidence=(self._analysis(replacement, lookup, exact=False),) if lookup is not None else (),
                    edit_operations=edit_operations(span.text, replacement),
                    metadata={"grapheme_edit_distance": grapheme_edit_distance(span.text, replacement)},
                ))
        return proposals


class NeuralCandidateAdapter:
    """Expose pre-validation BiGRU hypotheses without invoking ``correct()``."""

    def __init__(self, neural_corrector) -> None:
        self.corrector = neural_corrector

    def generate_candidates(self, text: str, top_k: int = 3) -> list[SpanCandidate]:
        from neural_corrector.models.alignment import COPY_ACTION, apply_actions

        if not text or top_k <= 0:
            return []
        position_candidates: list[list[tuple[str, float]]] = []
        for start in range(0, len(text), self.corrector.max_length):
            chunk = text[start : start + self.corrector.max_length]
            _actions, _confidences, chunk_candidates = self.corrector._predict_chunk(chunk)
            position_candidates.extend(chunk_candidates)

        primary = [rows[0][0] for rows in position_candidates]
        hypotheses: dict[str, tuple[float, tuple[tuple[int, str, float], ...]]] = {}

        def add_hypothesis(actions, changed) -> None:
            rendered = apply_actions(text, actions)
            probabilities = [row[2] for row in changed]
            score = (
                math.exp(
                    sum(math.log(max(value, 1e-12)) for value in probabilities)
                    / len(probabilities)
                )
                if probabilities
                else 1.0
            )
            previous = hypotheses.get(rendered)
            evidence = tuple(changed)
            if previous is None or score > previous[0]:
                hypotheses[rendered] = (score, evidence)

        add_hypothesis(
            primary,
            tuple(
                (index, action, rows[0][1])
                for index, (action, rows) in enumerate(zip(primary, position_candidates))
                if action != COPY_ACTION
            ),
        )
        alternatives = [
            (probability, index, action)
            for index, rows in enumerate(position_candidates)
            for action, probability in rows[1:]
            if action != primary[index]
        ]
        for probability, index, action in sorted(alternatives, reverse=True):
            actions = list(primary)
            actions[index] = action
            add_hypothesis(actions, ((index, action, probability),))
            if len(hypotheses) >= max(top_k * 4, top_k + 1):
                break

        ranked = sorted(
            (
                (rendered, score, evidence)
                for rendered, (score, evidence) in hypotheses.items()
            ),
            key=lambda item: (-item[1], item[0]),
        )[:top_k]
        raw = normalize_for_lattice(text)
        from .pipeline import sentence_id_for_text

        lattice = CandidateLattice(
            sentence_id=sentence_id_for_text(raw.normalized), raw=raw
        )
        proposals: list[SpanCandidate] = []
        for rank, (rendered, score, action_evidence) in enumerate(ranked, 1):
            if rendered == raw.normalized:
                continue
            alignment = align_texts(raw, normalize_for_lattice(rendered))
            for record in alignment.records:
                if record.operation == AlignmentOperation.EQUAL:
                    continue
                span = record.raw_span
                if span.token_end - span.token_start > lattice.limits.maximum_phrase_tokens:
                    continue
                replacement = _alignment_replacement(record)
                proposals.append(lattice.make_candidate(
                    span=span,
                    replacement=replacement,
                    operation=_candidate_operation(span, replacement),
                    sources=_source(
                        "bigru",
                        rule_id="pre_validation_sequence_hypothesis",
                        score=score,
                        confidence=score,
                        rank=rank,
                        details=(
                            ("model_version", str(self.corrector.model_version)),
                            ("hypothesis", rendered),
                        ),
                    ),
                    edit_operations=edit_operations(span.text, replacement),
                    metadata={
                        "neural_rank": rank,
                        "raw_neural_score": score,
                        "normalized_candidate_probability": None,
                        "grapheme_edit_distance": grapheme_edit_distance(span.text, replacement),
                        "word_boundaries_change": (
                            span.token_end - span.token_start
                            != len(record.s1_spans)
                        ),
                        "action_evidence": action_evidence,
                        "calibrated": False,
                    },
                ))
        return proposals


def _split_suffix_person(kind: str, person: str) -> tuple[str | None, str | None]:
    if kind == "DO_IDO":
        left, _, right = person.partition("+")
        return left or None, right or None
    if kind == "DO":
        return person or None, None
    if kind == "IDO":
        return None, person or None
    return None, None


def _generated_suffix_analysis(row, *, negative: bool = False) -> SuffixAnalysis:
    direct, indirect = _split_suffix_person(row.suffix_kind, row.suffix_person)
    return SuffixAnalysis(
        lemma=row.base, surface=row.surface, root_or_stem=row.root,
        paradigm=row.form_class, tense_or_mood=row.tense,
        subject_person=row.person[:1] if row.person else None,
        subject_number=row.person[1:2] if len(row.person) > 1 else None,
        subject_gender=row.person[2:3] if len(row.person) > 2 else None,
        direct_object=direct, indirect_object=indirect,
        has_direct_and_indirect_object=row.suffix_kind == "DO_IDO",
        negative=negative, surface_valid=True, validity_source="suffix_generator",
        rule_id=row.rule_id,
    )


def _verb_record_analysis(record, surface: str) -> SuffixAnalysis:
    return SuffixAnalysis(
        lemma=record.word, surface=surface, root_or_stem=record.root,
        paradigm=record.form_class, tense_or_mood=record.tense,
        subject_person=record.person[:1] if record.person else None,
        subject_number=record.person[1:2] if len(record.person) > 1 else None,
        subject_gender=record.person[2:3] if len(record.person) > 2 else None,
        surface_valid=True, validity_source="verb_form_index", rule_id="unsuffixed_paradigm",
    )


class SuffixCandidateAdapter:
    def __init__(self, suffix_generator, spellchecker=None) -> None:
        self.generator = suffix_generator
        self.spellchecker = spellchecker or suffix_generator.spellchecker

    def _orthographic_verb_surfaces(self, word: str) -> list[str]:
        orthographic = getattr(self.spellchecker, "orthographic_generator", None)
        if orthographic is None:
            return []
        variants: list[str] = []
        helpers = (
            "dictionary_shortcut_variants", "dictionary_gh_priority_variants",
            "dictionary_final_aw_to_ghu_variants", "dictionary_i_ie_variants",
            "dictionary_d_t_variants", "dictionary_b_p_variants",
            "dictionary_g_k_cluster_variants",
        )
        for name in helpers:
            helper = getattr(orthographic, name, None)
            if helper is not None:
                variants.extend(helper(word))
        normalized = self.spellchecker._normalize_word(word)
        if normalized.endswith("aw") and "gh" in normalized:
            combined = normalized[:-2].replace("gh", "għ") + "u"
            if combined in self.spellchecker.dictionary_set:
                variants.append(combined)
        return list(dict.fromkeys(variants))

    def _candidate(self, lattice, span, replacement, analyses, *, rule_id, introduced=()):
        introduced_features = tuple(
            IntroducedFeature(category, value, input_evidence=False)
            for category, value in introduced
        )
        return lattice.make_candidate(
            span=span, replacement=replacement, operation=CandidateOperation.REPLACE,
            sources=_source("suffix", rule_id=rule_id), suffix_evidence=tuple(analyses),
            introduced_features=introduced_features,
            unsupported_clitic_insertion=any(category in {"DO", "IDO"} for category, _ in introduced),
            edit_operations=edit_operations(span.text, replacement),
        )

    @staticmethod
    def _features_for_kind(kind: str, person: str) -> set[tuple[str, str]]:
        direct, indirect = _split_suffix_person(kind, person)
        features: set[tuple[str, str]] = set()
        if direct:
            features.add(("DO", direct))
        if indirect:
            features.add(("IDO", indirect))
        return features

    def generate_candidates(
        self, text: str, tokens: Sequence[LatticeToken], *, lattice: CandidateLattice
    ) -> list[SpanCandidate]:
        proposals: list[SpanCandidate] = []
        for token in tokens:
            if token.kind != TokenKind.WORD:
                continue
            span = lattice.span(token.index, token.index + 1)
            input_features: set[tuple[str, str]] = set()
            for parsed in self.generator.parse_possible_suffixes(token.text):
                input_features.update(
                    self._features_for_kind(parsed.spec.kind, parsed.spec.person)
                )
            exact_rows = self.generator.exact_suffix_matches(token.text)
            if exact_rows:
                proposals.append(self._candidate(
                    lattice, span, token.text,
                    [_generated_suffix_analysis(row, negative=token.text.casefold().endswith("x")) for row in exact_rows],
                    rule_id="exact_generated_suffix",
                ))
            suggestions = self.generator.suggest_suffixes(token.text, limit=4)
            for replacement in suggestions[:4]:
                rows = self.generator.candidates_for_surface(replacement, limit=16)
                if rows:
                    output_features = set().union(
                        *(
                            self._features_for_kind(row.suffix_kind, row.suffix_person)
                            for row in rows
                        )
                    )
                    proposals.append(self._candidate(
                        lattice, span, replacement,
                        [_generated_suffix_analysis(row, negative=replacement.casefold().endswith("x")) for row in rows],
                        rule_id="suffix_repair",
                        introduced=tuple(sorted(output_features - input_features)),
                    ))
            for replacement in self._orthographic_verb_surfaces(token.text):
                records = self.generator.verb_index.word_records(replacement)
                if not records:
                    continue
                proposals.append(self._candidate(
                    lattice, span, replacement,
                    [_verb_record_analysis(record, replacement) for record in records],
                    rule_id="orthographic_unsuffixed_paradigm",
                ))
                suffixed = replacement + "h"
                suffixed_rows = self.generator.candidates_for_surface(suffixed, limit=16)
                if suffixed_rows:
                    proposals.append(self._candidate(
                        lattice, span, suffixed,
                        [_generated_suffix_analysis(row) for row in suffixed_rows],
                        rule_id="orthographic_paradigm_with_introduced_clitic",
                        introduced=(("DO", "3SM"),),
                    ))
        return proposals


class PhraseOrthographicCandidateAdapter:
    PHRASE_RULES = (
        (re.compile(r"(?i)(?<!\w)ma\s+hawnx(?!\w)"), "m'hawnx", "contract_ma_hawnx"),
        (re.compile(r"(?i)(?<!\w)il\s+lejla(?!\w)"), "illejla", "join_illejla"),
        (re.compile(r"(?i)(?<!\w)illejla\s+tal-festa(?!\w)"), "il-lejla tal-festa", "article_illejla"),
        (re.compile(r"(?i)(?<!\w)daqs\s+li\s+kieku(?!\w)"), "daqslikieku", "join_daqslikieku"),
    )

    def __init__(self, spellchecker=None) -> None:
        self.spellchecker = spellchecker

    def generate_candidates(self, text: str, lattice: CandidateLattice) -> list[SpanCandidate]:
        proposals: list[SpanCandidate] = []
        for pattern, replacement, rule_id in self.PHRASE_RULES:
            for match in pattern.finditer(lattice.raw.normalized):
                span = _raw_span_for_char_range(lattice.raw.normalized, lattice.tokens, match.start(), match.end())
                if span is None or span.token_end - span.token_start > lattice.limits.maximum_phrase_tokens:
                    continue
                proposals.append(lattice.make_candidate(
                    span=span, replacement=replacement, operation=_candidate_operation(span, replacement),
                    sources=_source("phrase", rule_id=rule_id, deterministic=False),
                    edit_operations=edit_operations(span.text, replacement),
                ))
        if self.spellchecker is None:
            return proposals
        article_rules = getattr(self.spellchecker, "article_phrase_rules", None)
        lexical_tokens = [token for token in lattice.tokens if token.kind == TokenKind.WORD]
        article_prefixes = {
            "il", "l", "fi", "fil", "fl", "bi", "bil", "bl", "ma", "mal",
            "ta", "tal", "min", "minn", "għal", "ghal", "għall", "ghall",
            "bħal", "bhal", "bħall", "bhall", "ġol", "gol",
        }
        for left, right in zip(lexical_tokens, lexical_tokens[1:]):
            gap = lattice.raw.normalized[left.char_end:right.char_start]
            if not gap or not gap.isspace():
                continue
            left_norm = self.spellchecker._normalize_word(left.text)
            span = lattice.span(left.index, right.index + 1)
            if article_rules is not None and left_norm in article_prefixes:
                replacement = article_rules.preposition_article_form(left_norm, right.text)
                if replacement and replacement != span.text:
                    proposals.append(lattice.make_candidate(
                        span=span, replacement=replacement,
                        operation=_candidate_operation(span, replacement),
                        sources=_source(
                            "phrase", rule_id="article_preposition_surface",
                            details=(("contextual_decision_pending", "true"),),
                        ),
                        edit_operations=edit_operations(span.text, replacement),
                    ))
            is_verb = getattr(self.spellchecker, "_is_verb_tagged_word", lambda _word: False)
            if left_norm == "ma" and is_verb(right.text):
                proposals.append(lattice.make_candidate(
                    span=span, replacement=span.text, operation=CandidateOperation.REPLACE,
                    sources=_source(
                        "phrase", rule_id="negative_ma_verb_compatibility",
                        details=(("compatibility_evidence_only", "true"),),
                    ),
                ))
        orthographic = getattr(self.spellchecker, "orthographic_generator", None)
        for token in lattice.tokens:
            if token.kind != TokenKind.WORD:
                continue
            span = lattice.span(token.index, token.index + 1)
            normalized = self.spellchecker._normalize_word(token.text)
            variants: list[tuple[str, str]] = []
            if normalized.startswith("x") and not normalized.startswith("x'") and len(normalized) > 1:
                corrected_tail = self.spellchecker.correct_word(normalized[1:])
                if corrected_tail != normalized[1:]:
                    variants.append((f"x'{corrected_tail}", "split_x_apostrophe"))
            if normalized.startswith("mgh") and len(normalized) > 3:
                tail = self.spellchecker.correct_word(normalized[1:])
                variants.append((f"ma {tail}", "expand_negative_ma"))
            if orthographic is not None:
                for method_name in (
                    "dictionary_shortcut_variants", "dictionary_gh_priority_variants",
                    "dictionary_i_ie_variants", "dictionary_final_aw_to_ghu_variants",
                    "dictionary_d_t_variants", "dictionary_b_p_variants",
                    "dictionary_g_k_cluster_variants",
                ):
                    helper = getattr(orthographic, method_name, None)
                    if helper is not None:
                        variants.extend((value, method_name) for value in helper(normalized))
            for replacement, rule_id in dict.fromkeys(variants):
                if replacement == span.text:
                    continue
                proposals.append(lattice.make_candidate(
                    span=span, replacement=replacement, operation=_candidate_operation(span, replacement),
                    sources=_source("orthographic", rule_id=rule_id),
                    edit_operations=edit_operations(span.text, replacement),
                ))
        return proposals
