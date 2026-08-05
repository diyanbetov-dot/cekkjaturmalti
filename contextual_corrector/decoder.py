from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Callable

from .costs import (
    EditCostWeights,
    SoftEditBudget,
    candidate_edit_cost,
    incremental_edit_cost,
)
from .lattice import CandidateLattice
from .pipeline import apply_candidate_path
from .schema import (
    CandidateOperation,
    CandidateValidation,
    EditCost,
    SourceEvidence,
    SpanCandidate,
)
from .validation import ValidationResult, validate_lattice


CandidateScoreFunction = Callable[[SpanCandidate], float]


@dataclass(frozen=True, slots=True)
class PrunedState:
    position: int
    candidate_ids: tuple[str, ...]
    reason: str
    objective: float


@dataclass(frozen=True, slots=True)
class PathStep:
    candidate_id: str
    char_span: tuple[int, int]
    token_span: tuple[int, int]
    replacement: str
    sources: tuple[str, ...]
    source_evidence: tuple[SourceEvidence, ...]
    candidate_score: float
    edit_cost: EditCost
    incremental_edit_cost: float
    validation: CandidateValidation


@dataclass(frozen=True, slots=True)
class DecoderDiagnostics:
    selected_candidate_ids: tuple[str, ...]
    selected_steps: tuple[PathStep, ...]
    hard_invalid_candidates_excluded: tuple[CandidateValidation, ...]
    soft_penalties_by_candidate: tuple[tuple[str, float], ...]
    cumulative_edit_cost: float
    edit_budget_penalty: float
    beam_states_expanded: int
    states_pruned: int
    maximum_beam_size: int
    keep_fallback_used: bool
    input_coverage: tuple[int, ...]
    boundary_operations_selected: tuple[str, ...]
    pruned_states: tuple[PrunedState, ...]


@dataclass(frozen=True, slots=True)
class DecodedPath:
    candidates: tuple[SpanCandidate, ...]
    rendered_text: str
    candidate_score: float
    cumulative_edit_cost: float
    objective: float
    morphological_state: tuple[str, ...]
    diagnostics: DecoderDiagnostics

    def format(self) -> str:
        lines = [f"OUTPUT: {self.rendered_text}", "PATH:"]
        for step in self.diagnostics.selected_steps:
            lines.append(
                f"  {step.candidate_id} tokens{step.token_span} -> {step.replacement!r} "
                f"score={step.candidate_score:.3f} cost={step.incremental_edit_cost:.3f} "
                f"sources={','.join(step.sources)}"
            )
        lines.extend(
            (
                f"TOTAL SCORE: {self.candidate_score:.3f}",
                f"TOTAL EDIT COST: {self.cumulative_edit_cost:.3f}",
                f"OBJECTIVE: {self.objective:.3f}",
                f"COVERAGE: {self.diagnostics.input_coverage}",
            )
        )
        return "\n".join(lines)


@dataclass(slots=True)
class _BeamState:
    position: int
    candidates: tuple[SpanCandidate, ...] = ()
    candidate_score: float = 0.0
    edit_cost: float = 0.0
    signatures: Counter[str] = field(default_factory=Counter)
    boundary_ids: frozenset[str] = frozenset()
    morphology: tuple[str, ...] = ()
    steps: tuple[PathStep, ...] = ()


def _morphology_labels(candidate: SpanCandidate) -> tuple[str, ...]:
    if candidate.suffix_evidence:
        analysis = candidate.suffix_evidence[0]
        return tuple(
            value
            for value in (
                f"lemma={analysis.lemma}" if analysis.lemma else None,
                f"tense={analysis.tense_or_mood}" if analysis.tense_or_mood else None,
                f"subject={analysis.subject_person or ''}{analysis.subject_number or ''}{analysis.subject_gender or ''}",
                f"DO={analysis.direct_object}" if analysis.direct_object else None,
                f"IDO={analysis.indirect_object}" if analysis.indirect_object else None,
            )
            if value
        )
    if candidate.morphology:
        analysis = candidate.morphology[0]
        return tuple(analysis.part_of_speech) + tuple(analysis.features)
    return ()


def _state_objective(state: _BeamState, budget: SoftEditBudget, token_count: int) -> float:
    return state.candidate_score - state.edit_cost - budget.penalty(state.edit_cost, token_count)


def _extend(
    state: _BeamState,
    candidate: SpanCandidate,
    validation: CandidateValidation,
    score_fn: CandidateScoreFunction,
    weights: EditCostWeights,
) -> _BeamState:
    cost = candidate_edit_cost(candidate, validation, weights=weights)
    incremental = incremental_edit_cost(cost, state.signatures, weights=weights)
    signatures = state.signatures.copy()
    signatures.update(cost.coherent_signatures)
    score = float(score_fn(candidate))
    boundary_ids = state.boundary_ids
    position = candidate.raw_span.token_end
    if candidate.operation == CandidateOperation.BOUNDARY:
        boundary_ids = frozenset((*boundary_ids, candidate.candidate_id))
        position = state.position
    step = PathStep(
        candidate_id=candidate.candidate_id,
        char_span=(candidate.raw_span.char_start, candidate.raw_span.char_end),
        token_span=(candidate.raw_span.token_start, candidate.raw_span.token_end),
        replacement=candidate.replacement,
        sources=tuple(sorted(candidate.sources)),
        source_evidence=candidate.evidence_records(),
        candidate_score=score,
        edit_cost=cost,
        incremental_edit_cost=incremental,
        validation=validation,
    )
    return _BeamState(
        position=position,
        candidates=state.candidates + (candidate,),
        candidate_score=state.candidate_score + score,
        edit_cost=state.edit_cost + incremental,
        signatures=signatures,
        boundary_ids=boundary_ids,
        morphology=state.morphology + _morphology_labels(candidate),
        steps=state.steps + (step,),
    )


def _prune(
    states: list[_BeamState],
    *,
    position: int,
    beam_width: int,
    budget: SoftEditBudget,
    token_count: int,
    pruned: list[PrunedState],
) -> list[_BeamState]:
    states.sort(
        key=lambda state: (
            -_state_objective(state, budget, token_count),
            state.edit_cost,
            tuple(candidate.candidate_id for candidate in state.candidates),
        )
    )
    for state in states[beam_width:]:
        pruned.append(
            PrunedState(
                position=position,
                candidate_ids=tuple(candidate.candidate_id for candidate in state.candidates),
                reason="BEAM_WIDTH",
                objective=_state_objective(state, budget, token_count),
            )
        )
    return states[:beam_width]


def _keep_fallback(
    lattice: CandidateLattice,
    validation_by_id: dict[str, CandidateValidation],
    score_fn: CandidateScoreFunction,
    weights: EditCostWeights,
) -> _BeamState:
    state = _BeamState(position=0)
    keeps = sorted(
        (candidate for candidate in lattice.edges if candidate.keep),
        key=lambda candidate: candidate.raw_span.token_start,
    )
    for candidate in keeps:
        if candidate.raw_span.token_start != state.position:
            raise ValueError("KEEP fallback does not cover every input token.")
        state = _extend(state, candidate, validation_by_id[candidate.candidate_id], score_fn, weights)
    return state


def decode_lattice(
    lattice: CandidateLattice,
    candidate_score_fn: CandidateScoreFunction,
    beam_width: int = 24,
    edit_budget: SoftEditBudget | None = None,
    *,
    validation: ValidationResult | None = None,
    suffix_generator=None,
    cost_weights: EditCostWeights | None = None,
) -> DecodedPath:
    if beam_width <= 0:
        raise ValueError("Beam width must be positive.")
    budget = edit_budget or SoftEditBudget()
    weights = cost_weights or EditCostWeights()
    validation = validation or validate_lattice(
        lattice, suffix_generator=suffix_generator
    )
    validation_by_id = {record.candidate_id: record for record in validation.records}
    eligible = tuple(
        candidate
        for candidate in lattice.edges
        if validation_by_id[candidate.candidate_id].decodable
    )
    base_penalties = tuple(
        (
            candidate.candidate_id,
            candidate_edit_cost(
                candidate, validation_by_id[candidate.candidate_id], weights=weights
            ).total,
        )
        for candidate in eligible
    )
    lexical_by_start: dict[int, list[SpanCandidate]] = defaultdict(list)
    boundaries_by_position: dict[int, list[SpanCandidate]] = defaultdict(list)
    for candidate in eligible:
        if candidate.raw_span.is_boundary:
            boundaries_by_position[candidate.raw_span.token_start].append(candidate)
        else:
            lexical_by_start[candidate.raw_span.token_start].append(candidate)
    for candidates in (*lexical_by_start.values(), *boundaries_by_position.values()):
        candidates.sort(key=lambda candidate: candidate.candidate_id)

    token_count = len(lattice.tokens)
    buckets: dict[int, list[_BeamState]] = defaultdict(list)
    buckets[0].append(_BeamState(position=0))
    finals: list[_BeamState] = []
    pruned: list[PrunedState] = []
    expanded = 0
    maximum_beam = 1

    for position in range(token_count + 1):
        states = _prune(
            buckets.get(position, []),
            position=position,
            beam_width=beam_width,
            budget=budget,
            token_count=token_count,
            pruned=pruned,
        )
        maximum_beam = max(maximum_beam, len(states))
        for state in states:
            boundary_options: list[_BeamState] = [state]
            for boundary in boundaries_by_position.get(position, ()):
                if boundary.candidate_id in state.boundary_ids:
                    continue
                boundary_options.append(
                    _extend(
                        state,
                        boundary,
                        validation_by_id[boundary.candidate_id],
                        candidate_score_fn,
                        weights,
                    )
                )
                expanded += 1
            for prepared in boundary_options:
                if position == token_count:
                    finals.append(prepared)
                    continue
                advancing = lexical_by_start.get(position, ())
                if not advancing:
                    pruned.append(
                        PrunedState(
                            position=position,
                            candidate_ids=tuple(
                                candidate.candidate_id for candidate in prepared.candidates
                            ),
                            reason="UNCOVERED_LEXICAL_INPUT",
                            objective=_state_objective(prepared, budget, token_count),
                        )
                    )
                    continue
                for candidate in advancing:
                    if candidate.raw_span.token_end <= position:
                        pruned.append(
                            PrunedState(
                                position=position,
                                candidate_ids=tuple(
                                    row.candidate_id for row in prepared.candidates + (candidate,)
                                ),
                                reason="NON_ADVANCING_OR_OVERLAPPING_EDGE",
                                objective=_state_objective(prepared, budget, token_count),
                            )
                        )
                        continue
                    buckets[candidate.raw_span.token_end].append(
                        _extend(
                            prepared,
                            candidate,
                            validation_by_id[candidate.candidate_id],
                            candidate_score_fn,
                            weights,
                        )
                    )
                    expanded += 1

    keep_fallback_used = False
    if not finals:
        finals = [_keep_fallback(lattice, validation_by_id, candidate_score_fn, weights)]
        keep_fallback_used = True
    finals = _prune(
        finals,
        position=token_count,
        beam_width=beam_width,
        budget=budget,
        token_count=token_count,
        pruned=pruned,
    )
    winner = finals[0]
    lexical = [candidate for candidate in winner.candidates if not candidate.raw_span.is_boundary]
    coverage = tuple(
        index
        for candidate in lexical
        for index in range(candidate.raw_span.token_start, candidate.raw_span.token_end)
    )
    if coverage != tuple(range(token_count)):
        raise AssertionError("Decoded path does not cover every RAW token exactly once.")
    ordered_candidates = tuple(
        sorted(
            winner.candidates,
            key=lambda candidate: (
                candidate.raw_span.char_start,
                candidate.raw_span.char_end,
                candidate.operation != CandidateOperation.BOUNDARY,
            ),
        )
    )
    rendered = apply_candidate_path(lattice.raw.normalized, ordered_candidates)
    budget_penalty = budget.penalty(winner.edit_cost, token_count)
    diagnostics = DecoderDiagnostics(
        selected_candidate_ids=tuple(candidate.candidate_id for candidate in ordered_candidates),
        selected_steps=winner.steps,
        hard_invalid_candidates_excluded=validation.hard_invalid,
        soft_penalties_by_candidate=base_penalties,
        cumulative_edit_cost=winner.edit_cost,
        edit_budget_penalty=budget_penalty,
        beam_states_expanded=expanded,
        states_pruned=len(pruned),
        maximum_beam_size=maximum_beam,
        keep_fallback_used=keep_fallback_used,
        input_coverage=coverage,
        boundary_operations_selected=tuple(
            candidate.candidate_id
            for candidate in ordered_candidates
            if candidate.operation == CandidateOperation.BOUNDARY
        ),
        pruned_states=tuple(pruned),
    )
    return DecodedPath(
        candidates=ordered_candidates,
        rendered_text=rendered,
        candidate_score=winner.candidate_score,
        cumulative_edit_cost=winner.edit_cost,
        objective=winner.candidate_score - winner.edit_cost - budget_penalty,
        morphological_state=winner.morphology,
        diagnostics=diagnostics,
    )
