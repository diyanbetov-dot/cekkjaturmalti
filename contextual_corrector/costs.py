from __future__ import annotations

import math
import unicodedata
from collections import Counter
from dataclasses import dataclass

from .schema import CandidateOperation, CandidateValidation, EditCost, SpanCandidate
from .text import grapheme_edit_distance, tokenize_lattice


@dataclass(frozen=True, slots=True)
class EditCostWeights:
    grapheme: float = 0.20
    boundary: float = 0.45
    punctuation: float = 0.55
    lexical: float = 1.25
    morphology: float = 0.35
    unsupported_clitic: float = 2.50
    repeated_signature_factor: float = 0.35


@dataclass(frozen=True, slots=True)
class SoftEditBudget:
    base_allowance: float = 0.75
    per_token_allowance: float = 0.28
    excess_penalty: float = 0.45

    def allowance(self, token_count: int) -> float:
        return self.base_allowance + self.per_token_allowance * math.sqrt(max(token_count, 0))

    def penalty(self, cumulative_cost: float, token_count: int) -> float:
        return self.excess_penalty * max(0.0, cumulative_cost - self.allowance(token_count)) ** 2


def _punctuation_count(text: str) -> int:
    return sum(unicodedata.category(character).startswith("P") for character in text)


def _ascii_shape(text: str) -> str:
    value = unicodedata.normalize("NFC", text).casefold()
    for source, replacement in (
        ("għ", "gh"),
        ("ħ", "h"),
        ("ċ", "c"),
        ("ġ", "g"),
        ("ż", "z"),
    ):
        value = value.replace(source, replacement)
    return value


def coherent_edit_signatures(source: str, replacement: str) -> tuple[str, ...]:
    source_folded = source.casefold()
    replacement_folded = replacement.casefold()
    signatures: list[str] = []
    pairs = (
        ("gh", "għ", "gh->għ"),
        ("h", "ħ", "h->ħ"),
        ("c", "ċ", "c->ċ"),
        ("g", "ġ", "g->ġ"),
        ("z", "ż", "z->ż"),
    )
    for raw, corrected, label in pairs:
        difference = max(0, replacement_folded.count(corrected) - source_folded.count(corrected))
        available = source_folded.count(raw)
        signatures.extend([label] * min(difference, available))
    if any(character.isspace() for character in source) and not any(
        character.isspace() for character in replacement
    ):
        signatures.append("spacing_merge")
    if not any(character.isspace() for character in source) and any(
        character.isspace() for character in replacement
    ):
        signatures.append("spacing_split")
    if replacement.count("'") + replacement.count("’") > source.count("'") + source.count("’"):
        signatures.append("apostrophe_insertion")
    if replacement.count("-") > source.count("-"):
        signatures.append("hyphen_insertion")
    return tuple(signatures)


def candidate_edit_cost(
    candidate: SpanCandidate,
    validation: CandidateValidation,
    *,
    weights: EditCostWeights | None = None,
) -> EditCost:
    weights = weights or EditCostWeights()
    if candidate.keep:
        return EditCost()
    source = candidate.raw_span.text
    replacement = candidate.replacement
    signatures = coherent_edit_signatures(source, replacement)
    input_tokens = candidate.raw_span.token_end - candidate.raw_span.token_start
    output_tokens = len(tokenize_lattice(replacement))
    boundary_changes = abs(output_tokens - input_tokens)
    if candidate.operation == CandidateOperation.BOUNDARY:
        boundary_changes += 1
    punctuation_changes = abs(_punctuation_count(replacement) - _punctuation_count(source))
    morphology_changes = (
        len(validation.feature_delta.introduced)
        + len(validation.feature_delta.removed)
        + len(validation.feature_delta.changed)
    )
    shape_preserved = _ascii_shape(source) == _ascii_shape(replacement)
    structural_only = shape_preserved or (
        signatures
        and all(
            signature
            in {
                "gh->għ",
                "h->ħ",
                "c->ċ",
                "g->ġ",
                "z->ż",
                "spacing_merge",
                "spacing_split",
                "apostrophe_insertion",
                "hyphen_insertion",
            }
            for signature in signatures
        )
    )
    return EditCost(
        grapheme=weights.grapheme * grapheme_edit_distance(source, replacement),
        boundary=weights.boundary * boundary_changes,
        punctuation=weights.punctuation * punctuation_changes,
        lexical=0.0 if structural_only else weights.lexical,
        morphology=weights.morphology * morphology_changes,
        unsupported_clitic=(
            weights.unsupported_clitic
            if validation.clitic_evidence.unsupported_clitic_insertion
            else 0.0
        ),
        coherent_signatures=signatures,
    )


def incremental_edit_cost(
    cost: EditCost,
    prior_signatures: Counter[str],
    *,
    weights: EditCostWeights | None = None,
) -> float:
    weights = weights or EditCostWeights()
    if not cost.coherent_signatures:
        return cost.total
    repeated = sum(prior_signatures[signature] > 0 for signature in cost.coherent_signatures)
    if not repeated:
        return cost.total
    coherent_units = min(cost.grapheme, weights.grapheme * len(cost.coherent_signatures))
    discount = coherent_units * (1.0 - weights.repeated_signature_factor)
    return max(0.0, cost.total - discount)
