from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

from .adapters import (
    DictionaryCandidateAdapter,
    KeepCandidateAdapter,
    NeuralCandidateAdapter,
    PhraseOrthographicCandidateAdapter,
    Stage1CandidateAdapter,
    SuffixCandidateAdapter,
)
from .lattice import CandidateLattice, LatticeLimits
from .schema import SpanCandidate
from .text import normalize_for_lattice


def sentence_id_for_text(text: str) -> str:
    return "raw_" + sha256(text.encode("utf-8")).hexdigest()[:16]


def apply_candidate_path(
    raw_text: str,
    candidates: tuple[SpanCandidate, ...] | list[SpanCandidate],
) -> str:
    """Render an explicitly supplied non-overlapping path for diagnostics."""
    normalized = normalize_for_lattice(raw_text).normalized
    ordered = sorted(
        candidates,
        key=lambda candidate: (
            candidate.raw_span.char_start,
            candidate.raw_span.char_end,
        ),
    )
    output: list[str] = []
    cursor = 0
    for candidate in ordered:
        span = candidate.raw_span
        if span.char_start < cursor:
            raise ValueError("Candidate path contains overlapping edges.")
        output.append(normalized[cursor:span.char_start])
        output.append(candidate.replacement)
        cursor = span.char_end
    output.append(normalized[cursor:])
    return "".join(output)


@dataclass(slots=True)
class CandidateGenerationResult:
    raw_text: str
    s1_text: str
    alignment: object
    tokens: tuple
    lattice: CandidateLattice
    diagnostics: object
    stage1_baseline_path: tuple[str, ...]
    production_stage1_result: dict

    def as_dict(self) -> dict:
        return {
            "raw_text": self.raw_text,
            "s1_text": self.s1_text,
            "alignment": self.alignment,
            "tokens": self.tokens,
            "lattice": self.lattice,
            "diagnostics": self.diagnostics,
            "stage1_baseline_path": self.stage1_baseline_path,
            "production_stage1_result": self.production_stage1_result,
        }


class CandidateGenerationPipeline:
    """Candidate-only orchestration. It intentionally has no decoder."""

    def __init__(
        self,
        *,
        spellchecker,
        neural_corrector=None,
        dictionary_index=None,
        limits: LatticeLimits | None = None,
        include_fuzzy: bool = True,
    ) -> None:
        self.spellchecker = spellchecker
        self.neural_corrector = neural_corrector
        self.neural = (
            NeuralCandidateAdapter(neural_corrector)
            if neural_corrector is not None
            else None
        )
        self.limits = limits or LatticeLimits()
        self.stage1 = Stage1CandidateAdapter(spellchecker)
        self.dictionary = DictionaryCandidateAdapter(
            spellchecker, dictionary_index, include_fuzzy=include_fuzzy
        )
        self.suffix = SuffixCandidateAdapter(spellchecker.suffix_generator, spellchecker)
        self.phrase_orthographic = PhraseOrthographicCandidateAdapter(spellchecker)
        self.keep = KeepCandidateAdapter()

    def generate_candidate_lattice(self, raw_text: str) -> CandidateGenerationResult:
        raw = normalize_for_lattice(raw_text)
        production_result = self.spellchecker.correct_text_rich(raw_text)
        stage1_result = self.stage1.generate_candidates(
            raw_text, production_result=production_result
        )
        lattice = CandidateLattice(
            sentence_id=sentence_id_for_text(raw.normalized),
            raw=raw,
            s1_alignment=stage1_result.alignment,
            limits=self.limits,
        )
        # Regenerate Stage 1 proposals against the shared lattice so every
        # provider has identical stable IDs and RAW offsets.
        stage1_result = self.stage1.generate_candidates(
            raw_text,
            s1_text=stage1_result.s1_text,
            alignment=stage1_result.alignment,
            lattice=lattice,
            production_result=production_result,
        )
        provider_candidates: list[SpanCandidate] = []
        provider_candidates.extend(stage1_result.candidates)
        provider_candidates.extend(self.dictionary.generate_candidates(raw_text, lattice))
        provider_candidates.extend(
            self.suffix.generate_candidates(raw_text, lattice.tokens, lattice=lattice)
        )
        provider_candidates.extend(
            self.phrase_orthographic.generate_candidates(raw_text, lattice)
        )
        if self.neural is not None:
            for candidate in self.neural.generate_candidates(raw_text, top_k=3):
                # NeuralCorrector uses the same sentence-ID contract, but
                # rematerialising here also binds its alignment to S1.
                provider_candidates.append(
                    lattice.make_candidate(
                        span=candidate.raw_span,
                        replacement=candidate.replacement,
                        operation=candidate.operation,
                        sources=candidate.sources,
                        morphology=candidate.morphology,
                        dictionary_evidence=candidate.dictionary_evidence,
                        suffix_evidence=candidate.suffix_evidence,
                        edit_operations=candidate.edit_operations,
                        introduced_features=candidate.introduced_features,
                        unsupported_clitic_insertion=candidate.unsupported_clitic_insertion,
                        metadata=candidate.metadata,
                    )
                )
        for candidate in provider_candidates:
            lattice.add(candidate)
        lattice.finalize()
        return CandidateGenerationResult(
            raw_text=raw_text,
            s1_text=stage1_result.s1_text,
            alignment=stage1_result.alignment,
            tokens=lattice.tokens,
            lattice=lattice,
            diagnostics=lattice.diagnostics(),
            stage1_baseline_path=stage1_result.baseline_candidate_ids,
            production_stage1_result=production_result,
        )


def generate_candidate_lattice(
    raw_text: str,
    *,
    spellchecker=None,
    neural_corrector=None,
    dictionary_index=None,
    limits: LatticeLimits | None = None,
    include_fuzzy: bool = True,
) -> dict:
    if spellchecker is None:
        from Essentials.app import spellchecker as production_spellchecker

        spellchecker = production_spellchecker
    return CandidateGenerationPipeline(
        spellchecker=spellchecker,
        neural_corrector=neural_corrector,
        dictionary_index=dictionary_index,
        limits=limits,
        include_fuzzy=include_fuzzy,
    ).generate_candidate_lattice(raw_text).as_dict()
