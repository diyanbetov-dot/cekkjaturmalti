"""Stage-1-aware contextual correction infrastructure.

The package remains candidate-only: it exposes evidence adapters and lattice
building blocks without altering production correction or selecting output.
"""

from .alignment import AlignmentMap, align_texts
from .lattice import CandidateLattice, LatticeLimits
from .schema import (
    AlignmentOperation,
    AlignmentRecord,
    CandidateOperation,
    CandidateEligibility,
    CandidateSupport,
    CandidateValidation,
    CliticEvidence,
    DictionaryAnalysis,
    EditOperation,
    EditCost,
    FeatureDelta,
    IntroducedFeature,
    LatticeDiagnostics,
    LatticeToken,
    MorphologyAnalysis,
    SourceEvidence,
    SpanCandidate,
    SuffixAnalysis,
    TextSpan,
    TokenKind,
)
from .costs import (
    EditCostWeights,
    SoftEditBudget,
    candidate_edit_cost,
    coherent_edit_signatures,
    incremental_edit_cost,
)
from .decoder import (
    DecodedPath,
    DecoderDiagnostics,
    PathStep,
    PrunedState,
    decode_lattice,
)
from .validation import CandidateValidator, ValidationResult, validate_lattice
from .text import (
    NormalizedText,
    grapheme_edit_distance,
    normalize_for_lattice,
    tokenize_lattice,
)
from .adapters import (
    DictionaryCandidateAdapter,
    KeepCandidateAdapter,
    NeuralCandidateAdapter,
    PhraseOrthographicCandidateAdapter,
    Stage1CandidateAdapter,
    Stage1CandidateResult,
    SuffixCandidateAdapter,
)
from .pipeline import (
    apply_candidate_path,
    CandidateGenerationPipeline,
    CandidateGenerationResult,
    generate_candidate_lattice,
    sentence_id_for_text,
)

__all__ = [
    "AlignmentMap",
    "AlignmentOperation",
    "AlignmentRecord",
    "CandidateLattice",
    "CandidateGenerationPipeline",
    "CandidateGenerationResult",
    "CandidateOperation",
    "CandidateEligibility",
    "CandidateSupport",
    "CandidateValidation",
    "CandidateValidator",
    "CliticEvidence",
    "DictionaryAnalysis",
    "DictionaryCandidateAdapter",
    "EditOperation",
    "EditCost",
    "EditCostWeights",
    "FeatureDelta",
    "IntroducedFeature",
    "LatticeDiagnostics",
    "LatticeLimits",
    "LatticeToken",
    "KeepCandidateAdapter",
    "NeuralCandidateAdapter",
    "MorphologyAnalysis",
    "NormalizedText",
    "DecodedPath",
    "DecoderDiagnostics",
    "PathStep",
    "PrunedState",
    "PhraseOrthographicCandidateAdapter",
    "SourceEvidence",
    "SpanCandidate",
    "Stage1CandidateAdapter",
    "Stage1CandidateResult",
    "SuffixAnalysis",
    "SuffixCandidateAdapter",
    "SoftEditBudget",
    "TextSpan",
    "TokenKind",
    "align_texts",
    "apply_candidate_path",
    "candidate_edit_cost",
    "coherent_edit_signatures",
    "decode_lattice",
    "grapheme_edit_distance",
    "generate_candidate_lattice",
    "normalize_for_lattice",
    "incremental_edit_cost",
    "tokenize_lattice",
    "sentence_id_for_text",
    "ValidationResult",
    "validate_lattice",
]
