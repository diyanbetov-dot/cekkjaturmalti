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
    DictionaryAnalysis,
    EditOperation,
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
    "DictionaryAnalysis",
    "DictionaryCandidateAdapter",
    "EditOperation",
    "IntroducedFeature",
    "LatticeDiagnostics",
    "LatticeLimits",
    "LatticeToken",
    "KeepCandidateAdapter",
    "NeuralCandidateAdapter",
    "MorphologyAnalysis",
    "NormalizedText",
    "PhraseOrthographicCandidateAdapter",
    "SourceEvidence",
    "SpanCandidate",
    "Stage1CandidateAdapter",
    "Stage1CandidateResult",
    "SuffixAnalysis",
    "SuffixCandidateAdapter",
    "TextSpan",
    "TokenKind",
    "align_texts",
    "apply_candidate_path",
    "grapheme_edit_distance",
    "generate_candidate_lattice",
    "normalize_for_lattice",
    "tokenize_lattice",
    "sentence_id_for_text",
]
