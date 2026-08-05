"""Stage-1-aware contextual correction infrastructure.

Commit 1 intentionally exposes only text, alignment, schema, and lattice
building blocks.  It does not alter either the production spellchecker or the
existing sequential hybrid.
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

__all__ = [
    "AlignmentMap",
    "AlignmentOperation",
    "AlignmentRecord",
    "CandidateLattice",
    "CandidateOperation",
    "DictionaryAnalysis",
    "EditOperation",
    "IntroducedFeature",
    "LatticeDiagnostics",
    "LatticeLimits",
    "LatticeToken",
    "MorphologyAnalysis",
    "NormalizedText",
    "SourceEvidence",
    "SpanCandidate",
    "SuffixAnalysis",
    "TextSpan",
    "TokenKind",
    "align_texts",
    "grapheme_edit_distance",
    "normalize_for_lattice",
    "tokenize_lattice",
]
