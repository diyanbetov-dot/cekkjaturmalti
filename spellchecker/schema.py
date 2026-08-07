from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, List, Dict, Optional, Tuple


class ErrorClass(str, Enum):
    KEEP = "KEEP"
    DIACRITIC = "DIACRITIC"
    GH_H = "GH_H"
    DOUBLING = "DOUBLING"
    VOWEL = "VOWEL"
    INITIAL_I = "INITIAL_I"
    VOICING = "VOICING"
    MORPHOLOGY = "MORPHOLOGY"
    SUFFIX = "SUFFIX"
    ARTICLE_PREPOSITION = "ARTICLE_PREPOSITION"
    NUMERAL = "NUMERAL"
    SPLIT_JOIN = "SPLIT_JOIN"
    AGREEMENT = "AGREEMENT"
    ENGLISH = "ENGLISH"
    ENTITY = "ENTITY"
    CAPITALIZATION = "CAPITALIZATION"
    PUNCTUATION = "PUNCTUATION"
    REVIEWED_ERROR_MEMORY = "REVIEWED_ERROR_MEMORY"


class RiskClass(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    PROTECTED = "PROTECTED"


@dataclass(frozen=True)
class Token:
    text: str
    normalized: str
    start: int
    end: int
    token_type: str  # "word", "punct", "number", "space"
    casing: str  # "lower", "title", "upper", "mixed"


@dataclass(frozen=True)
class MorphAnalysis:
    surface: str
    lemma: str = ""
    root: str = ""
    upos: str = ""
    morph_tags: str = ""
    source_dictionary: str = ""
    tam_person: str = ""
    gender_number_type: str = ""
    suffix_meta: str = ""
    language: str = "MT"
    entity_class: str = ""


@dataclass
class Candidate:
    source_start: int
    source_end: int
    original_text: str
    replacement: str
    operation_type: ErrorClass = ErrorClass.KEEP
    risk_class: RiskClass = RiskClass.LOW
    sources: List[str] = field(default_factory=list)
    lexical_analyses: List[MorphAnalysis] = field(default_factory=list)
    morph_analyses: List[MorphAnalysis] = field(default_factory=list)
    language_entity_evidence: Dict[str, Any] = field(default_factory=dict)
    grapheme_edit_features: Dict[str, Any] = field(default_factory=dict)
    phonological_features: Dict[str, Any] = field(default_factory=dict)
    suffix_article_numeral_meta: Dict[str, Any] = field(default_factory=dict)
    hard_valid: bool = True
    invalid_reason: str = ""
    detector_probability: float = 0.0
    rank_score: float = 0.0
    keep_score: float = 0.0
    calibrated_confidence: float = 0.0


@dataclass
class SelectedEdit:
    source_span: Tuple[int, int]
    original: str
    replacement: str
    reason: str
    confidence: float
    candidate_source: str
    diagnostics: Dict[str, Any] = field(default_factory=dict)
