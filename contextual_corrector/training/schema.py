from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..lattice import CandidateLattice
    from .gold_forest import GoldPathForest


@dataclass(slots=True)
class TrainingMetadata:
    source_dataset: str = "maltese_corpus"
    parent_id: str = ""
    manually_reviewed: bool = True
    corruption_family: str | None = None
    clean_identity: bool = False
    split_group: str = "train"
    stage1_version: str = "stage1-v1.0"
    generator_versions: dict[str, str] = field(default_factory=dict)
    accepted_alternatives: tuple[str, ...] = field(default_factory=tuple)


@dataclass(slots=True)
class ContextualTrainingExample:
    example_id: str
    raw_text: str
    accepted_outputs: tuple[str, ...]
    s1_text: str
    s1_out_of_fold: bool
    lattice: Any  # CandidateLattice
    gold_forest: Any  # GoldPathForest
    metadata: TrainingMetadata
