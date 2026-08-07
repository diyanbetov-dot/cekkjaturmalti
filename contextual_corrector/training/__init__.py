"""Training infrastructure for contextual candidate ranker."""

from .schema import ContextualTrainingExample, TrainingMetadata
from .gold_forest import GoldPathForest, build_gold_forest, inject_oracle_candidates
from .loss import ContextualStructuredLoss, LossComponents
from .corruption import corrupt_stage1_output
from .train import train_contextual_ranker

__all__ = [
    "ContextualTrainingExample",
    "TrainingMetadata",
    "GoldPathForest",
    "build_gold_forest",
    "inject_oracle_candidates",
    "ContextualStructuredLoss",
    "LossComponents",
    "corrupt_stage1_output",
    "train_contextual_ranker",
]
