from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScoreRow:
    candidate: str
    score: float
    edit_distance: int
    consonant_score: float
    vowel_slot_score: float
    vowel_count_score: float
    length_score: float
    stage: str
    matched_typo_form: str


@dataclass(frozen=True)
class UnifiedMatch:
    start_index: int
    end_index: int
    group0: str
    is_quote: bool = False
    inner_text: str = ""

    def start(self) -> int:
        return self.start_index

    def end(self) -> int:
        return self.end_index

    def group(self, idx: int) -> str:
        return self.group0 if idx == 0 else self.inner_text
