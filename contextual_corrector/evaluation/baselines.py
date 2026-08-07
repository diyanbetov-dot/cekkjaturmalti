from __future__ import annotations

from typing import Protocol


class BaselineCorrector(Protocol):
    def correct(self, text: str) -> str:
        ...


class Stage1Baseline:
    """Mechanical Stage 1 rules and dictionary spellchecker baseline."""

    def __init__(self, spellchecker: any) -> None:
        self.spellchecker = spellchecker

    def correct(self, text: str) -> str:
        if hasattr(self.spellchecker, "correct_paragraph"):
            return self.spellchecker.correct_paragraph(text)
        elif hasattr(self.spellchecker, "correct_text"):
            return self.spellchecker.correct_text(text)
        return text


class IdentityBaseline:
    """Pass-through identity baseline (returns raw input)."""

    def correct(self, text: str) -> str:
        return text


class MockNeuralBaseline:
    """Mock neural model baseline for lightweight testing."""

    def __init__(self, fixes: dict[str, str] | None = None) -> None:
        self.fixes = fixes or {}

    def correct(self, text: str) -> str:
        return self.fixes.get(text, text)
