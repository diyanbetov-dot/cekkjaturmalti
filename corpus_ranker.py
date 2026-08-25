from __future__ import annotations

import gzip
import json
import math
import unicodedata
from dataclasses import dataclass
from pathlib import Path


def _normalize(value: str | None) -> str:
    return (
        unicodedata.normalize("NFC", value or "")
        .casefold()
        .strip()
        .replace("’", "'")
        .replace("‘", "'")
    )


def _apostrophe_variants(value: str) -> tuple[str, ...]:
    if "'" not in value:
        return (value,)
    return (value, value.replace("'", "’"), value.replace("'", "‘"))


def _combine_log_counts(values: list[float]) -> float:
    present = [value for value in values if value > 0.0]
    if not present:
        return 0.0
    return math.log1p(sum(math.expm1(value) for value in present))


@dataclass(frozen=True, slots=True)
class CorpusEvidence:
    score: float = 0.0
    unigram: float = 0.0
    left_bigram: float = 0.0
    right_bigram: float = 0.0

    @property
    def contextual_hits(self) -> int:
        return int(self.left_bigram > 0.0) + int(self.right_bigram > 0.0)

    @property
    def contextual_score(self) -> float:
        return self.left_bigram + self.right_bigram


class CorpusCandidateRanker:
    """Read-only corpus evidence for candidates generated elsewhere."""

    def __init__(self, corpus_dir: Path, *, enabled: bool = True) -> None:
        self.corpus_dir = Path(corpus_dir)
        self.enabled = bool(enabled)
        self.available = False
        self.status = "disabled"
        self.meta: dict[str, object] = {}
        self.unigrams: dict[str, float] = {}
        self.bigrams: dict[str, dict[str, float]] = {}
        self._max_log = 1.0
        if self.enabled:
            self._load()

    def _load(self) -> None:
        try:
            with (self.corpus_dir / "meta.json").open("r", encoding="utf-8") as stream:
                self.meta = json.load(stream)
            with gzip.open(self.corpus_dir / "unigrams.json.gz", "rt", encoding="utf-8") as stream:
                self.unigrams = json.load(stream)
            with gzip.open(self.corpus_dir / "bigrams.json.gz", "rt", encoding="utf-8") as stream:
                self.bigrams = json.load(stream)
            self._max_log = max(self.unigrams.values(), default=1.0) or 1.0
            self.available = True
            self.status = "ready"
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            self.status = f"unavailable: {exc}"

    def evidence(
        self,
        candidate: str,
        *,
        previous: str | None = None,
        following: str | None = None,
    ) -> CorpusEvidence:
        left_ctx = [previous] if previous else []
        right_ctx = [following] if following else []
        return self.window_evidence(
            candidate,
            left_context=left_ctx,
            right_context=right_ctx,
            max_distance=1,
        )

    def _unigram_score(self, word: str) -> float:
        return _combine_log_counts(
            [float(self.unigrams.get(variant, 0.0)) for variant in _apostrophe_variants(word)]
        )

    def _bigram_score(self, left: str, right: str) -> float:
        values: list[float] = []
        for left_variant in _apostrophe_variants(left):
            row = self.bigrams.get(left_variant, {})
            for right_variant in _apostrophe_variants(right):
                values.append(float(row.get(right_variant, 0.0)))
        return _combine_log_counts(values)

    def window_evidence(
        self,
        candidate: str,
        *,
        left_context: list[str] | None = None,
        right_context: list[str] | None = None,
        max_distance: int = 10,
    ) -> CorpusEvidence:
        if not (self.enabled and self.available):
            return CorpusEvidence()

        word = _normalize(candidate)
        unigram = self._unigram_score(word)
        if unigram == 0.0 and (word.endswith("ha") or word.endswith("h") or word.endswith("hom")):
            for sfx in ("ha", "hom", "h"):
                if word.endswith(sfx) and len(word) > len(sfx):
                    stem = word[:-len(sfx)]
                    stem_score = self._unigram_score(stem)
                    if stem_score > 0.0:
                        unigram = 0.75 * stem_score
                        break

        left_score = 0.0
        if left_context:
            # Evaluate left context up to max_distance tokens to the left
            tokens = left_context[-max_distance:]
            for dist, w in enumerate(reversed(tokens), 1):
                w_norm = _normalize(w)
                bg = self._bigram_score(w_norm, word)
                if bg > 0:
                    left_score += bg / dist

        right_score = 0.0
        if right_context:
            # Evaluate right context up to max_distance tokens to the right
            tokens = right_context[:max_distance]
            for dist, w in enumerate(tokens, 1):
                w_norm = _normalize(w)
                bg = self._bigram_score(word, w_norm)
                if bg > 0:
                    right_score += bg / dist

        left_1 = _normalize(left_context[-1]) if left_context else ""
        right_1 = _normalize(right_context[0]) if right_context else ""
        left_bigram = self._bigram_score(left_1, word) if left_1 else 0.0
        right_bigram = self._bigram_score(word, right_1) if right_1 else 0.0

        total_score = (
            0.20 * (unigram / self._max_log)
            + 0.45 * (left_score / self._max_log)
            + 0.35 * (right_score / self._max_log)
        )
        return CorpusEvidence(
            score=round(max(0.0, min(1.0, total_score)), 6),
            unigram=unigram,
            left_bigram=left_bigram,
            right_bigram=right_bigram,
        )

    def status_payload(self) -> dict[str, object]:
        stats = self.meta.get("stats", {}) if isinstance(self.meta, dict) else {}
        return {
            "enabled": self.enabled,
            "available": self.available,
            "status": self.status,
            "unigrams": len(self.unigrams),
            "bigram_roots": len(self.bigrams),
            "tokens": stats.get("total_tokens") if isinstance(stats, dict) else None,
        }
