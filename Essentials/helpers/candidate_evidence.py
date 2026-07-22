from dataclasses import dataclass, field


CONFIDENCE_WEIGHT = {
    "exact": 1.0,
    "very-high": 0.92,
    "high": 0.82,
    "medium": 0.62,
    "low": 0.38,
    "shadow": 0.18,
}


@dataclass(slots=True)
class CandidateEvidence:
    word: str
    sources: set[str] = field(default_factory=set)
    candidate_types: set[str] = field(default_factory=set)
    reasons: set[str] = field(default_factory=set)
    confidence: float = 0.0
    apertium_available: bool | None = None
    apertium_analyses: tuple[str, ...] = ()

    def add_evidence(
        self,
        *,
        source: str,
        candidate_type: str = "candidate",
        confidence: str | float = "medium",
        reason: str = "",
    ) -> None:
        self.sources.add(source)
        if candidate_type:
            self.candidate_types.add(candidate_type)
        if reason:
            self.reasons.add(reason)
        if isinstance(confidence, str):
            weight = CONFIDENCE_WEIGHT.get(confidence, CONFIDENCE_WEIGHT["medium"])
        else:
            weight = float(confidence)
        self.confidence = max(self.confidence, max(0.0, min(1.0, weight)))


class CandidateEvidencePool:
    def __init__(self, *, normalizer) -> None:
        self.normalizer = normalizer
        self._items: dict[str, CandidateEvidence] = {}

    def add(
        self,
        word: str,
        source: str,
        *,
        candidate_type: str = "candidate",
        confidence: str | float = "medium",
        reason: str = "",
    ) -> None:
        normalized = self.normalizer(word)
        if not normalized:
            return
        item = self._items.get(normalized)
        if item is None:
            item = CandidateEvidence(word=normalized)
            self._items[normalized] = item
        item.add_evidence(
            source=source,
            candidate_type=candidate_type,
            confidence=confidence,
            reason=reason,
        )

    def add_many(
        self,
        words,
        source: str,
        *,
        candidate_type: str = "candidate",
        confidence: str | float = "medium",
        reason: str = "",
    ) -> None:
        for word in words:
            self.add(
                word,
                source,
                candidate_type=candidate_type,
                confidence=confidence,
                reason=reason,
            )

    def annotate_apertium(self, analyzer) -> None:
        if analyzer is None or not getattr(analyzer, "enabled", False):
            return
        for item in self._items.values():
            analysis = analyzer.analyze(item.word)
            item.apertium_available = analysis.available
            item.apertium_analyses = analysis.analyses
            if analysis.available:
                item.add_evidence(
                    source="apertium",
                    candidate_type="morphology",
                    confidence="medium",
                    reason="morphologically recognized",
                )

    def words(self) -> list[str]:
        return [item.word for item in self._items.values()]

    def items(self) -> list[CandidateEvidence]:
        return list(self._items.values())

    def by_word(self) -> dict[str, CandidateEvidence]:
        return dict(self._items)

    def diagnostics(self, *, limit: int = 12) -> list[dict]:
        out: list[dict] = []
        for item in list(self._items.values())[:limit]:
            out.append(
                {
                    "word": item.word,
                    "sources": sorted(item.sources),
                    "types": sorted(item.candidate_types),
                    "reasons": sorted(item.reasons),
                    "confidence": round(item.confidence, 3),
                    "apertium_available": item.apertium_available,
                    "apertium_analyses": list(item.apertium_analyses[:4]),
                }
            )
        return out
