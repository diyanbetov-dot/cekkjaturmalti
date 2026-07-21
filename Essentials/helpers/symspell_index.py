import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Iterable


@dataclass(frozen=True)
class SymSpellStats:
    built: bool
    words: int
    delete_keys: int
    elapsed_ms: float


class MalteseSymSpellIndex:
    """Small delete-index candidate retriever.

    This intentionally does not choose corrections. It only returns a bounded
    candidate set that the existing Maltese scorer and rule guards must accept.
    """

    def __init__(
        self,
        *,
        normalizer: Callable[[str], str],
        token_key: Callable[[str], tuple[str, ...]],
        max_edit_distance: int = 2,
        max_word_length: int = 32,
        max_bucket_size: int = 96,
    ) -> None:
        self.normalizer = normalizer
        self.token_key = token_key
        self.max_edit_distance = max(1, int(max_edit_distance))
        self.max_word_length = max(1, int(max_word_length))
        self.max_bucket_size = max(1, int(max_bucket_size))
        self._delete_index: dict[str, list[str]] = defaultdict(list)
        self._words: set[str] = set()
        self.stats = SymSpellStats(False, 0, 0, 0.0)

    def build(self, words: Iterable[str]) -> SymSpellStats:
        started = time.perf_counter()
        for word in words:
            normalized = self.normalizer(word)
            if not normalized or len(normalized) > self.max_word_length:
                continue
            if "-" in normalized or "'" in normalized or " " in normalized:
                continue
            if not any(ch.isalpha() for ch in normalized):
                continue
            self._words.add(normalized)
            for delete in self._deletes(normalized):
                bucket = self._delete_index[delete]
                if len(bucket) < self.max_bucket_size and normalized not in bucket:
                    bucket.append(normalized)
        self.stats = SymSpellStats(
            built=True,
            words=len(self._words),
            delete_keys=len(self._delete_index),
            elapsed_ms=(time.perf_counter() - started) * 1000,
        )
        return self.stats

    def lookup(self, word: str, *, limit: int = 64) -> list[str]:
        normalized = self.normalizer(word)
        if not normalized:
            return []

        ordered: list[str] = []
        seen: set[str] = set()

        def add(candidate: str) -> None:
            if candidate and candidate not in seen:
                seen.add(candidate)
                ordered.append(candidate)

        if normalized in self._words:
            add(normalized)
        for delete in self._deletes(normalized):
            for candidate in self._delete_index.get(delete, ()):
                add(candidate)
                if len(ordered) >= limit:
                    return ordered
        return ordered

    def _deletes(self, word: str) -> set[str]:
        tokens = self.token_key(word)
        if not tokens:
            return {word}
        levels: set[tuple[str, ...]] = {tokens}
        all_deletes: set[tuple[str, ...]] = {tokens}
        for _ in range(self.max_edit_distance):
            next_level: set[tuple[str, ...]] = set()
            for item in levels:
                if len(item) <= 1:
                    continue
                for index in range(len(item)):
                    next_level.add(item[:index] + item[index + 1 :])
            all_deletes.update(next_level)
            levels = next_level
        return {"".join(item) for item in all_deletes}
