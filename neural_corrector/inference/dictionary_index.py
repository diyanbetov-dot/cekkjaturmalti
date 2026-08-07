from __future__ import annotations

import re
import sqlite3
import threading
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable

TAG_NOUN = 1 << 0
TAG_PLURAL_NOUN = 1 << 12
TAG_CARDINAL_NUMBER = 1 << 13
TAG_SHORT_ATTRIBUTIVE_NUMBER = 1 << 14
TAG_LONG_ATTRIBUTIVE_NUMBER = 1 << 15
TAG_NUMBER = (
    TAG_CARDINAL_NUMBER
    | TAG_SHORT_ATTRIBUTIVE_NUMBER
    | TAG_LONG_ATTRIBUTIVE_NUMBER
)
TAG_NAME = 1 << 8

WORD_RE = re.compile(r"[^\W\d_]+(?:[-'’][^\W\d_]+)*", re.UNICODE)

CLITIC_PARTS = {
    "b",
    "bi",
    "bil",
    "bħal",
    "bħall",
    "da",
    "dal",
    "dan",
    "dar",
    "das",
    "dat",
    "dax",
    "daż",
    "di",
    "dil",
    "din",
    "dir",
    "dis",
    "dit",
    "dix",
    "diż",
    "f",
    "fi",
    "fil",
    "fis",
    "fit",
    "fix",
    "fiż",
    "għal",
    "għall",
    "għan",
    "iċ",
    "id",
    "il",
    "in",
    "ir",
    "is",
    "it",
    "ix",
    "iż",
    "l",
    "lil",
    "lill",
    "m",
    "ma",
    "mal",
    "mi",
    "mil",
    "mill",
    "min",
    "mis",
    "mit",
    "mix",
    "miż",
    "ta",
    "tal",
    "tan",
    "tar",
    "tas",
    "tat",
    "tax",
    "taż",
    "x",
    "xi",
}


def normalize_key(text: str) -> str:
    return " ".join(
        unicodedata.normalize("NFC", text)
        .replace("’", "'")
        .replace("ʼ", "'")
        .casefold()
        .split()
    )


def fuzzy_key(text: str) -> str:
    folded = normalize_key(text).translate(
        str.maketrans({"ċ": "c", "ġ": "g", "ħ": "h", "ż": "z"})
    )
    return "".join(character for character in folded if character.isalpha())


def bounded_edit_distance(source: str, target: str, limit: int) -> int:
    if abs(len(source) - len(target)) > limit:
        return limit + 1
    previous = list(range(len(target) + 1))
    for source_index, source_character in enumerate(source, start=1):
        current = [source_index]
        row_minimum = source_index
        for target_index, target_character in enumerate(target, start=1):
            value = min(
                current[-1] + 1,
                previous[target_index] + 1,
                previous[target_index - 1]
                + (source_character != target_character),
            )
            current.append(value)
            row_minimum = min(row_minimum, value)
        if row_minimum > limit:
            return limit + 1
        previous = current
    return previous[-1]


@dataclass(frozen=True)
class DictionaryEntry:
    canonical: str
    tag_bits: int
    source_bits: int


class DictionaryIndex:
    def __init__(self, path: Path) -> None:
        self.path = Path(path).resolve()
        if not self.path.exists():
            raise FileNotFoundError(f"Dictionary index not found: {self.path}")
        self._local = threading.local()

    def _connection(self) -> sqlite3.Connection:
        connection = getattr(self._local, "connection", None)
        if connection is None:
            connection = sqlite3.connect(
                f"{self.path.as_uri()}?mode=ro",
                uri=True,
                check_same_thread=False,
            )
            self._local.connection = connection
        return connection

    @lru_cache(maxsize=32768)
    def lookup(self, surface: str) -> DictionaryEntry | None:
        row = self._connection().execute(
            "SELECT canonical, tag_bits, source_bits FROM words WHERE word_key = ?",
            (normalize_key(surface),),
        ).fetchone()
        return DictionaryEntry(*row) if row else None

    def contains(self, surface: str) -> bool:
        return self.lookup(surface) is not None

    @lru_cache(maxsize=32)
    def entries_with_tags(self, tag_mask: int) -> tuple[DictionaryEntry, ...]:
        rows = self._connection().execute(
            """
            SELECT canonical, tag_bits, source_bits
            FROM words
            WHERE (tag_bits & ?) != 0
            """,
            (tag_mask,),
        ).fetchall()
        return tuple(DictionaryEntry(*row) for row in rows)

    @lru_cache(maxsize=8192)
    def nearby_with_tags(
        self, surface: str, tag_mask: int, max_distance: int = 2
    ) -> tuple[tuple[DictionaryEntry, int], ...]:
        source_key = fuzzy_key(surface)
        if not source_key:
            return ()
        matches = []
        for entry in self.entries_with_tags(tag_mask):
            candidate_key = fuzzy_key(entry.canonical)
            if (
                not candidate_key
                or candidate_key[0] != source_key[0]
                or abs(len(candidate_key) - len(source_key)) > max_distance
            ):
                continue
            distance = bounded_edit_distance(
                source_key, candidate_key, max_distance
            )
            if distance <= max_distance:
                matches.append((entry, distance))
        return tuple(
            sorted(
                matches,
                key=lambda item: (
                    item[1],
                    abs(len(item[0].canonical) - len(surface)),
                    normalize_key(item[0].canonical),
                ),
            )
        )

    @lru_cache(maxsize=32768)
    def nearby(
        self, surface: str, max_distance: int = 2, limit: int = 48
    ) -> tuple[tuple[DictionaryEntry, int], ...]:
        source_key = fuzzy_key(surface)
        if not source_key:
            return ()
        minimum = max(1, len(source_key) - max_distance)
        maximum = len(source_key) + max_distance
        rows = self._connection().execute(
            """
            SELECT canonical, tag_bits, source_bits, fuzzy_key
            FROM words
            WHERE fuzzy_length BETWEEN ? AND ?
              AND (fuzzy_head = ? OR fuzzy_tail = ?)
            LIMIT 3000
            """,
            (minimum, maximum, source_key[:1], source_key[-2:]),
        ).fetchall()
        matches = []
        source_is_capitalized = surface[:1].isupper()
        for canonical, tag_bits, source_bits, candidate_key in rows:
            if tag_bits & TAG_NAME and not source_is_capitalized:
                continue
            distance = bounded_edit_distance(
                source_key, candidate_key, max_distance
            )
            if distance <= max_distance:
                matches.append(
                    (
                        DictionaryEntry(
                            canonical, tag_bits, source_bits
                        ),
                        distance,
                    )
                )
        return tuple(
            sorted(
                matches,
                key=lambda item: (
                    item[1],
                    abs(len(fuzzy_key(item[0].canonical)) - len(source_key)),
                    normalize_key(item[0].canonical),
                ),
            )[:limit]
        )

    def contains_surface_form(self, surface: str) -> bool:
        if self.contains(surface):
            return True
        normalized = normalize_key(surface)
        parts = [
            part
            for part in re.split(r"[-']", normalized)
            if part
        ]
        if len(parts) < 2:
            return False
        return all(
            part in CLITIC_PARTS or self.contains(part)
            for part in parts
        )

    def guard_text(
        self,
        original: str,
        corrected: str,
        generated_form_validator: Callable[[str, str], bool] | None = None,
    ) -> tuple[str, list[dict]]:
        original_words = list(WORD_RE.finditer(original))
        corrected_words = list(WORD_RE.finditer(corrected))
        if len(original_words) != len(corrected_words):
            return corrected, []

        replacements: list[tuple[int, int, str]] = []
        decisions: list[dict] = []
        for source_match, candidate_match in zip(
            original_words, corrected_words
        ):
            source = source_match.group(0)
            candidate = candidate_match.group(0)
            if source == candidate:
                continue
            source_key = normalize_key(source)
            candidate_key = normalize_key(candidate)
            if source_key == candidate_key:
                decisions.append(
                    {
                        "source": source,
                        "candidate": candidate,
                        "decision": "accept_case_only",
                    }
                )
                continue
            if self.contains_surface_form(candidate):
                decisions.append(
                    {
                        "source": source,
                        "candidate": candidate,
                        "decision": "accept_dictionary_candidate",
                    }
                )
                continue
            if (
                generated_form_validator is not None
                and generated_form_validator(candidate, source)
            ):
                decisions.append(
                    {
                        "source": source,
                        "candidate": candidate,
                        "decision": "accept_generated_suffix_candidate",
                    }
                )
                continue
            replacements.append(
                (candidate_match.start(), candidate_match.end(), source)
            )
            decisions.append(
                {
                    "source": source,
                    "candidate": candidate,
                    "source_is_dictionary_word": self.contains_surface_form(source),
                    "decision": "reject_unknown_candidate",
                }
            )

        guarded = corrected
        for start, end, replacement in reversed(replacements):
            guarded = guarded[:start] + replacement + guarded[end:]
        return guarded, decisions
