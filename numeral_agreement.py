from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from pathlib import Path

from morphology_agreement import parse_tagged_dictionary


def normalize(value: str) -> str:
    return unicodedata.normalize("NFC", value).casefold()


def graphemes(word: str) -> list[str]:
    normalized = normalize(word)
    values: list[str] = []
    index = 0
    while index < len(normalized):
        if normalized.startswith("għ", index):
            values.append("għ")
            index += 2
        else:
            values.append(normalized[index])
            index += 1
    return values


def begins_two_consonants(word: str) -> bool:
    values = graphemes(word)
    vowels = frozenset("aeiouàèìòù")
    return len(values) >= 2 and values[0] not in vowels and values[1] not in vowels


@dataclass(frozen=True, slots=True)
class NumeralPhraseCandidate:
    numeral: str
    noun: str
    noun_base: str
    noun_tag: str


class AttributiveNumeralResolver:
    def __init__(
        self,
        base_dics: Path,
        *,
        corpus_bigrams: dict[str, dict[str, float]] | None = None,
        enabled: bool = True,
    ) -> None:
        self.enabled = bool(enabled)
        self.available = False
        self.status = "disabled"
        self.short_to_long: dict[str, str] = {}
        self.long_to_short: dict[str, str] = {}
        self.short_forms: set[str] = set()
        self.singular_to_plural: dict[str, tuple[str, str]] = {}
        self.corpus_i_nouns: set[str] = set()
        self.noun_tags = parse_tagged_dictionary(base_dics / "fixednouns.dic")
        if self.enabled:
            self._load_numerals(base_dics / "prepositions.dic")
            self._load_noun_families(base_dics / "fixednouns.dic")
            self._load_corpus_i_nouns(corpus_bigrams or {})

    def _load_numerals(self, path: Path) -> None:
        short_by_meaning: dict[str, list[str]] = {}
        long_by_meaning: dict[str, list[str]] = {}
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "/" not in stripped:
                continue
            surface, remainder = stripped.split("/", 1)
            tag, separator, meaning = remainder.partition("-")
            if not separator or tag not in {"SHORTATTNUM", "LONGATTNUM"}:
                continue
            key = normalize(meaning.strip())
            if tag == "SHORTATTNUM":
                short_by_meaning.setdefault(key, []).append(normalize(surface.strip()))
                self.short_forms.add(normalize(surface.strip()))
            else:
                long_by_meaning.setdefault(key, []).append(normalize(surface.strip()))

        for meaning, short_forms in short_by_meaning.items():
            long_forms = long_by_meaning.get(meaning, [])
            for index, short in enumerate(short_forms):
                long = long_forms[index] if index < len(long_forms) else None
                if long is None and short.endswith("t"):
                    long = short
                if long is None:
                    continue
                self.short_to_long[short] = long
                self.long_to_short[long] = short
        self.available = bool(self.short_to_long)
        self.status = "ready" if self.available else "unavailable: no numeral pairs"

    @staticmethod
    def _meaning_key(meaning: str) -> str:
        value = normalize(meaning).split(";", 1)[0].split(",", 1)[0].strip()
        if value.endswith("ies"):
            return value[:-3] + "y"
        if value.endswith("ses"):
            return value[:-2]
        if value.endswith("s") and not value.endswith("ss"):
            return value[:-1]
        return value

    @staticmethod
    def _edit_distance(first: str, second: str) -> int:
        previous = list(range(len(second) + 1))
        for row, left in enumerate(first, 1):
            current = [row]
            for column, right in enumerate(second, 1):
                current.append(min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + int(left != right),
                ))
            previous = current
        return previous[-1]

    def _load_noun_families(self, path: Path) -> None:
        singulars: list[tuple[str, str]] = []
        plurals: list[tuple[str, str, str]] = []
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "/" not in stripped:
                continue
            surface, remainder = stripped.split("/", 1)
            tag, _, meaning = remainder.partition("-")
            word = normalize(surface.strip())
            if not word or " " in word or "-" in word:
                continue
            meaning_key = self._meaning_key(meaning)
            if tag in {"SINGNOUNM", "SINGNOUNF", "SINGNOUN"}:
                singulars.append((word, meaning_key))
            elif tag in {"PAUCNOUN", "PLUNOUN", "COLLNOUN"}:
                plurals.append((word, tag, meaning_key))

        for singular, singular_meaning in singulars:
            prefix = singular[:3]
            candidates = [
                (plural, tag, meaning)
                for plural, tag, meaning in plurals
                if plural[:3] == prefix
            ]
            if not candidates:
                continue
            plural, tag, _ = min(
                candidates,
                key=lambda item: (
                    0 if singular_meaning and item[2] == singular_meaning else 1,
                    0 if item[1] == "PAUCNOUN" else 1,
                    self._edit_distance(singular, item[0]),
                    len(item[0]),
                    item[0],
                ),
            )
            self.singular_to_plural[singular] = (plural, tag)

    def _load_corpus_i_nouns(self, bigrams: dict[str, dict[str, float]]) -> None:
        for long in set(self.short_to_long.values()):
            for following, frequency in bigrams.get(long, {}).items():
                if frequency <= 0.0 or not following.startswith("i"):
                    continue
                base = following[1:]
                tags = self.noun_tags.get(base, set())
                if {"PLUNOUN", "COLLNOUN"}.intersection(tags):
                    self.corpus_i_nouns.add(following)

    def _plural_noun(self, word: str) -> tuple[str, str] | None:
        normalized = normalize(word)
        tags = self.noun_tags.get(normalized, set())
        plural_tag = next((tag for tag in ("PLUNOUN", "COLLNOUN") if tag in tags), None)
        if plural_tag:
            return normalized, plural_tag

        if normalized.startswith("i"):
            base = normalized[1:]
            base_tags = self.noun_tags.get(base, set())
            plural_tag = next((tag for tag in ("PLUNOUN", "COLLNOUN") if tag in base_tags), None)
            if plural_tag and (
                begins_two_consonants(base)
                or normalized in self.corpus_i_nouns
            ):
                return base, plural_tag
        return None

    def resolve(self, numeral: str, noun: str) -> NumeralPhraseCandidate | None:
        if not (self.enabled and self.available):
            return None
        numeral_key = normalize(numeral)
        if numeral_key not in self.short_forms and numeral_key + "'" in self.short_forms:
            numeral_key += "'"
        long = self.short_to_long.get(numeral_key)
        short = numeral_key
        if numeral_key in self.long_to_short:
            long = numeral_key
            short = self.long_to_short[numeral_key]
        elif numeral_key not in self.short_forms:
            return None

        noun_match = self._plural_noun(noun)
        if noun_match is None:
            noun_match = self.singular_to_plural.get(normalize(noun))
            if noun_match is None:
                return None
        noun_base, noun_tag = noun_match
        needs_long = begins_two_consonants(noun_base) and long is not None
        contextual_noun = "i" + noun_base if needs_long else noun_base
        contextual_numeral = long if needs_long else short
        if numeral_key == contextual_numeral and normalize(noun) == contextual_noun:
            return None
        return NumeralPhraseCandidate(
            numeral=contextual_numeral,
            noun=contextual_noun,
            noun_base=noun_base,
            noun_tag=noun_tag,
        )

    def status_payload(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "available": self.available,
            "status": self.status,
            "numeral_pairs": len(self.short_to_long),
            "corpus_i_nouns": len(self.corpus_i_nouns),
        }
