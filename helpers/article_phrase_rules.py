from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from pathlib import Path


SUN_LETTERS = {"ċ", "d", "n", "r", "s", "t", "x", "z", "ż"}
VOWELS = set("aeiouàèìòù")
CONSONANTS = set("bcċdfġgħhħjklmnpqrstvxżz")

NOUN_TAG_MARKERS = ("NOUN",)
NUM_TAG_MARKERS = ("NUM",)


@dataclass(frozen=True)
class WordToken:
    text: str
    start: int
    end: int


@dataclass(frozen=True)
class ArticlePhraseSuggestion:
    start: int
    end: int
    corrected: str
    choices: list[dict[str, str]]


def normalize_word(word: str) -> str:
    return (
        unicodedata.normalize("NFC", str(word).strip()).casefold()
        .replace("\u2019", "'")
        .replace("\u2018", "'")
        .replace("\u02bc", "'")
    )


def normalize_word_exact(word: str) -> str:
    return (
        unicodedata.normalize("NFC", str(word).strip())
        .replace("\u2019", "'")
        .replace("\u2018", "'")
        .replace("\u02bc", "'")
    )


def split_dictionary_line(line: str) -> tuple[str, str]:
    line = str(line).strip()
    if not line or line.startswith("#") or "/" not in line:
        return "", ""
    surface, payload = line.split("/", 1)
    return surface.strip(), payload.strip()


def load_noun_words(paths: list[Path]) -> set[str]:
    nouns: set[str] = set()
    for path in paths:
        try:
            lines = Path(path).read_text(encoding="utf-8", errors="ignore").splitlines()
        except FileNotFoundError:
            continue
        if lines and lines[0].strip().isdigit():
            lines = lines[1:]
        for line in lines:
            surface, payload = split_dictionary_line(line)
            if not surface or not payload:
                continue
            tag = payload.split("-", 1)[0].upper()
            if any(marker in tag for marker in NOUN_TAG_MARKERS):
                exact_surface = normalize_word_exact(surface)
                nouns.add(normalize_word(surface))
                if exact_surface != normalize_word(surface):
                    nouns.add(exact_surface)
    return nouns


def load_num_words(paths: list[Path]) -> set[str]:
    nums: set[str] = set()
    for path in paths:
        try:
            lines = Path(path).read_text(encoding="utf-8", errors="ignore").splitlines()
        except FileNotFoundError:
            continue
        if lines and lines[0].strip().isdigit():
            lines = lines[1:]
        for line in lines:
            surface, payload = split_dictionary_line(line)
            if not surface or not payload:
                continue
            tag = payload.split("-", 1)[0].upper()
            if any(marker in tag for marker in NUM_TAG_MARKERS):
                exact_surface = normalize_word_exact(surface)
                nums.add(normalize_word(surface))
                if exact_surface != normalize_word(surface):
                    nums.add(exact_surface)
    return nums


class MalteseArticlePhraseRules:
    def __init__(
        self,
        *,
        dictionary_files,
        meaning_index,
        normalizer=normalize_word,
    ) -> None:
        self.noun_words = load_noun_words([Path(path) for path in dictionary_files])
        self.num_words = load_num_words([Path(path) for path in dictionary_files])
        self.meaning_index = meaning_index
        self.normalizer = normalizer

    def normalize(self, word: str) -> str:
        return self.normalizer(word)

    def is_noun(self, word: str) -> bool:
        return self._contains_surface(self.noun_words, word)

    def is_num(self, word: str) -> bool:
        return self._contains_surface(self.num_words, word)

    def _contains_surface(self, surfaces: set[str], word: str) -> bool:
        exact = normalize_word_exact(word)
        if exact in surfaces:
            return True
        lowered = self.normalize(word)
        return lowered in surfaces

    def previous_ends_vowelish(self, previous: str | None) -> bool:
        if not previous:
            return False
        normalized = self.normalize(previous).rstrip("-")
        if not normalized:
            return False
        if normalized[-1] in VOWELS:
            return True
        return len(normalized) >= 2 and normalized[-1] == "'" and normalized[-2] in VOWELS

    def assimilate(self, article: str, noun: str) -> str:
        normalized_article = self.normalize(article).rstrip("-")
        normalized_noun = self.normalize(noun)
        if not normalized_noun:
            return article
        first = normalized_noun[0]
        if first in SUN_LETTERS:
            return f"i{first}-" if normalized_article == "il" else f"{first}-"
        return f"{normalized_article}-"

    def article_from_previous(self, previous: str | None) -> str:
        return "l-" if self.previous_ends_vowelish(previous) else "il-"

    def corrected_article_phrase(
        self,
        typed_article: str,
        noun: str,
        previous: str | None,
    ) -> str:
        base_article = self.article_from_previous(previous)
        return f"{self.assimilate(base_article, noun)}{noun}"

    def phrase_choices(self, noun: str, previous: str | None) -> list[dict[str, str]]:
        noun_meaning = self.meaning_index.meaning_for(noun)

        def add_runtime_meaning(prefix: str) -> str:
            return f"{prefix} {noun_meaning}" if noun_meaning else prefix

        definite = self.corrected_article_phrase("il", noun, previous)
        return [{"word": definite, "meaning": add_runtime_meaning("the")}]

    def literal_article_choices(
        self,
        article: str,
        noun: str,
        previous: str | None,
    ) -> list[dict[str, str]]:
        noun_meaning = self.meaning_index.meaning_for(noun)

        def add_runtime_meaning(prefix: str) -> str:
            return f"{prefix} {noun_meaning}" if noun_meaning else prefix

        to_article = "l" if self.normalize(article) == "l" else "il"
        return [
            {"word": f"'{to_article} {noun}", "meaning": add_runtime_meaning("to a")},
            {"word": f"'{to_article}-{noun}", "meaning": add_runtime_meaning("to the")},
        ]

    def num_choices(self, numeral: str, previous: str | None) -> list[dict[str, str]]:
        meaning = self.meaning_index.meaning_for(numeral)
        return [{"word": numeral, "meaning": f"the {meaning}" if meaning else "the"}]

    def _is_article_target(self, word: str) -> bool:
        return self.is_noun(word) or self.is_num(word)

    def _starts_vowel_gh_or_h(self, word: str) -> bool:
        normalized = self.normalize(word)
        return bool(
            normalized
            and (
                normalized[0] in VOWELS
                or normalized.startswith(("għ", "gh", "h", "ħ"))
            )
        )

    def _strict_dictionary_tail(self, word: str) -> str | None:
        spellchecker = getattr(self, "spellchecker", None)
        if spellchecker is None:
            return None
        normalized = self.normalize(word)
        if normalized in spellchecker.dictionary_set:
            return normalized
        for candidate in spellchecker._strict_lookup_variants(normalized):
            if candidate in spellchecker.dictionary_set:
                return candidate
        orthographic = getattr(spellchecker, "orthographic_generator", None)
        if orthographic is None:
            return None
        helper_names = (
            "dictionary_gh_priority_variants",
            "dictionary_shortcut_variants",
            "dictionary_final_gh_h_hbar_variants",
            "dictionary_i_ie_variants",
        )
        for helper_name in helper_names:
            helper = getattr(orthographic, helper_name, None)
            if helper is None:
                continue
            for candidate in helper(normalized):
                candidate = self.normalize(candidate)
                if candidate in spellchecker.dictionary_set:
                    return candidate
        return None

    def _compact_prefix_surface(self, canonical_prefix: str, tail: str) -> str | None:
        if not tail or not self._starts_vowel_gh_or_h(tail):
            return None
        if canonical_prefix == "bil":
            return f"bl-{tail}"
        if canonical_prefix == "fil":
            return f"fl-{tail}"
        if canonical_prefix == "xil":
            return f"x'l-{tail}"
        return None

    def preposition_article_form(self, prefix: str, noun: str) -> str | None:
        prefix = self.normalize(prefix).rstrip("-")
        noun = self.normalize(noun)
        if not noun or not self._is_article_target(noun):
            return None
        first = noun[0]
        starts_vowelish = self._starts_vowel_gh_or_h(noun)

        if prefix == "tal":
            return f"t{self.assimilate('il-', noun)}{noun}"
        if prefix == "mal":
            return f"m{self.assimilate('il-', noun)}{noun}"
        if prefix == "lill":
            return f"lill-{noun}" if starts_vowelish else f"lil-{noun}"
        if prefix in {"bil", "bl"}:
            return f"bl-{noun}" if starts_vowelish else f"bi{self.assimilate('il-', noun)}{noun}"
        if prefix in {"fil", "fl"}:
            return f"fl-{noun}" if starts_vowelish else f"fi{self.assimilate('il-', noun)}{noun}"
        if prefix in {"xil", "x'l"}:
            return f"x'l-{noun}" if starts_vowelish else f"xi{self.assimilate('il-', noun)}{noun}"
        if prefix in {"il", "l"} or prefix in SUN_LETTERS:
            return f"{self.assimilate('il-', noun)}{noun}"
        return None

    def preposition_article_choices(
        self,
        prefix: str,
        noun: str,
        previous: str | None,
    ) -> list[dict[str, str]]:
        corrected = self.preposition_article_form(prefix, noun)
        if not corrected:
            return []
        meaning = self.meaning_index.meaning_for(noun)
        return [{"word": corrected, "meaning": meaning}]

    def match_split_article(
        self,
        words: list[WordToken],
        index: int,
    ) -> ArticlePhraseSuggestion | None:
        if index + 1 >= len(words):
            return None

        article = self.normalize(words[index].text).rstrip("-")
        noun = self.normalize(words[index + 1].text)
        corrected_noun = noun

        if not self._is_article_target(corrected_noun):
            spellchecker = getattr(self, "spellchecker", None)
            if spellchecker is not None:
                candidate = self.normalize(spellchecker.correct_word(noun))
                if candidate != noun and self._is_article_target(candidate):
                    corrected_noun = candidate

        if not self._is_article_target(corrected_noun):
            return None

        previous = words[index - 1].text if index > 0 else None

        if article in {"għar", "ghar"} and corrected_noun.startswith("r"):
            corrected = f"għar-{corrected_noun}"
            return ArticlePhraseSuggestion(index, index + 2, corrected, [])

        if article in {"il", "l"} or article in SUN_LETTERS:
            corrected = self.corrected_article_phrase(article, corrected_noun, previous)
            choices = self.phrase_choices(corrected_noun, previous)
            if article == "l":
                choices.extend(self.literal_article_choices(article, corrected_noun, previous))
            return ArticlePhraseSuggestion(index, index + 2, corrected, choices)

        if article in {"tal", "mal", "bil", "fil", "lill", "xil"}:
            corrected = self.preposition_article_form(article, corrected_noun)
            if not corrected:
                return None
            choices = self.preposition_article_choices(article, corrected_noun, previous)
            return ArticlePhraseSuggestion(index, index + 2, corrected, choices)

        return None

    def match_hyphenated_article(
        self,
        word: str,
    ) -> ArticlePhraseSuggestion | None:
        return self.match_hyphenated_article_after(word, previous=None)

    def match_hyphenated_article_after(
        self,
        word: str,
        *,
        previous: str | None,
    ) -> ArticlePhraseSuggestion | None:
        normalized = self.normalize(word)
        if "-" not in normalized:
            return None

        prefix, noun = normalized.split("-", 1)
        if not noun:
            return None

        corrected_noun = self._strict_dictionary_tail(noun) or noun
        if not self._is_article_target(corrected_noun):
            return None

        if prefix in SUN_LETTERS:
            return ArticlePhraseSuggestion(
                0,
                1,
                f"{prefix}-{corrected_noun}",
                [],
            )

        if prefix in {"il", "l", "din", "dan"} or prefix.startswith("i"):
            corrected = self.corrected_article_phrase(prefix, corrected_noun, previous)
            choices = self.phrase_choices(corrected_noun, previous)
            return ArticlePhraseSuggestion(0, 1, corrected, choices)

        if prefix in {"tal", "mal", "bil", "fil", "lill", "xil"}:
            corrected = self.preposition_article_form(prefix, corrected_noun)
            if corrected:
                choices = self.preposition_article_choices(prefix, corrected_noun, previous)
                return ArticlePhraseSuggestion(0, 1, corrected, choices)

        return None

    def match_preposition_article_contraction(
        self,
        words: list[WordToken],
        index: int,
    ) -> ArticlePhraseSuggestion | None:
        if index + 1 >= len(words):
            return None

        preposition = self.normalize(words[index].text)
        next_word = self.normalize(words[index + 1].text)
        prefix_map = {
            "ta": "tal",
            "ma": "mal",
            "bi": "bil",
            "fi": "fil",
            "lil": "lill",
            "għal": "għall",
            "ghal": "għall",
        }
        canonical_prefix = prefix_map.get(preposition)
        if not canonical_prefix:
            return None

        if next_word.startswith(("il-", "l-")):
            noun = next_word.split("-", 1)[1]
        elif next_word in {"il", "l"} and index + 2 < len(words):
            noun = self.normalize(words[index + 2].text)
        else:
            return None

        corrected = self.preposition_article_form(canonical_prefix, noun)
        if not corrected:
            return None
        choices = self.preposition_article_choices(canonical_prefix, noun, None)
        return ArticlePhraseSuggestion(index, index + 2, corrected, choices)

    def match_compact_preposition_article(
        self,
        word: str,
    ) -> ArticlePhraseSuggestion | None:
        normalized = self.normalize(word)
        compact_prefixes = (
            ("għall", "għal"),
            ("mill", "mi"),
            ("tal", "ta"),
            ("mal", "ma"),
            ("bil", "bi"),
            ("fil", "fi"),
            ("xil", "xi"),
            ("fl", "fi"),
            ("bl", "bi"),
            ("x'l", "xi"),
        )

        for typed_prefix, canonical in compact_prefixes:
            if not normalized.startswith(typed_prefix) or len(normalized) <= len(typed_prefix):
                continue
            tail = normalized[len(typed_prefix) :]
            exact_tail = self._strict_dictionary_tail(tail) or tail
            if not self._is_article_target(exact_tail):
                continue
            corrected = self.preposition_article_form(typed_prefix, exact_tail)
            if not corrected:
                continue
            choices = self.preposition_article_choices(typed_prefix, exact_tail, None)
            return ArticlePhraseSuggestion(0, 1, corrected, choices)

        return None

    def collapse_three_same_consonants(self, word: str) -> str:
        letters = list(self.normalize(word))
        out: list[str] = []
        i = 0
        while i < len(letters):
            if (
                i + 2 < len(letters)
                and letters[i] == letters[i + 1] == letters[i + 2]
                and letters[i] in CONSONANTS
            ):
                out.extend([letters[i], letters[i + 1]])
                i += 3
                continue
            out.append(letters[i])
            i += 1
        return "".join(out)
