from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from pathlib import Path


SUN_LETTERS = {"ċ", "d", "n", "r", "s", "t", "x", "z", "ż"}
VOWELS = set("aeiouàèìòù")
CONSONANTS = set("bcċdfġgħhħjklmnpqrstvxżz")

# Canonical article-like forms and their sun-letter stems.  The matcher below
# compares shortcut-insensitively, so typed forms such as tac-, bhac-, and
# goc- resolve through the same table as their fully Maltese spellings.
ASSIMILATED_PREFIX_FAMILIES = {
    "tal": "ta",
    "bil": "bi",
    "fil": "fi",
    "mill": "mi",
    "mal": "ma",
    "ġol": "ġo",
    "għat": "għa",
    "għall": "għa",
    "bħall": "bħa",
    "dal": "da",
    "dil": "di",
}

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
        dictionary_files=None,
        meaning_index,
        normalizer=normalize_word,
        noun_words=None,
        num_words=None,
    ) -> None:
        if noun_words is not None:
            self.noun_words = set(noun_words)
        else:
            self.noun_words = load_noun_words([Path(path) for path in (dictionary_files or ())])
        if num_words is not None:
            self.num_words = set(num_words)
        else:
            self.num_words = load_num_words([Path(path) for path in (dictionary_files or ())])
        self.meaning_index = meaning_index
        self.normalizer = normalizer

    def normalize(self, word: str) -> str:
        return self.normalizer(word)

    def is_noun(self, word: str) -> bool:
        return self._contains_surface(self.noun_words, word)

    def is_num(self, word: str) -> bool:
        return self._contains_surface(self.num_words, word)

    def is_adjective(self, word: str) -> bool:
        spellchecker = getattr(self, "spellchecker", None)
        return bool(
            spellchecker is not None
            and spellchecker._is_adjective_tagged_word(word)
        )

    def is_adjective_like(self, word: str) -> bool:
        if self.is_adjective(word):
            return True
        spellchecker = getattr(self, "spellchecker", None)
        return bool(
            spellchecker is not None
            and any(
                tag.split("-", 1)[0] == "PASTPAR"
                for tag in spellchecker.word_tags.get(self.normalize(word), set())
            )
        )

    def _requires_article_epenthetic_i(self, word: str) -> bool:
        normalized = self.normalize(word)
        return bool(
            len(normalized) >= 2
            and normalized[0] in {"m", "s"}
            and normalized[1] not in VOWELS
        )

    def _adjectival_article_surface(self, word: str) -> str:
        return f"l-i{word}" if self._requires_article_epenthetic_i(word) else f"l-{word}"

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
        spellchecker = getattr(self, "spellchecker", None)
        place_display = (
            spellchecker._exact_place_word(noun)
            if spellchecker is not None
            else None
        )
        surface_noun = place_display or noun
        if self._requires_article_epenthetic_i(noun):
            return f"l-i{surface_noun}"
        typed = self.normalize(typed_article).rstrip("-")
        if typed == "l":
            base_article = "l-"
        else:
            base_article = self.article_from_previous(previous)
        return f"{self.assimilate(base_article, noun)}{surface_noun}"

    def phrase_choices(self, noun: str, previous: str | None) -> list[dict[str, str]]:
        noun_meaning = self.meaning_index.meaning_for(noun)
        spellchecker = getattr(self, "spellchecker", None)
        english_equivalents = (
            spellchecker._english_fixed_noun_suggestions(noun)
            if spellchecker is not None
            else ()
        )

        if english_equivalents:
            choices: list[dict[str, str]] = []
            for equivalent in english_equivalents:
                article_word = self.corrected_article_phrase("il", equivalent, previous)
                meaning = self.meaning_index.meaning_for(equivalent)
                choices.append(
                    {
                        "word": article_word,
                        "meaning": f"the {meaning}" if meaning else "the",
                    }
                )
            return choices

        def add_runtime_meaning(prefix: str) -> str:
            return f"{prefix} {noun_meaning}" if noun_meaning else prefix

        definite = self.corrected_article_phrase("il", noun, previous)
        if self.is_adjective_like(noun):
            definite = self._adjectival_article_surface(noun)
            superlative = self._superlative_meaning(noun_meaning)
            choices = [
                {"word": definite, "meaning": superlative or add_runtime_meaning("the")}
            ]
            spellchecker = getattr(self, "spellchecker", None)
            if spellchecker is not None and spellchecker._supports_l_apostrophe_tail(noun):
                choices.append(
                    {
                        "word": f"l'{noun}",
                        "meaning": f"which is {noun_meaning}"
                        if noun_meaning
                        else "which is",
                    }
                )
            return choices
        return [{"word": definite, "meaning": add_runtime_meaning("the")}]

    def _belt_place_choices(self) -> list[dict[str, str]]:
        return [
            {"word": "belt", "meaning": "city"},
            {"word": "il-Belt", "meaning": "the city of Valletta"},
        ]

    def _superlative_meaning(self, meaning: str) -> str:
        irregular = {
            "better": "best",
            "bigger": "biggest",
            "worse": "worst",
            "less": "least",
            "more": "most",
            "farther": "farthest",
            "further": "furthest",
            "elder": "eldest",
        }
        converted: list[str] = []
        found_comparative = False

        for raw_part in str(meaning or "").split(","):
            part = raw_part.strip()
            lowered = part.casefold()
            if lowered in irregular:
                converted.append(irregular[lowered])
                found_comparative = True
            elif lowered.startswith("more "):
                converted.append(f"most {part[5:]}")
                found_comparative = True
            elif lowered.startswith("less "):
                converted.append(f"least {part[5:]}")
                found_comparative = True
            elif lowered.endswith("er") and len(part) > 3:
                converted.append(f"{part[:-2]}est")
                found_comparative = True
            else:
                converted.append(part)

        if not found_comparative:
            return ""
        return f"the {', '.join(converted)}"

    def literal_article_choices(
        self,
        article: str,
        noun: str,
        previous: str | None,
    ) -> list[dict[str, str]]:
        noun_meaning = self.meaning_index.meaning_for(noun)

        def add_runtime_meaning(prefix: str) -> str:
            return f"{prefix} {noun_meaning}" if noun_meaning else prefix

        # Keep the literal article family that the writer supplied.  A vowel
        # or apostrophe immediately before it contracts ``'il`` to ``'l``.
        to_article = (
            "l"
            if self.normalize(article).rstrip("-") == "l"
            or self.previous_ends_vowelish(previous)
            else "il"
        )
        literal_noun = (
            f"i{noun}" if self._requires_article_epenthetic_i(noun) else noun
        )
        return [
            {"word": f"'{to_article} {literal_noun}", "meaning": add_runtime_meaning("to a")},
            {"word": f"'{to_article}-{literal_noun}", "meaning": add_runtime_meaning("to the")},
        ]

    def num_choices(self, numeral: str, previous: str | None) -> list[dict[str, str]]:
        meaning = self.meaning_index.meaning_for(numeral)
        return [{"word": numeral, "meaning": f"the {meaning}" if meaning else "the"}]

    def _is_article_target(self, word: str) -> bool:
        normalized = self.normalize(word)
        if normalized in {"hawn", "hemm", "hinn", "i", "ek", "u", "ha", "na", "kom", "hom"}:
            return False
        if self.is_noun(word) or self.is_num(word):
            return True
        spellchecker = getattr(self, "spellchecker", None)
        if spellchecker is None:
            return False
        if spellchecker._correct_noun_possessive_suffix(normalized) == normalized:
            return True
        if getattr(spellchecker, "_accepted_article_english", lambda _word: False)(
            normalized
        ):
            return True
        if normalized in spellchecker.place_word_set:
            return True
        return self.is_adjective_like(normalized)

    def _starts_vowel_gh_or_h(self, word: str) -> bool:
        normalized = self.normalize(word)
        return bool(
            normalized
            and (
                normalized[0] in VOWELS
                or normalized.startswith(("għ", "gh", "h"))
            )
        )

    def _strict_dictionary_tail(self, word: str) -> str | None:
        spellchecker = getattr(self, "spellchecker", None)
        if spellchecker is None:
            return None
        normalized = self.normalize(word)
        if normalized in spellchecker.dictionary_set:
            return normalized
        # Possessive noun surfaces are generated rather than all stored as
        # standalone dictionary rows.  They are nevertheless valid article
        # tails and must never degrade into a missing ``None`` placeholder.
        possessive = spellchecker._correct_noun_possessive_suffix(normalized)
        if possessive == normalized:
            return normalized
        if normalized.startswith("i"):
            epenthetic_base = normalized[1:]
            if (
                self._requires_article_epenthetic_i(epenthetic_base)
                and epenthetic_base in spellchecker.dictionary_set
            ):
                return epenthetic_base
        for candidate in spellchecker._strict_lookup_variants(normalized):
            if candidate in spellchecker.dictionary_set:
                return candidate
        orthographic = getattr(spellchecker, "orthographic_generator", None)
        if orthographic is None:
            return None
        simplified_tails = self._tail_surface_variants(normalized)
        for candidate in simplified_tails:
            if candidate in spellchecker.dictionary_set:
                return candidate
            for strict_candidate in spellchecker._strict_lookup_variants(candidate):
                if strict_candidate in spellchecker.dictionary_set:
                    return strict_candidate
        helper_names = (
            "dictionary_shortcut_variants",
            "dictionary_gh_priority_variants",
            "dictionary_final_gh_h_hbar_variants",
            "dictionary_i_ie_variants",
        )
        for helper_name in helper_names:
            helper = getattr(orthographic, helper_name, None)
            if helper is None:
                continue
            for surface in (normalized, *simplified_tails):
                for candidate in helper(surface):
                    candidate = self.normalize(candidate)
                    if candidate in spellchecker.dictionary_set:
                        return candidate

        # Whole-word correction is deliberately last.  In an already parsed
        # article phrase, a dictionary-backed keyboard repair is more reliable
        # than a broad fuzzy candidate such as ``cella -> bella'``.
        corrected = self.normalize(spellchecker.correct_word(normalized))
        if corrected != normalized and corrected in spellchecker.dictionary_set:
            return corrected
        return None

    def _tail_surface_variants(self, word: str) -> list[str]:
        """Generate narrow, dictionary-gated repairs for an article tail."""
        spellchecker = getattr(self, "spellchecker", None)
        if spellchecker is None:
            return []

        graphemes = spellchecker._graphemes(self.normalize(word))
        variants: list[str] = []

        for index, grapheme in enumerate(graphemes):
            if grapheme not in CONSONANTS:
                continue
            if (
                (index and graphemes[index - 1] == grapheme)
                or (index + 1 < len(graphemes) and graphemes[index + 1] == grapheme)
            ):
                continue
            variants.append(
                spellchecker._from_graphemes(
                    graphemes[:index] + [grapheme] + graphemes[index:]
                )
            )

        for index in range(len(graphemes) - 2):
            if (
                graphemes[index] == "i"
                and graphemes[index + 1] == "j"
                and graphemes[index + 2] in CONSONANTS
            ):
                variants.append(
                    spellchecker._from_graphemes(
                        graphemes[: index + 1] + graphemes[index + 2 :]
                    )
                )

        return list(dict.fromkeys(variants))

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

        spellchecker = getattr(self, "spellchecker", None)
        place_display = None
        if spellchecker is not None:
            place_display = spellchecker._exact_place_word(noun)
            if place_display:
                if prefix in {"ta", "ta'"}:
                    return f"ta' {place_display}"
                if prefix in {"minn", "min", "mil", "mid", "mill"}:
                    return f"minn {place_display}"
                if prefix in {"għal", "ghal", "għall", "ghall"}:
                    return f"għal {place_display}"
                if prefix in {"bħal", "bhal", "bħall", "bhall"}:
                    return f"bħal {place_display}"
                if prefix in {"ma", "ma'", "mal"}:
                    return f"ma' {place_display}"
                if prefix in {"fi", "fil", "fl"}:
                    return f"f'{place_display}" if self._starts_vowel_gh_or_h(noun) else f"fi {place_display}"
                if prefix in {"bi", "bil", "bl"}:
                    return f"b'{place_display}" if self._starts_vowel_gh_or_h(noun) else f"bi {place_display}"

        surface_noun = place_display or noun

        if not noun or not self._is_article_target(noun):
            return None
        starts_vowelish = self._starts_vowel_gh_or_h(noun)

        aliases = {
            "tal": "tal",
            "mal": "mal",
            "bħal": "bħall", "bhal": "bħall",
            "bħall": "bħall", "bhall": "bħall",
            "bil": "bil", "bl": "bil",
            "fil": "fil", "fl": "fil",
            "fis": "fis",
            "fir": "fir",
            "minn": None, "mil": "mill", "mid": "mill", "mill": "mill",
            "mic": "mill", "miċ": "mill",
            "għat": "għat", "ghat": "għat",
            "għal": "għall", "ghal": "għall",
            "għall": "għall", "ghall": "għall",
            "sa": "sal", "sal": "sal",
            "lil": "lill", "lill": "lill",
        }
        canonical = self._assimilated_prefix_canonical(prefix) or aliases.get(prefix)
        if canonical:
            if canonical == "fis":
                if starts_vowelish:
                    return f"fi-{surface_noun}"
                if noun[0] in SUN_LETTERS:
                    return f"fis-{surface_noun}"
                return f"fis-{surface_noun}"
            if canonical == "fir":
                if starts_vowelish:
                    return f"fi-{surface_noun}"
                return f"fir-{surface_noun}"
            vowel_forms = {
                "tal": "tal", "mal": "mal", "bħall": "bħall",
                "bil": "bl", "fil": "fl",
                "mill": "mill", "fir": "fi", "għall": "għall", "għat": "għat", "sal": "sal", "lill": "lill",
            }
            sun_stems = {
                "tal": "ta", "mal": "ma", "bħall": "bħa",
                "bil": "bi", "fil": "fi",
                "mill": "mi", "fir": "fir", "għall": "għa", "għat": "għa", "sal": "sa", "lill": "li",
            }
            if starts_vowelish:
                return f"{vowel_forms[canonical]}-{surface_noun}"
            if noun[0] in SUN_LETTERS:
                return f"{sun_stems[canonical]}{noun[0]}-{surface_noun}"
            return f"{canonical}-{surface_noun}"

        if prefix in {"xil", "x'l"}:
            return f"x'l-{surface_noun}" if starts_vowelish else f"xi{self.assimilate('il-', noun)}{surface_noun}"
        if prefix in {"il", "l", "ir", "in", "is", "it", "id", "iċ", "ic", "iż", "iz"} or prefix in SUN_LETTERS:
            if prefix == "l" and starts_vowelish:
                return f"l-{surface_noun}"
            return f"{self.assimilate('il-', noun)}{surface_noun}"
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
        choices: list[dict[str, str]] = []

        def add(word: str, choice_meaning: str | None = None) -> None:
            normalized = self.normalize(word)
            if not normalized:
                return
            if any(self.normalize(choice["word"]) == normalized for choice in choices):
                return
            choices.append(
                {
                    "word": word,
                    "meaning": choice_meaning if choice_meaning is not None else meaning,
                }
            )

        add(corrected, meaning)
        canonical = self._assimilated_prefix_canonical(prefix) or {
            "bħal": "bħall",
            "bhal": "bħall",
            "bħall": "bħall",
            "bhall": "bħall",
            "għal": "għall",
            "ghal": "għall",
            "għall": "għall",
            "ghall": "għall",
        }.get(self.normalize(prefix).rstrip("-"))
        if canonical == "bħall":
            add(f"bħal {noun}", meaning)
        elif canonical == "għall":
            add(f"għal {noun}", meaning)
        return choices

    def _bare_preposition_article_choice(
        self,
        prefix: str,
        noun: str,
    ) -> ArticlePhraseSuggestion | None:
        normalized_prefix = self.normalize(prefix).rstrip("-")
        if normalized_prefix not in {
            "bħal", "bhal", "bħall", "bhall",
            "għal", "ghal", "għall", "ghall",
        }:
            return None

        corrected_noun = self._strict_dictionary_tail(noun) or noun
        if not self._is_article_target(corrected_noun):
            spellchecker = getattr(self, "spellchecker", None)
            if spellchecker is not None:
                candidate = self.normalize(spellchecker.correct_word(noun))
                if candidate != noun and self._is_article_target(candidate):
                    corrected_noun = candidate
        if not self._is_article_target(corrected_noun):
            return None

        article_prefix = (
            "bħall"
            if normalized_prefix in {"bħal", "bhal", "bħall", "bhall"}
            else "għall"
        )
        literal_prefix = "bħal" if article_prefix == "bħall" else "għal"
        article_form = self.preposition_article_form(article_prefix, corrected_noun)
        if normalized_prefix in {"bħall", "bhall", "għall", "ghall"}:
            corrected = article_form
        else:
            corrected = f"{literal_prefix} {corrected_noun}"
        if not corrected:
            return None

        meaning = self.meaning_index.meaning_for(corrected_noun)
        choices: list[dict[str, str]] = []
        for word in (corrected, article_form, f"{literal_prefix} {corrected_noun}"):
            if word and all(
                self.normalize(choice["word"]) != self.normalize(word)
                for choice in choices
            ):
                choices.append({"word": word, "meaning": meaning})
        return ArticlePhraseSuggestion(0, 2, corrected, choices)

    def match_split_article(
        self,
        words: list[WordToken],
        index: int,
    ) -> ArticlePhraseSuggestion | None:
        if index + 1 >= len(words):
            return None

        article = self.normalize(words[index].text).rstrip("-")
        noun = self.normalize(words[index + 1].text)
        article_canonical = self._assimilated_prefix_canonical(article)
        if article_canonical and "-" in noun:
            typed_tail_prefix, possible_tail = noun.split("-", 1)
            if (
                possible_tail
                and typed_tail_prefix
                and typed_tail_prefix[-1:] == article[-1:]
            ):
                noun = possible_tail
        corrected_noun = self.normalize(self._strict_dictionary_tail(noun) or noun)

        if not self._is_article_target(corrected_noun):
            spellchecker = getattr(self, "spellchecker", None)
            if spellchecker is not None:
                candidate = self.normalize(spellchecker.correct_word(noun))
                if candidate != noun and self._is_article_target(candidate):
                    corrected_noun = candidate

        if not self._is_article_target(corrected_noun):
            return None

        previous = words[index - 1].text if index > 0 else None

        bare_preposition = self._bare_preposition_article_choice(article, corrected_noun)
        if bare_preposition is not None:
            return ArticlePhraseSuggestion(
                index,
                index + 2,
                bare_preposition.corrected,
                bare_preposition.choices,
            )

        if article in {"għar", "ghar"} and corrected_noun.startswith("r"):
            corrected = f"għar-{corrected_noun}"
            return ArticlePhraseSuggestion(index, index + 2, corrected, [])

        if article in {
            "il", "l", "ic", "iċ", "id", "in", "ir", "is", "it",
            "ix", "iz", "iż",
        } or article in SUN_LETTERS:
            corrected = (
                self._adjectival_article_surface(corrected_noun)
                if article == "l" or self.is_adjective_like(corrected_noun)
                else self.corrected_article_phrase(
                    article,
                    corrected_noun,
                    previous,
                )
            )
            choices = self.phrase_choices(corrected_noun, previous)
            if article == "il" and corrected_noun == "belt":
                choices = self._belt_place_choices()
            if article == "l" or self._requires_article_epenthetic_i(corrected_noun):
                choices.extend(self.literal_article_choices(article, corrected_noun, previous))
            return ArticlePhraseSuggestion(index, index + 2, corrected, choices)

        if article_canonical or article in {
            "tal", "mal", "bil", "fil", "fis", "lill", "xil", "mil", "mid", "mill",
            "għal", "ghal", "għall", "ghall", "għat", "ghat",
            "bħal", "bhal", "bħall", "bhall", "sal", "mic", "miċ",
        }:
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

        spellchecker = getattr(self, "spellchecker", None)
        noun_display = (
            spellchecker._exact_place_word(corrected_noun)
            if spellchecker is not None
            else None
        ) or corrected_noun

        assimilated = self._assimilated_prefix_surface(prefix, corrected_noun)
        if assimilated is not None:
            surface_prefix, canonical = assimilated
            choices = (
                self.preposition_article_choices(canonical, corrected_noun, previous)
                if canonical in {"tal", "bil", "fil", "mill", "mal", "bħall", "għall", "għat"}
                else []
            )
            return ArticlePhraseSuggestion(
                0,
                1,
                f"{surface_prefix}-{noun_display}",
                choices,
            )

        sun_stems = {
            "tal": "ta", "mal": "ma", "bħall": "bħa", "bil": "bi", "fil": "fi",
            "mill": "mi", "għall": "għa", "sal": "sa", "lill": "li",
        }
        if corrected_noun[0] in SUN_LETTERS:
            for canonical, stem in sun_stems.items():
                if prefix == stem + corrected_noun[0]:
                    corrected = self.preposition_article_form(
                        canonical,
                        corrected_noun,
                    )
                    if corrected:
                        return ArticlePhraseSuggestion(
                            0,
                            1,
                            corrected,
                            self.preposition_article_choices(
                                canonical,
                                corrected_noun,
                                previous,
                            ),
                        )

        if prefix in SUN_LETTERS and not (
            prefix == "l" and self.is_adjective_like(corrected_noun)
        ):
            return ArticlePhraseSuggestion(
                0,
                1,
                f"{prefix}-{noun_display}",
                [],
            )

        if prefix in {"il", "l", "din", "dan"} or prefix.startswith("i"):
            corrected = (
                self._adjectival_article_surface(corrected_noun)
                if self.is_adjective_like(corrected_noun)
                else self.corrected_article_phrase(
                    prefix,
                    corrected_noun,
                    previous,
                )
            )
            choices = self.phrase_choices(corrected_noun, previous)
            if prefix == "il" and corrected_noun == "belt":
                choices = self._belt_place_choices()
            if prefix == "l" or prefix.startswith("i"):
                literal_article = "l" if prefix == "l" else "il"
                choices.extend(
                    self.literal_article_choices(
                        literal_article,
                        corrected_noun,
                        previous,
                    )
                )
            return ArticlePhraseSuggestion(0, 1, corrected, choices)

        if prefix in {
            "tal", "mal", "bil", "fil", "fis", "lill", "xil", "mil", "mis", "mill",
            "għal", "ghal", "għall", "ghall", "għat", "ghat",
            "bħal", "bhal", "bħall", "bhall", "sal", "mic", "miċ",
        }:
            corrected = self.preposition_article_form(prefix, corrected_noun)
            if corrected:
                choices = self.preposition_article_choices(prefix, corrected_noun, previous)
                return ArticlePhraseSuggestion(0, 1, corrected, choices)

        return None

    def _assimilated_prefix_surface(
        self,
        prefix: str,
        noun: str,
    ) -> tuple[str, str] | None:
        """Return the canonical compact prefix for a validated article tail."""
        typed_key = self._assimilated_prefix_key(prefix)
        if not typed_key or not noun:
            return None

        initial = noun[0]
        for canonical, sun_stem in ASSIMILATED_PREFIX_FAMILIES.items():
            expected = sun_stem + initial if initial in SUN_LETTERS else canonical
            if self._assimilated_prefix_canonical(prefix) == canonical:
                return expected, canonical
        return None

    def _assimilated_prefix_canonical(self, prefix: str) -> str | None:
        """Identify a compact preposition independently of its typed sun letter."""
        typed_key = self._assimilated_prefix_key(prefix)
        if not typed_key:
            return None

        compact_sun_letters = {"c", "d", "n", "r", "s", "t", "x", "z"}
        for canonical, sun_stem in ASSIMILATED_PREFIX_FAMILIES.items():
            canonical_key = self._assimilated_prefix_key(canonical)
            stem_key = self._assimilated_prefix_key(sun_stem)
            # A bare stem (ta, ma, bi, fi...) is a preposition, not an
            # already fused article.  It can only fuse once an explicit
            # following article has been parsed by the caller.  Treating it
            # as ``tal-/mal-/...`` here swallowed legitimate phrases such as
            # ``ta' għajnuna``.
            if typed_key == canonical_key:
                return canonical
            if (
                typed_key.startswith(stem_key)
                and len(typed_key) == len(stem_key) + 1
                and typed_key[-1] in compact_sun_letters
            ):
                return canonical
            if (
                typed_key.startswith(stem_key)
                and len(typed_key) == len(stem_key) + 2
                and typed_key[-1] == typed_key[-2]
                and typed_key[-1] in compact_sun_letters
            ):
                return canonical
        return None

    @staticmethod
    def _assimilated_prefix_key(value: str) -> str:
        return (
            str(value or "")
            .casefold()
            .replace("għ", "gh")
            .translate(str.maketrans({"ġ": "g", "ħ": "h", "ċ": "c", "ż": "z"}))
        )

    def match_preposition_article_contraction(
        self,
        words: list[WordToken],
        index: int,
    ) -> ArticlePhraseSuggestion | None:
        if index + 1 >= len(words):
            return None

        preposition = self.normalize(words[index].text)
        next_word = self.normalize(words[index + 1].text)
        spellchecker = getattr(self, "spellchecker", None)
        if (
            spellchecker is not None
            and getattr(spellchecker, "_fixed_time_expression_word", lambda _word: None)(
                next_word
            )
        ):
            return None
        prefix_map = {
            "ta": "tal",
            "ta'": "tal",
            "ma": "mal",
            "ma'": "mal",
            "bħal": "bħall",
            "bhal": "bħall",
            "bi": "bil",
            "fi": "fil",
            "lil": "lill",
            "għal": "għall",
            "ghal": "għall",
            "sa": "sal",
        }
        canonical_prefix = prefix_map.get(preposition)
        if not canonical_prefix:
            return None

        consumed = 2
        hyphenated_article_prefixes = {
            "il", "l", "ic", "iċ", "id", "in", "ir", "is", "it",
            "ix", "iz", "iż", *SUN_LETTERS,
        }
        if (
            "-" in next_word
            and next_word.split("-", 1)[0] in hyphenated_article_prefixes
        ):
            noun = next_word.split("-", 1)[1]
        elif next_word in {
            "il", "l", "ic", "iċ", "id", "in", "ir", "is", "it",
            "ix", "iz", "iż", *SUN_LETTERS,
        } and index + 2 < len(words):
            noun = self.normalize(words[index + 2].text)
            consumed = 3
        else:
            return None

        corrected_noun = self._strict_dictionary_tail(noun)
        if corrected_noun is None:
            spellchecker = getattr(self, "spellchecker", None)
            if spellchecker is not None:
                corrected_place = spellchecker._correct_place_word(
                    words[index + consumed - 1].text
                )
                if corrected_place:
                    corrected_noun = self.normalize(corrected_place)
        noun = corrected_noun or noun

        corrected = self.preposition_article_form(canonical_prefix, noun)
        if not corrected:
            return None
        choices = self.preposition_article_choices(canonical_prefix, noun, None)
        if preposition in {"ta", "ta'"}:
            literal_noun = (
                f"i{noun}"
                if self._requires_article_epenthetic_i(noun)
                else noun
            )
            choices.extend(
                [
                    {
                        "word": f"ta 'l {literal_noun}",
                        "meaning": self.meaning_index.meaning_for(noun),
                    },
                    {
                        "word": f"ta' l-{literal_noun}",
                        "meaning": self.meaning_index.meaning_for(noun),
                    },
                ]
            )
        return ArticlePhraseSuggestion(index, index + consumed, corrected, choices)

    def match_compact_preposition_article(
        self,
        word: str,
    ) -> ArticlePhraseSuggestion | None:
        normalized = self.normalize(word)
        compact_prefixes = (
            ("għall", "għal"),
            ("ghall", "għal"),
            ("għal", "għal"),
            ("ghal", "għal"),
            ("għat", "għat"),
            ("ghat", "għat"),
            ("mill", "mi"),
            ("miss", "mi"),
            ("mis", "mi"),
            ("mid", "mi"),
            ("sal", "sa"),
            ("lill", "lil"),
            ("tal", "ta"),
            ("mal", "ma"),
            ("bil", "bi"),
            ("fil", "fi"),
            ("bħall", "bħal"),
            ("bhall", "bħal"),
            ("bħal", "bħal"),
            ("bhal", "bħal"),
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

    def match_compact_definite_article(
        self,
        word: str,
        *,
        previous: str | None,
    ) -> ArticlePhraseSuggestion | None:
        """Recognize unhyphenated ``l-i...`` and ``il-i...`` article forms.

        This is limited to the epenthetic ``mC``/``sC`` article class.  The
        dictionary-backed tail check prevents broad ``li...`` words from being
        mistaken for a detached article.
        """
        normalized = self.normalize(word)
        if "-" in normalized or "'" in normalized:
            return None

        for article in ("il", "l"):
            if not normalized.startswith(article) or len(normalized) <= len(article):
                continue
            tail = normalized[len(article) :]
            noun = self._strict_dictionary_tail(tail)
            if noun is None or not self._requires_article_epenthetic_i(noun):
                continue
            if not self._is_article_target(noun):
                continue

            corrected = self.corrected_article_phrase(article, noun, previous)
            choices = self.phrase_choices(noun, previous)
            choices.extend(self.literal_article_choices(article, noun, previous))
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
