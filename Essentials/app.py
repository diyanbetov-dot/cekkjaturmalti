import json
import os
import re
import unicodedata
from Essentials.dictionary_meanings import (
    MeaningIndex,
    extract_meaning_from_payload,
    format_suffix_candidate_meaning,
)
from collections import defaultdict
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterable
from Essentials.helpers.article_phrase_rules import MalteseArticlePhraseRules, WordToken
from Essentials.helpers.spellchecker_types import ScoreRow, UnifiedMatch
from Essentials.helpers.fused_preposition_rules import MalteseFusedPrepositionRules
from Essentials.helpers.suffix_generator import MalteseSuffixGenerator
from Essentials.helpers.orthographic_generator import MalteseOrthographicGenerator
from Essentials.helpers.doubled_letter_generator import MalteseDoubledLetterGenerator
from flask import Flask, jsonify, request, send_from_directory


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

FINAL_DICS_DIR = BASE_DIR / "finaldics"
EU_COUNTRIES_DIC = FINAL_DICS_DIR / "eu_countries.dic"

DICTIONARY_FILES = sorted(
    path
    for path in FINAL_DICS_DIR.glob("*.dic")
    if path.name not in {"places.dic", EU_COUNTRIES_DIC.name}
)

MAX_TEXT_LENGTH = 10_000
MAX_WORD_LENGTH = 100

# Place fuzzy correction is isolated behind this switch. Set it to False to
# keep exact/shortcut place recognition while disabling typo-based place lookup.
PLACE_FUZZY_CORRECTION_ENABLED = True

class UniversalMalteseSpellchecker:
    """
    Maltese spellchecker with strict priority stages plus soft scoring.

    Main design decisions:
    - għ is treated as one logical Maltese letter for distance and anchors.
    - gh -> għ is NOT globally applied to the whole text. It is only used as a
      candidate lookup variant, so English words and names are not blindly changed.
    - Vowel count/vector is important, but it is not the only truth. It is used
      as a priority filter first, then as part of a score.
    - Special stages handle missing/extra għ and h near vowels.
    """

    VOWELS = set("aeiouàèìòù")

    # Maltese-like word tokens, including apostrophes and hyphens inside words.
    WORD_PATTERN = re.compile(
        r"[^\W\d_]+(?:[-'\u2018\u2019\u02bc][^\W\d_]+)*(?:['\u2018\u2019\u02bc])?",
        re.UNICODE,
    )

    # English is intentionally checked only when users mark it with paired
    # straight single quotes. The boundary guards stop Maltese apostrophes
    # like ma', t'ommkom, and ssibu'x from being mistaken for English quotes.
    ENGLISH_QUOTES_PATTERN = re.compile(
        r"(?<![^\W\d_])['\u2018](?P<inner>[^'\u2019\n]+)['\u2019](?![^\W\d_])",
        re.UNICODE,
    )

    ENGLISH_MAPPINGS = {
        "washing machine": ["magna tal-ħasil"],
        "roundabout": ["dawwara"],
        "traffic light": ["dawl tat-traffiku"],
        "bus stop": ["waqfa"],
        "bus terminal": ["venda"],
        "sandwich": ["ħobża"],
        "indicator": ["indikatur"],
        "parking": ["parkeġġ"],
        "parking lot": ["parkeġġ"],
        "nutckracker": ["krakku"],
        "bowl": ["skutella"],
        "spoilsport": ["fottafesti"],
        "database": ["ġabradati"],
        "surgery": ["kirurġija"],
        "website": ["sit web"],
        "seatbelt": ["ċinturin tas-sigurtà"],
        "lecture": ["lettura"],
        "lecturer": ["lettur"],
        "bullying": ["bulliżmu"],
        "whiteboard": ["bord"],
        "squirrel": ["skajjotlu"],
        "trend": ["xejra"],
        "cybersecurity": ["ċibersigurtà"],
        "date": ["ħarġa romantika"],
        "meeting": ["laqgħa"],
        "suit": ["libsa"],
        "headphones": ["kaffli"],
        "social media": ["midja soċjali"],
        "selfie": ["stessu"],
        "snowman": ["borrinu"],
        "snowball": ["ballun tas-silġ"],
        "injection": ["tilqima"],
        "vaccine": ["tilqima"],
        "north": ["tramuntana"],
        "south": ["nofsinhar"],
        "east": ["lvant"],
        "west": ["punent"],
        "northwest": ["majjistral"],
        "northeast": ["grigal"],
        "southwest": ["lbiċ"],
        "southeast": ["xlokk"],
        "cancer": ["il-Qabru"],
        "scorpio": ["l-Għakreb"],
        "taurus": ["il-Fart"],
        "gemini": ["it-Tewmin"],
        "virgo": ["ix-Xebba"],
        "libra": ["il-Miżien"],
        "pisces": ["il-Ħut"],
        "capricorn": ["il-Gidi l-Kbir"],
        "aquarius": ["iż-Żir"],
        "aries": ["il-Wott"],
        "leo": ["id-Dorbies"],
        "sagittarius": ["il-Qaws"],
        "new moon": ["qamar mitluf"],
        "full moon": ["qamar kwinta"],
        "waxing cresent": ["qamar ġdid"],
        "waning gibbous": ["qamar muqsar"],
        "waxing gibbous": ["qamar mużqaq"],
        "waning cresent": ["qamar xiħ"],
        "first quarter": ["l-ewwel kwart"],
        "last quarter": ["l-aħħar kwart"],
        "background": ["sfond"],
        "genie": ["ġinn"],  # The mythical creature, not the software tool.
        "algorithm": ["algoritmu"],
        "shark": ["kelb il-baħar"]
    }

    # Noun possessive suffixes can be tested conservatively before enabling
    # them across every tagged noun.
    #
    # Use:
    #   "manual"    -> only nouns listed in NOUN_POSSESSIVE_MANUAL_BASES
    #   "automatic" -> every dictionary entry tagged as a noun
    #   "off"       -> disable noun possessive suffix generation
    NOUN_POSSESSIVE_MODE = "automatic"

    # Add noun bases here while testing manual mode.
    # Examples:
    #   ħajja -> ħajti, ħajtek, ħajtu...
    #   dar   -> dari, darek/darok, daru...
    #   omm   -> ommi, ommek/ommok, ommu...
    NOUN_POSSESSIVE_MANUAL_BASES = {
        "ħajja",
        "dar",
        "omm",
        "missier",
        "ħabib",
        "ħabiba",
        "saħħa",
        "moħħ",
    }

    # Example custom verb tags:
    #   ibgħat/T-bgħt-F1-IMP-2S
    # Hunspell affix flags such as /AB are ignored as tags.
    PARADIGM_TAG_PATTERN = re.compile(r"^[^-\s/]+-[^-\s/]+-F\d+(?:-|$)")

    # Conservative suffix repairs. These are variants, not global replacements.
    # Keep longer suffixes before shorter ones.
    SUFFIX_REPAIRS = {
        "ulhom": "hulhom",
        "ulkom": "hulkom",
        "ulna": "hulna",
        "ulek": "hulek",
        "uli": "huli",
        "ila": "ilha",
        "ilom": "hielhom",
        "ulom": "hulhom",
        "ija": "iha",
        "ja": "ha",
        "ijom": "ihom",
        "jom": "hom",
        "iju": "ihu",
        "jul": "hul",
        "uwom": "uhom",
    }

    # Last-resort whole-word exceptions only. Prefer dictionary data or
    # orthographic/suffix rules for anything that can be modelled generally.
    MANUAL_WORD_REPAIRS = {
        "atijulu": "agħtihulu",
        "edd": "qed",
        "laqi": "lagħaqi",
        "minom": "minnhom",
        "possiblil": "possibbli",
        "qas": "lanqas",
        "tikbi": "tikbi",
        "nikbi": "nikbi",
        "jikbi": "jikbi",
        "tikbu": "tikbu",
        "nikbu": "nikbu",
        "jikbu": "jikbu",
        "uwx": "hux",
        "uwajma": "uu ajma",
        "uwijwa": "uwiwa",
        "ijwa": "iwa",
        "ijskom": "qiskom",
        "di": "din",
        "da": "dan",
        "xol": "xogħol",
        "nom": "ngħum",
        "nowm": "ngħum"
    }

    MANUAL_WORD_SUGGESTIONS = {
        "ilbierah": ("ilbieraħ",),
        "shark": ("xark",),
    }

    MANUAL_EJD_AJD_TAILS = (
        "",
        "u",
        "x",
        "ux",
        "li",
        "lu",
        "la",
        "lek",
        "lok",
        "lkom",
        "lna",
        "lhom",
        "uli",
        "ulu",
        "ula",
    )

    MANUAL_ENDING_REPAIRS = (
        ("ejd", "għid"),
        ("ajd", "għid"),
        ("ijgha", "iegħe"),
        ("ijgħa", "iegħe"),
        ("ijej", "iegħi"),
        ("ijaj", "iegħi"),
        ("ijew", "iegħu"),
        ("ijaw", "iegħu"),
        ("ije", "iegħe"),
    )

    MANUAL_SEQUENCE_REPAIRS = (
        ("ijgha", "iegħe"),
        ("ijgħa", "iegħe"),
    )

    MANUAL_VERB_GH_ENDING_REPAIRS = (("ahhom", "agħhom"),)

    CAPITALIZED_PLACE_CANDIDATE_LIMIT = 5
    SENTENCE_INITIAL_CANDIDATE_LIMIT = 10
    COMPACT_XI_ARTICLE_PREFIXES = (
        "x'l-",
        "x'l",
        "xl",
        "xil-",
        "xir-",
        "xis-",
        "xiż-",
        "xiz-",
        "xit-",
        "xid-",
        "xin-",
        "xiċ-",
        "xix-",
    )

    LEXICALIZED_FORM_RULES = (
        ("bilqiegħda", ("bil-qiegħda",)),
        ("bilwieqfa", ("bil-wieqfa",)),
        ("bilkemm", ()),
        ("biżżejjed", ("biż-żejjed",)),
        ("kulma", ("kull ma",)),
        ("bħalma", ("bħal ma",)),
        ("għalfejn", ("għal fejn",)),
        ("għalissa", ()),
        ("għalkemm", ()),
        ("għalkollox", ("għal kollox",)),
        ("qabelxejn", ("qabel xejn",)),
        ("kulħadd", ("kull ħadd",)),
        ("lanqas", ("l-anqas",)),
        ("l-inqas", ()),
        ("għalxejn", ("għal xejn",)),
    )

    LEXICALIZED_ANALYTIC_MEANING_BASES = {
        "bil-qiegħda": "qiegħda",
        "bil-wieqfa": "wieqfa",
        "biż-żejjed": "żejjed",
        "kull ma": "ma",
        "bħal ma": "ma",
        "għal fejn": "fejn",
        "għal kollox": "kollox",
        "qabel xejn": "xejn",
        "kull ħadd": "ħadd",
        "l-anqas": "anqas",
        "l-inqas": "inqas",
        "għal xejn": "xejn",
    }

    NUMBER_FORM_REPAIRS = {
        "erba": ("erbgħa", "erba'"),
        "seba": ("sebgħa", "seba'"),
        "disgha": ("disgħa",),
        "disgħa": ("disgħa",),
        "seta": ("seta'", "setgħa"),
    }

    SOCIAL_COMMENT_REPAIRS = {
        "kolla": "kollha",
        "issabat": "issabbat",
        "xem": "x'hemm",
        "siqa": "sieqha",
        "tiggagbina": "tiġġakbina",
        "twehhel": "tweħħel",
        "tatakhom": "tattakkhom",
        "themm": "t'hemm",
        "jithaq": "jidħaq",
        "hireg": "ħiereġ",
        "ukoll": "wkoll",
        "alkemm": "għalkemm",
        "jafuwha": "jafuha",
        "mghamila": "m'għamilha",
        "em": "hemm",
        "memx": "m'hemmx",
        "m'hemmx": "m'hemmx",
        "terbahx": "tirbaħx",
        "f'ghoxx": "f'għoxx",
    }

    def __init__(
        self,
        dictionary_words: Iterable[str] | None = None,
        dictionary_files: Iterable[Path] | None = None,
    ) -> None:
        self._manual_noun_bases = frozenset(
            self._normalize_word(base) for base in self.NOUN_POSSESSIVE_MANUAL_BASES
        )
        self._sorted_suffix_repairs = tuple(
            sorted(
                self.SUFFIX_REPAIRS.items(),
                key=lambda item: len(item[0]),
                reverse=True,
            )
        )
        self.dictionary: list[str] = []
        self.dictionary_set: set[str] = set()
        self.place_entries: list[str] = []
        self.place_words: list[str] = []
        self.place_word_set: set[str] = set()
        self.place_word_display: dict[str, str] = {}
        self.place_word_buckets: dict[tuple[str, int], list[str]] = defaultdict(list)
        self.place_word_anchor_map: dict[str, list[str]] = defaultdict(list)
        self.place_word_anchor_buckets: dict[tuple[str, int], list[str]] = defaultdict(list)
        self.place_phrases: list[str] = []
        self.place_phrase_display: dict[str, str] = {}
        self.place_phrase_anchor_map: dict[str, list[str]] = defaultdict(list)
        self.place_phrase_anchor_buckets: dict[tuple[str, int], list[str]] = defaultdict(list)
        self.country_english_to_maltese: dict[str, str] = {}
        self.country_english_display: dict[str, str] = {}
        self.country_maltese_to_english: dict[str, str] = {}
        self.country_maltese_names: list[str] = []

        # surface form -> all paradigm keys for that surface form
        self.word_tags: dict[str, set[str]] = defaultdict(set)

        # paradigm key -> all surface forms in that paradigm
        self.paradigm_forms: dict[str, list[str]] = defaultdict(list)

        # consonant anchor -> surface forms
        self.anchor_map: dict[str, list[str]] = defaultdict(list)

        # Cached metadata
        self.word_tokens: dict[str, list[str]] = {}
        self.word_lengths: dict[str, int] = {}
        self.word_vowel_counts: dict[str, int] = {}
        self.word_vowel_slots: dict[str, list[tuple[int, str]]] = {}
        self.word_anchors: dict[str, str] = {}
        self._missing_h_verb_repairs: dict[str, str] | None = None

        raw_entries: list[tuple[str, str | None]] = []

        if dictionary_words:
            raw_entries.extend((word, None) for word in dictionary_words)

        if dictionary_files:
            raw_entries.extend(self._load_dictionary_files(list(dictionary_files)))
        raw_entries.extend(self._load_eu_single_word_entries(EU_COUNTRIES_DIC))

        places_file = FINAL_DICS_DIR / "places.dic"
        raw_entries.extend(self._load_eu_single_word_entries(places_file))

        seen_words: set[str] = set()
        seen_paradigm_forms: dict[str, set[str]] = defaultdict(set)

        for word, tag in raw_entries:
            normalized = self._normalize_word(word)
            if not normalized:
                continue

            if normalized not in seen_words:
                self.dictionary.append(normalized)
                seen_words.add(normalized)

            if tag:
                self.word_tags[normalized].add(tag)
                if (
                    self._is_paradigm_tag(tag)
                    and normalized not in seen_paradigm_forms[tag]
                ):
                    self.paradigm_forms[tag].append(normalized)
                    seen_paradigm_forms[tag].add(normalized)

        self.dictionary_set = set(self.dictionary)
        self._load_country_place_index(FINAL_DICS_DIR / "eu_countries.json")
        self._load_places_dictionary(FINAL_DICS_DIR / "places.dic")
        self._build_word_metadata()
        self._build_anchor_index()

        print(
            f"Loaded {len(self.dictionary)} dictionary words, "
            f"{len(self.paradigm_forms)} paradigms."
        )

    # ------------------------------------------------------------------
    # Normalisation/tokenisation
    # ------------------------------------------------------------------

    def _normalize_word(self, word: str) -> str:
        """Lower-case, NFC-normalise, and unify apostrophe variants."""
        return self._normalize_word_cached(str(word))

    @staticmethod
    @lru_cache(maxsize=65536)
    def _normalize_word_cached(word: str) -> str:
        return (
            unicodedata.normalize("NFC", str(word).strip().lower())
            .replace("\u2019", "'")
            .replace("\u2018", "'")
            .replace("\u02bc", "'")
        )

    def _graphemes(self, word: str) -> list[str]:
        """Split a word into rough graphemes, keeping għ as one item."""
        return list(self._graphemes_cached(self._normalize_word(word)))

    @staticmethod
    @lru_cache(maxsize=32768)
    def _graphemes_cached(word: str) -> tuple[str, ...]:
        out: list[str] = []
        i = 0
        while i < len(word):
            if word.startswith("għ", i):
                out.append("għ")
                i += 2
            else:
                out.append(word[i])
                i += 1
        return tuple(out)

    def _from_graphemes(self, graphemes: Iterable[str]) -> str:
        return "".join(graphemes)

    def _letter_tokens_raw(self, word: str) -> list[str]:
        """
        Splits a Maltese word into logical spelling tokens.
        għ is represented internally as ʕ so it has token length 1.
        Apostrophes and hyphens are ignored for distance/anchor purposes.
        """
        tokens: list[str] = []
        for g in self._graphemes(word):
            if g == "għ":
                tokens.append("ʕ")
            elif len(g) == 1 and g.isalpha():
                tokens.append(g)
        return tokens

    def _letter_tokens(self, word: str) -> list[str]:
        normalized = self._normalize_word(word)
        return self.word_tokens.get(normalized) or self._letter_tokens_raw(normalized)

    def _build_word_metadata(self) -> None:
        for word in self.dictionary:
            tokens = self._letter_tokens_raw(word)
            self.word_tokens[word] = tokens
            self.word_lengths[word] = len(tokens)
            self.word_vowel_counts[word] = sum(1 for t in tokens if t in self.VOWELS)
            self.word_vowel_slots[word] = [
                (i, t) for i, t in enumerate(tokens) if t in self.VOWELS
            ]
            self.word_anchors[word] = self._extract_consonant_anchor_from_tokens(tokens)

    def _build_anchor_index(self) -> None:
        for word in self.dictionary:
            self.anchor_map[self.word_anchors[word]].append(word)

    def _starts_vowel_gh_or_h(self, word: str) -> bool:
        normalized = self._normalize_word(word)
        if not normalized:
            return False
        return normalized.startswith(
            (
                "a",
                "e",
                "i",
                "o",
                "u",
                "à",
                "è",
                "ì",
                "ò",
                "ù",
                "għ",
                "gh",
                "h",
                "ħ",
            )
        )

    def _is_verb_tagged_word(self, word: str) -> bool:
        normalized = self._normalize_word(word)
        tags = self.word_tags.get(normalized, set())
        if any(tag.startswith(("T-", "Q-", "S-", "AS-", "IS-")) for tag in tags):
            return True

        suffix_generator = getattr(self, "suffix_generator", None)
        verb_index = getattr(suffix_generator, "verb_index", None)
        if verb_index is not None and verb_index.word_records(normalized):
            return True
        return bool(
            suffix_generator is not None
            and suffix_generator.exact_suffix_matches(normalized)
        )

    def _is_adjective_tagged_word(self, word: str) -> bool:
        normalized = self._normalize_word(word)
        return any(
            "ADJ" in tag.split("-", 1)[0]
            for tag in self.word_tags.get(normalized, set())
        )

    def _verb_records_for_surface(self, word: str):
        normalized = self._normalize_word(word)
        suffix_generator = getattr(self, "suffix_generator", None)
        verb_index = getattr(suffix_generator, "verb_index", None)
        if verb_index is None:
            return []

        records = verb_index.word_records(normalized)
        if records:
            return records

        generated = suffix_generator.exact_suffix_matches(normalized)
        return [
            record
            for candidate in generated
            for record in verb_index.word_records(candidate.base)
        ]

    def _is_exclusively_imperative(self, word: str) -> bool:
        records = self._verb_records_for_surface(word)
        return bool(records) and all(record.tense == "IMP" for record in records)

    def _negative_imperative_form(self, word: str) -> str | None:
        normalized = self._normalize_word(word)
        records = []
        if normalized.endswith("x"):
            stripped_records = self._verb_records_for_surface(normalized[:-1])
            if stripped_records and all(
                record.tense == "IMP" and record.person in {"2S", "2P"}
                for record in stripped_records
            ):
                records = stripped_records
        if not records:
            records = self._verb_records_for_surface(normalized)
        if not records or not all(
            record.tense == "IMP" and record.person in {"2S", "2P"}
            for record in records
        ):
            return None

        suffix_generator = getattr(self, "suffix_generator", None)
        verb_index = getattr(suffix_generator, "verb_index", None)
        persons = {record.person for record in records}
        if len(persons) != 1:
            return None
        target_person = next(iter(persons))

        matches: set[str] = set()
        for record in records:
            for related in verb_index.by_short_tag.get(record.short_tag, ()):
                if (
                    related.tense == "MPERF"
                    and related.person == target_person
                    and related.word.endswith("x")
                ):
                    matches.add(related.word)

        return next(iter(matches)) if len(matches) == 1 else None

    def _is_probable_noun(self, word: str) -> bool:
        normalized = self._normalize_word(word)
        if not normalized:
            return False

        tags = self.word_tags.get(normalized)
        if not tags:
            return True

        return not any(
            tag.startswith(("T-", "Q-", "S-", "AS-", "IS-")) for tag in tags
        )

    @lru_cache(maxsize=32768)
    def _is_noun_tagged_word(self, word: str) -> bool:
        normalized = self._normalize_word(word)
        tags = self.word_tags.get(normalized, set())
        return any("NOUN" in tag.split("-", 1)[0] for tag in tags)

    @lru_cache(maxsize=32768)
    def _is_dual_noun(self, word: str) -> bool:
        normalized = self._normalize_word(word)
        return any(
            tag.split("-", 1)[0] == "DUALNOUN"
            for tag in self.word_tags.get(normalized, set())
        )

    @lru_cache(maxsize=32768)
    def _is_pronoun_tagged_word(self, word: str) -> bool:
        normalized = self._normalize_word(word)
        tags = self.word_tags.get(normalized, set())
        return any(tag.startswith("PRON") for tag in tags)

    @lru_cache(maxsize=32768)
    def _is_adverb_tagged_word(self, word: str) -> bool:
        normalized = self._normalize_word(word)
        tags = self.word_tags.get(normalized, set())
        return any(
            tag.startswith(("ADVERB", "ADV-", "SHORTADVERB", "LADVERB")) for tag in tags
        )

    @lru_cache(maxsize=32768)
    def _is_preposition_tagged_word(self, word: str) -> bool:
        normalized = self._normalize_word(word)
        tags = self.word_tags.get(normalized, set())
        return any(
            tag.startswith(("PREP", "SHORTPREP", "DEFPREP", "ISHORTDEFPREP"))
            for tag in tags
        )

    def _supports_l_apostrophe_tail(self, word: str) -> bool:
        normalized = self._normalize_word(word)
        if (
            self._is_adjective_tagged_word(normalized)
            or self._is_verb_tagged_word(normalized)
            or self._is_adverb_tagged_word(normalized)
        ):
            return True
        return self._is_preposition_tagged_word(normalized) and self._starts_vowel_gh_or_h(
            normalized
        )

    def _is_verb_or_pronoun_tagged(self, word: str) -> bool:
        """True when *word* is tagged as a verb or pronoun."""
        normalized = self._normalize_word(word)
        tags = self.word_tags.get(normalized, set())
        return self._is_verb_tagged_word(normalized) or any(
            tag.startswith("PRON") for tag in tags
        )

    def _xi_form_for_word(self, next_word: str) -> str:
        """
        Determine whether the Maltese indefinite particle should
        surface as ``xi`` or ``x'`` before *next_word*, based on
        phonological rules:

        * CC… (consonant cluster)   → xi   (xi sport)
        * CV… (consonant + vowel)   → x'   (x'karozza)
        * V…  (starts with vowel)   → x'   (x'arja)
        * għ… (starts with għ)      → x'   (x'għandek)
        """
        normalized = self._normalize_word(next_word)
        if not normalized:
            return "xi"

        tokens = self._letter_tokens(normalized)
        if not tokens:
            return "xi"

        # starts with vowel or għ  → x'
        if tokens[0] in self.VOWELS or normalized.startswith(("għ", "gh")):
            return "x'"

        # single consonant + vowel (CV…) → x'
        if (
            len(tokens) >= 2
            and tokens[0] not in self.VOWELS
            and tokens[1] in self.VOWELS
        ):
            return "x'"

        # consonant cluster (CC…) → xi
        return "xi"

    @lru_cache(maxsize=32768)
    def _is_feminine_noun(self, word: str) -> bool:
        normalized = self._normalize_word(word)
        tags = self.word_tags.get(normalized, set())
        if any(tag == "SINGNOUNF" for tag in tags):
            return True

        # Fallback for manual words lacking dictionary tags: assume feminine if ending in 'a'
        if self.NOUN_POSSESSIVE_MODE.lower().strip() == "manual":
            if normalized in self._manual_noun_bases:
                return normalized.endswith("a")
        return False

    @lru_cache(maxsize=32768)
    def _noun_possessive_base_is_enabled(self, word: str) -> bool:
        normalized = self._normalize_word(word)
        mode = self.NOUN_POSSESSIVE_MODE.lower().strip()

        if mode == "off":
            return False

        if mode == "automatic":
            return self._is_noun_tagged_word(normalized)

        return normalized in self._manual_noun_bases

    def _word_ends_with_vowel(self, word: str) -> bool:
        graphemes = self._graphemes(self._normalize_word(word))
        for token in reversed(graphemes):
            if token and any(ch.isalpha() for ch in token):
                return token in self.VOWELS
        return False

    def _noun_uses_plural_it_stem(self, noun: str) -> bool:
        normalized = self._normalize_word(noun)
        if not (
            self._is_feminine_noun(normalized)
            and normalized.endswith("a")
            and not normalized.endswith("jja")
        ):
            return False

        tokens = self._letter_tokens(normalized)
        if len(tokens) < 3 or tokens[-1] != "a":
            return False

        return (
            tokens[-2] not in self.VOWELS
            and tokens[-3] not in self.VOWELS
            and tokens[-2] == tokens[-3]
            and tokens[-2] != "j"
        )

    def _noun_possessive_stems(self, noun: str) -> list[tuple[str, bool]]:
        normalized = self._normalize_word(noun)
        if not self._noun_possessive_base_is_enabled(normalized):
            return []

        stems: list[tuple[str, bool]] = []

        def add(stem: str, vowel_like: bool) -> None:
            if stem and (stem, vowel_like) not in stems:
                stems.append((stem, vowel_like))

        if normalized.endswith("'"):
            add(normalized[:-1] + "j", True)
        elif self._is_feminine_noun(normalized) and normalized.endswith("a"):
            stem = normalized[:-1] + "t"
            stem = re.sub(r"jj(?=t$)", "j", stem)
            add(stem, False)
        else:
            add(normalized, self._word_ends_with_vowel(normalized))

        return stems

    @lru_cache(maxsize=8192)
    def _noun_possessive_surfaces_for_base(self, noun: str) -> frozenset[str]:
        surfaces: set[str] = set()
        normalized = self._normalize_word(noun)

        if self._is_dual_noun(normalized) and normalized.endswith("n"):
            stem = normalized[:-1]
            surfaces.update(
                {
                    stem + "ja",
                    stem + "k",
                    stem + "h",
                    stem + "ha",
                    stem + "na",
                    stem + "kom",
                    stem + "hom",
                }
            )
            return frozenset(surfaces)

        for stem, vowel_like in self._noun_possessive_stems(normalized):
            plural_stem = stem
            if (
                self._is_feminine_noun(normalized)
                and normalized.endswith("a")
                and self._noun_uses_plural_it_stem(normalized)
            ):
                plural_stem = normalized[:-1] + "it"

            surfaces.add(stem + "i")
            surfaces.add(stem + ("k" if vowel_like else "ek"))
            if not vowel_like:
                surfaces.add(stem + "ok")
            surfaces.add(stem + ("h" if vowel_like else "u"))
            # 3SF 'ha' also uses the it-stem for CCa nouns (e.g. saħħitha)
            if plural_stem != stem:
                surfaces.add(plural_stem + "ha")
            else:
                surfaces.add(stem + "ha")
            surfaces.add(plural_stem + "na")
            surfaces.add(plural_stem + "kom")
            surfaces.add(plural_stem + "hom")
        return frozenset(surfaces)

    @lru_cache(maxsize=32768)
    def _noun_possessive_base_for_surface(self, word: str) -> str | None:
        normalized = self._normalize_word(word)
        suffixes = ("kom", "hom", "ek", "ok", "ha", "na", "i", "u", "h", "k")

        if normalized.endswith("ja"):
            dual_base = normalized[:-2] + "n"
            if (
                dual_base in self.dictionary_set
                and self._is_dual_noun(dual_base)
                and normalized in self._noun_possessive_surfaces_for_base(dual_base)
            ):
                return dual_base

        for suffix in suffixes:
            if not normalized.endswith(suffix) or len(normalized) <= len(suffix):
                continue

            stem = normalized[: -len(suffix)]
            possible_bases = []

            if stem.endswith("j"):
                possible_bases.append(stem[:-1] + "'")

            if stem.endswith("jt"):
                possible_bases.append(stem[:-2] + "jja")

            if stem.endswith("it"):
                possible_bases.append(stem[:-2] + "a")

            if stem.endswith("t"):
                possible_bases.append(stem[:-1] + "a")

            possible_bases.append(stem + "n")
            possible_bases.append(stem)

            for base in possible_bases:
                if (
                    base in self.dictionary_set
                    and self._noun_possessive_base_is_enabled(base)
                    and normalized in self._noun_possessive_surfaces_for_base(base)
                ):
                    return base

        return None

    def _correct_noun_possessive_suffix(self, word: str) -> str | None:
        normalized = self._normalize_word(word)
        if self._noun_possessive_base_for_surface(normalized):
            return normalized

        variants = list(self._strict_lookup_variants(normalized))
        orthographic_generator = getattr(self, "orthographic_generator", None)
        if orthographic_generator is not None and hasattr(
            orthographic_generator, "shortcut_letter_variants"
        ):
            variants.extend(
                orthographic_generator.shortcut_letter_variants(
                    normalized,
                    max_changes=3,
                    max_variants=64,
                )
            )

        if "jjt" in normalized:
            variants.append(normalized.replace("jjt", "jt"))
        if normalized.endswith("a"):
            variants.append(normalized[:-1] + "ha")
        for plural_suffix in ("na", "kom", "hom"):
            if normalized.endswith("t" + plural_suffix):
                stem = normalized[: -len(plural_suffix)]
                if stem.endswith("t"):
                    variants.append(stem[:-1] + "it" + plural_suffix)

        seen: set[str] = set()
        for variant in variants:
            variant = self._normalize_word(variant)
            if not variant or variant in seen:
                continue
            seen.add(variant)

            if self._noun_possessive_base_for_surface(variant):
                return variant

            if "jjt" in variant:
                reduced = variant.replace("jjt", "jt")
                if self._noun_possessive_base_for_surface(reduced):
                    return reduced

        return None

    # ------------------------------------------------------------------
    # Dictionary loading
    # ------------------------------------------------------------------

    def _load_dictionary_files(
        self, file_paths: list[Path]
    ) -> list[tuple[str, str | None]]:
        entries: list[tuple[str, str | None]] = []

        for file_path in file_paths:
            try:
                with open(file_path, encoding="utf-8") as fp:
                    lines = fp.read().splitlines()
            except FileNotFoundError:
                print(f"Warning: dictionary file not found: {file_path}")
                continue

            if lines and lines[0].strip().isdigit():
                lines = lines[1:]

            for line in lines:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                # Keep only the first whitespace field:
                #   word/FLAGS po:noun -> word/FLAGS
                first_field = line.split()[0]

                if "/" in first_field:
                    word, raw_tag = first_field.split("/", 1)
                    word = word.strip()
                    raw_tag = raw_tag.strip()
                    if self._is_paradigm_tag(raw_tag):
                        entries.append((word, self._parse_paradigm_key(raw_tag)))
                    elif "-" in raw_tag:
                        entries.append((word, raw_tag))
                    else:
                        entries.append((word, None))
                else:
                    entries.append((first_field, None))

        return entries

    def _load_eu_single_word_entries(
        self, file_path: Path
    ) -> list[tuple[str, str | None]]:
        entries: list[tuple[str, str | None]] = []
        try:
            lines = file_path.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            return entries

        for line in lines:
            line = line.strip()
            if not line or line.startswith("#") or "/" not in line:
                continue
            word, raw_tag = line.split("/", 1)
            word = word.strip()
            raw_tag = raw_tag.strip()
            if (
                not word
                or any(character.isspace() for character in word)
                or not raw_tag.startswith("MLT-")
            ):
                continue
            entries.append((word, raw_tag))
        return entries

    def _load_places_dictionary(self, file_path: Path) -> None:
        try:
            with open(file_path, encoding="utf-8") as fp:
                lines = fp.read().splitlines()
        except FileNotFoundError:
            return

        entries: list[str] = list(self.country_maltese_names)
        in_maltese_section = False
        for line in lines:
            line = line.strip()
            if line == "# MALTESE GLOBAL PLACE NAMES":
                in_maltese_section = True
                continue
            if not line or line.startswith("#") or not in_maltese_section:
                continue
            entry = unicodedata.normalize("NFC", line.split("/", 1)[0].strip())
            if entry:
                entries.append(entry)

        entries = list(dict.fromkeys(entries))
        self.place_entries = entries
        for entry in entries:
            normalized = self._normalize_word(entry)
            if not normalized:
                continue
            if " " in normalized:
                self.place_phrases.append(normalized)
                self.place_phrase_display[normalized] = entry
                tokens = self._letter_tokens_raw(normalized)
                anchor = self._extract_consonant_anchor_from_tokens(tokens)
                if anchor:
                    if anchor not in self.place_phrase_anchor_map:
                        self.place_phrase_anchor_buckets[
                            (anchor[0], len(anchor))
                        ].append(anchor)
                    self.place_phrase_anchor_map[anchor].append(normalized)
            else:
                self.place_words.append(normalized)
                self.place_word_set.add(normalized)
                self.place_word_display[normalized] = entry
                tokens = self._letter_tokens_raw(normalized)
                if tokens:
                    self.place_word_buckets[(tokens[0], len(tokens))].append(normalized)
                    anchor = self._extract_consonant_anchor_from_tokens(tokens)
                    if anchor:
                        if anchor not in self.place_word_anchor_map:
                            self.place_word_anchor_buckets[
                                (anchor[0], len(anchor))
                            ].append(anchor)
                        self.place_word_anchor_map[anchor].append(normalized)

    def _load_country_place_index(self, file_path: Path) -> None:
        try:
            payload = json.loads(file_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return

        for record in payload.get("records", ()):
            english = str(record.get("english", "")).strip()
            maltese = str(record.get("maltese", "")).strip()
            official = str(record.get("maltese_official", "")).strip()
            if not english or not maltese:
                continue
            english_key = self._normalize_word(english)
            self.country_english_to_maltese[english_key] = maltese
            self.country_english_display[english_key] = english
            for name in (maltese, official):
                name_key = self._normalize_word(name)
                if name_key:
                    self.country_maltese_to_english[name_key] = english
                    self.country_maltese_names.append(name)
            for demonym in record.get("demonyms", ()):
                demonym = str(demonym).strip()
                if demonym:
                    self.country_maltese_names.append(demonym)

    def _add_country_translation_choices(self, tokens: list[dict]) -> None:
        for token in tokens:
            if token.get("type") == "text":
                continue
            corrected = str(token.get("corrected", "")).strip()
            maltese = self.country_english_to_maltese.get(
                self._normalize_word(corrected)
            )
            if not maltese or self._normalize_word(maltese) == self._normalize_word(
                corrected
            ):
                continue
            token["place_translation"] = True
            token["choices"] = [
                {
                    "word": maltese,
                    "meaning": corrected,
                    "suggestion_kind": "place_translation",
                }
            ]

    def _contract_negative_ma(self, phrase: str) -> str:
        normalized = self._normalize_word(phrase)
        if not normalized.startswith("ma "):
            return phrase
        tail = normalized[3:].strip()
        if (
            not tail
            or tail[0] not in self.VOWELS
            or not self._is_verb_tagged_word(tail)
        ):
            return phrase
        contracted = f"m'{tail}"
        return self._match_capitalisation(phrase, contracted)

    def _is_paradigm_tag(self, tag: str) -> bool:
        return bool(self.PARADIGM_TAG_PATTERN.match(tag))

    def _parse_paradigm_key(self, tag: str) -> str:
        # T-bgħt-F1-IMP-2S -> T-bgħt-F1
        parts = tag.split("-")
        return "-".join(parts[:3]) if len(parts) >= 3 else tag

    # ------------------------------------------------------------------
    # Anchors/vowels
    # ------------------------------------------------------------------

    def _extract_consonant_anchor_from_tokens(self, tokens: list[str]) -> str:
        """Consonant skeleton with doubled consonants collapsed."""
        consonants = [token for token in tokens if token not in self.VOWELS]
        collapsed: list[str] = []
        for token in consonants:
            if not collapsed or collapsed[-1] != token:
                collapsed.append(token)
        return "".join(collapsed)

    def _extract_consonant_anchor(self, word: str) -> str:
        normalized = self._normalize_word(word)
        if normalized in self.word_anchors:
            return self.word_anchors[normalized]
        return self._extract_consonant_anchor_from_tokens(
            self._letter_tokens_raw(normalized)
        )

    def _vowel_slots(self, word: str) -> list[tuple[int, str]]:
        normalized = self._normalize_word(word)
        if normalized in self.word_vowel_slots:
            return self.word_vowel_slots[normalized]
        tokens = self._letter_tokens_raw(normalized)
        return [(i, t) for i, t in enumerate(tokens) if t in self.VOWELS]

    def _count_vowels(self, word: str) -> int:
        normalized = self._normalize_word(word)
        if normalized in self.word_vowel_counts:
            return self.word_vowel_counts[normalized]
        return sum(1 for t in self._letter_tokens_raw(normalized) if t in self.VOWELS)

    def _vowel_sequence(self, word: str) -> str:
        return "".join(v for _, v in self._vowel_slots(word))

    # ------------------------------------------------------------------
    # Distance/scoring
    # ------------------------------------------------------------------

    def _damerau_levenshtein_distance(self, a: list, b: list) -> int:
        """Optimal-string-alignment Damerau-Levenshtein distance."""
        n, m = len(a), len(b)
        if n == 0:
            return m
        if m == 0:
            return n

        dp = [[0] * (m + 1) for _ in range(n + 1)]
        for i in range(n + 1):
            dp[i][0] = i
        for j in range(m + 1):
            dp[0][j] = j

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = 0 if a[i - 1] == b[j - 1] else 1
                dp[i][j] = min(
                    dp[i - 1][j] + 1,
                    dp[i][j - 1] + 1,
                    dp[i - 1][j - 1] + cost,
                )
                if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                    dp[i][j] = min(dp[i][j], dp[i - 2][j - 2] + 1)

        return dp[n][m]

    def _word_distance(self, word1: str, word2: str) -> int:
        return self._damerau_levenshtein_distance(
            self._letter_tokens(word1),
            self._letter_tokens(word2),
        )

    def _vowel_slot_vector_score(
        self,
        typo_slots: list[tuple[int, str]],
        candidate_slots: list[tuple[int, str]],
    ) -> float:
        """0..1 score for matching vowel identity and approximate slot position."""
        if not typo_slots and not candidate_slots:
            return 1.0
        if not typo_slots or not candidate_slots:
            return 0.0

        max_pos = max(typo_slots[-1][0], candidate_slots[-1][0], 1)
        used: set[int] = set()
        matched = 0.0

        for t_pos, t_vowel in typo_slots:
            best = 0.0
            best_idx = -1
            for idx, (c_pos, c_vowel) in enumerate(candidate_slots):
                if idx in used or c_vowel != t_vowel:
                    continue
                pos_score = max(0.0, 1.0 - abs(t_pos - c_pos) / max_pos)
                if pos_score > best:
                    best = pos_score
                    best_idx = idx
            if best_idx >= 0:
                used.add(best_idx)
                matched += best

        count_ratio = min(len(typo_slots), len(candidate_slots)) / max(
            len(typo_slots), len(candidate_slots)
        )
        return max(0.0, (matched / len(typo_slots)) * count_ratio)

    def _score_once(self, typo_form: str, candidate: str, stage: str) -> ScoreRow:
        typo_tokens = self._letter_tokens(typo_form)
        candidate_tokens = self._letter_tokens(candidate)
        max_len = max(1, max(len(typo_tokens), len(candidate_tokens)))

        edit_distance = self._word_distance(typo_form, candidate)
        edit_score = edit_distance / max_len

        typo_anchor = self._extract_consonant_anchor(typo_form)
        candidate_anchor = self._extract_consonant_anchor(candidate)
        consonant_dist = self._damerau_levenshtein_distance(
            list(typo_anchor), list(candidate_anchor)
        )
        consonant_score = consonant_dist / max(
            1, max(len(typo_anchor), len(candidate_anchor))
        )

        vowel_slot_score = self._vowel_slot_vector_score(
            self._vowel_slots(typo_form), self._vowel_slots(candidate)
        )

        typo_vowels = self._count_vowels(typo_form)
        candidate_vowels = self._count_vowels(candidate)
        vowel_count_score = abs(typo_vowels - candidate_vowels) / max(
            1, max(typo_vowels, candidate_vowels)
        )

        length_score = abs(len(typo_tokens) - len(candidate_tokens)) / max_len

        final_score = (
            (1.0 - vowel_slot_score) * 0.40
            + edit_score * 0.25
            + consonant_score * 0.20
            + vowel_count_score * 0.10
            + length_score * 0.05
        )

        return ScoreRow(
            candidate=candidate,
            score=final_score,
            edit_distance=edit_distance,
            consonant_score=consonant_score,
            vowel_slot_score=vowel_slot_score,
            vowel_count_score=vowel_count_score,
            length_score=length_score,
            stage=stage,
            matched_typo_form=typo_form,
        )

    def _candidate_score(self, typo: str, candidate: str, stage: str) -> ScoreRow:
        """
        Score candidate against the best safe orthographic form of the typo.
        This lets ibaght be compared through ibgħat without rewriting the text first.
        """
        typo_forms = self._strict_lookup_variants(typo)
        rows = [self._score_once(form, candidate, stage) for form in typo_forms]
        return min(rows, key=lambda row: row.score)

    def _max_distance(self, word: str) -> int:
        length = len(self._letter_tokens(word))
        if length <= 4:
            return 1
        if length <= 8:
            return 2
        return 3

    def _is_acceptable_match(
        self,
        row: ScoreRow,
        max_distance: int,
        score_limit: float = 0.55,
    ) -> bool:
        if self._violates_ghi_sequence_rule(row.matched_typo_form, row.candidate):
            return False
        if row.score > score_limit:
            return False
        if row.edit_distance <= max_distance:
            return True
        return (
            row.edit_distance <= max_distance + 1
            and row.vowel_slot_score >= 0.70
            and row.score <= min(score_limit, 0.48)
        )

    # ------------------------------------------------------------------
    # Variant generation
    # ------------------------------------------------------------------

    def _strict_lookup_variants(self, word: str) -> list[str]:
        """
        Ordered safe high-priority variants for exact lookup and scoring.

        The real generation lives in helpers/orthographic_generator.py.
        This fallback exists only so the class still works if the helper
        has not yet been attached during startup.
        """
        if hasattr(self, "orthographic_generator"):
            return self.orthographic_generator.strict_lookup_variants(word)

        normalized = self._normalize_word(word)
        variants: list[str] = []

        def add(candidate: str) -> None:
            candidate = self._normalize_word(candidate)
            if candidate and candidate not in variants:
                variants.append(candidate)

        add(normalized)
        add(normalized.replace("gh", "għ"))

        return variants

    def _remove_token(self, word: str, token: str) -> list[str]:
        """
        Ordered removal variants.

        The real generation lives in helpers/orthographic_generator.py.
        """
        if hasattr(self, "orthographic_generator"):
            return self.orthographic_generator.remove_token(word, token)

        g = self._graphemes(word)
        variants: list[str] = []
        for i, ch in enumerate(g):
            if ch == token:
                candidate = self._from_graphemes(g[:i] + g[i + 1 :])
                if candidate not in variants:
                    variants.append(candidate)
        return variants

    def _suffix_repair_variants(self, word: str) -> set[str]:
        normalized = self._normalize_word(word)
        variants: set[str] = set()
        for suffix, replacement in self._sorted_suffix_repairs:
            if normalized.endswith(suffix):
                variants.add(normalized[: -len(suffix)] + replacement)
        return variants

    def _validated_liex_to_lix_repair(self, word: str) -> str | None:
        normalized = self._normalize_word(word)

        if not normalized.endswith("liex"):
            return None

        candidate = normalized[:-4] + "lix"
        suffix_generator = getattr(self, "suffix_generator", None)

        if suffix_generator is not None and suffix_generator.exact_suffix_matches(
            candidate
        ):
            return candidate

        return None

    def _manual_repair_variants(self, word: str) -> list[str]:
        normalized = self._normalize_word(word)
        variants: list[str] = []

        def add(candidate: str) -> None:
            candidate = self._normalize_word(candidate)
            if candidate and candidate != normalized and candidate not in variants:
                variants.append(candidate)

        if normalized in self.MANUAL_WORD_REPAIRS:
            add(self.MANUAL_WORD_REPAIRS[normalized])

        for candidate in self.MANUAL_WORD_SUGGESTIONS.get(normalized, ()):
            add(candidate)

        if normalized.endswith("x") and normalized[:-1] in self.MANUAL_WORD_REPAIRS:
            add(self.MANUAL_WORD_REPAIRS[normalized[:-1]] + "x")

        for typo_core in ("ejd", "ajd"):
            index = normalized.rfind(typo_core)
            if index < 0:
                continue

            tail = normalized[index + len(typo_core) :]
            prefix = normalized[:index]
            if tail in self.MANUAL_EJD_AJD_TAILS:
                add(prefix + "għid" + tail)
            elif tail == "ila":
                add(prefix + "għidilha")
            elif tail.startswith(("il", "l")):
                add(prefix + "għid" + tail)

        for sequence, replacement in self.MANUAL_SEQUENCE_REPAIRS:
            if sequence in normalized:
                add(normalized.replace(sequence, replacement, 1))

        for suffix, replacement in self.MANUAL_ENDING_REPAIRS:
            if normalized.endswith(suffix):
                add(normalized[: -len(suffix)] + replacement)

        for suffix, replacement in self.MANUAL_VERB_GH_ENDING_REPAIRS:
            if not normalized.endswith(suffix):
                continue

            candidate = normalized[: -len(suffix)] + replacement
            if self._supports_verb_final_gh_repair(candidate):
                add(candidate)

        return variants

    def _lexicalized_key(self, word: str) -> str:
        normalized = self._normalize_word(word)
        folded = (
            normalized.replace("għ", "gh")
            .replace("ħ", "h")
            .replace("ċ", "c")
            .replace("ġ", "g")
            .replace("ż", "z")
        )
        return re.sub(r"[-'\s]+", "", folded)

    def _lexicalized_form_variants(self, word: str) -> list[str]:
        normalized = self._normalize_word(word)
        if not normalized:
            return []

        word_keys = {self._lexicalized_key(normalized)}
        for lookup in self._strict_lookup_variants(normalized):
            word_keys.add(self._lexicalized_key(lookup))

        ranked: list[tuple[int, int, int, str, tuple[str, ...]]] = []
        for rule_index, (canonical, alternatives) in enumerate(
            self.LEXICALIZED_FORM_RULES
        ):
            targets = (canonical, *alternatives)
            target_keys = {self._lexicalized_key(target) for target in targets}
            best_distance = min(
                self._damerau_levenshtein_distance(list(word_key), list(target_key))
                for word_key in word_keys
                for target_key in target_keys
            )
            exact_key_match = bool(word_keys & target_keys)
            if normalized in self.dictionary_set and not exact_key_match:
                continue
            if len(self._letter_tokens(normalized)) <= 5 and not exact_key_match:
                continue
            max_distance = max(1, min(3, self._max_distance(canonical)))
            if not exact_key_match and best_distance > max_distance:
                continue

            exact_canonical = int(normalized == canonical)
            ranked.append(
                (
                    0 if exact_canonical else best_distance,
                    len(canonical),
                    rule_index,
                    canonical,
                    alternatives,
                )
            )

        if not ranked:
            return []

        ranked.sort()
        _distance, _length, _rule_index, canonical, alternatives = ranked[0]
        variants: list[str] = []

        def add(candidate: str) -> None:
            normalized_candidate = self._normalize_word(candidate)
            if normalized_candidate and normalized_candidate not in variants:
                variants.append(normalized_candidate)

        add(canonical)
        for alternative in alternatives:
            add(alternative)

        return variants

    def _valid_generated_surface(self, candidate: str) -> bool:
        normalized = self._normalize_word(candidate)
        if normalized in self.dictionary_set:
            return True
        if normalized.startswith("m'") and normalized[2:] in self.dictionary_set:
            return True
        suffix_generator = getattr(self, "suffix_generator", None)
        if suffix_generator is None:
            return False
        return bool(
            suffix_generator.exact_suffix_matches(normalized)
            or self._normalize_word(suffix_generator.correct_suffix(normalized) or "")
            == normalized
        )

    def _negative_ix_variants(self, word: str) -> list[str]:
        normalized = self._normalize_word(word)
        if not normalized.endswith("ix") or len(normalized) <= 3:
            return []

        variants: list[str] = []

        def add(candidate: str) -> None:
            candidate = self._normalize_word(candidate)
            if (
                candidate
                and candidate != normalized
                and candidate not in variants
                and self._valid_generated_surface(candidate)
            ):
                variants.append(candidate)

        stem = normalized[:-2]
        add(stem + "iex")
        add(stem + "hiex")

        if normalized.startswith("mand"):
            suffix = normalized[4:]
            add("m'għand" + suffix)
            add("m'għandhiex")

        return variants

    def _article_tail_repair(self, tail: str) -> str | None:
        normalized = self._normalize_word(tail)
        if self._valid_generated_surface(normalized):
            return normalized

        orthographic_generator = getattr(self, "orthographic_generator", None)
        if orthographic_generator is not None and hasattr(
            orthographic_generator, "dictionary_shortcut_variants"
        ):
            for candidate in orthographic_generator.dictionary_shortcut_variants(
                normalized
            ):
                if self._valid_generated_surface(candidate):
                    return candidate

        for ending in ("ijh", "ieh"):
            if normalized.endswith(ending):
                candidate = normalized[: -len(ending)] + "iħ"
                if self._valid_generated_surface(candidate):
                    return candidate

        doubled = getattr(self, "doubled_letter_generator", None)
        if doubled is not None and hasattr(doubled, "missing_double_variants"):
            for candidate in doubled.missing_double_variants(normalized):
                if self._valid_generated_surface(candidate):
                    return candidate

        return None

    def _valid_apostrophe_prefix_word(self, word: str) -> str | None:
        normalized = self._normalize_word(word)
        if "'" not in normalized:
            return None
        if normalized in self.dictionary_set:
            return normalized

        prefix, remainder = normalized.split("'", 1)
        if prefix == "l" and remainder:
            repaired_remainder = self._article_tail_repair(remainder) or remainder
            if self._supports_l_apostrophe_tail(repaired_remainder):
                return f"l'{repaired_remainder}"
            return None

        if prefix not in {"b", "f", "m", "t", "x"} or not remainder:
            return None

        repaired_remainder = self._article_tail_repair(remainder) or remainder
        if self._valid_generated_surface(repaired_remainder):
            return f"{prefix}'{repaired_remainder}"

        return None

    def _pattern_repair_variants(self, word: str) -> list[str]:
        normalized = self._normalize_word(word)
        variants: list[str] = []

        if normalized in self.dictionary_set and not (
            normalized.startswith("taj") and normalized != "tajtu"
        ):
            return variants

        def add(candidate: str) -> None:
            candidate = self._normalize_word(candidate)
            if candidate and candidate != normalized and candidate not in variants:
                variants.append(candidate)

        if normalized == "ikolna" and "ikollna" in self.dictionary_set:
            add("ikollna")

        if normalized == "kollhu" and "kollu" in self.dictionary_set:
            add("kollu")

        if normalized == "pero" and "però" in self.dictionary_set:
            add("però")

        social_comment_repair = self.SOCIAL_COMMENT_REPAIRS.get(normalized)
        if social_comment_repair:
            add(social_comment_repair)

        missing_gh_mperf = self._missing_gh_mperf_repair(normalized)
        if missing_gh_mperf:
            add(missing_gh_mperf)

        for candidate in self.NUMBER_FORM_REPAIRS.get(normalized, ()):
            if candidate in self.dictionary_set:
                add(candidate)

        if normalized.startswith("u") and len(normalized) > 2:
            w_candidate = "w" + normalized[1:]
            if w_candidate in self.dictionary_set:
                add(w_candidate)

        for marker, corrected_marker in (
            ("ha", "ħa"),
            ("ħa", "ħa"),
            ("se", "se"),
            ("sa", "sa"),
        ):
            if normalized.startswith(marker) and len(normalized) > len(marker) + 1:
                tail = normalized[len(marker) :]
                if (
                    tail in self.dictionary_set
                    and self._is_verb_tagged_word(tail)
                    and self._is_future_particle_complement(tail)
                ):
                    add(f"{corrected_marker} {tail}")
                if tail.startswith("n") and len(tail) > 2:
                    m_tail = "m" + tail[1:]
                    if (
                        m_tail in self.dictionary_set
                        and self._is_verb_tagged_word(m_tail)
                        and self._is_future_particle_complement(m_tail)
                    ):
                        add(f"{corrected_marker} {m_tail}")

        if normalized.startswith("ma") and len(normalized) > 4:
            tail = normalized[2:]
            if tail in self.dictionary_set and self._is_verb_tagged_word(tail):
                add(f"ma {tail}")

        for candidate in self._negative_ix_variants(normalized):
            add(candidate)

        for ending in ("ijh", "ieh"):
            if normalized.endswith(ending):
                candidate = normalized[: -len(ending)] + "iħ"
                if candidate in self.dictionary_set:
                    add(candidate)

        compact_prefixes = (
            ("bi", "bil"),
            ("fi", "fil"),
            ("ta", "tal"),
            ("ma", "mal"),
            ("i", "il"),
            ("", "il"),
        )
        for base_prefix, _article_prefix in compact_prefixes:
            for sun in ("d", "n", "r", "s", "t", "x", "z", "ċ", "ż"):
                typed_prefix = f"{base_prefix}{sun}"
                if not normalized.startswith(typed_prefix + sun):
                    continue
                tail = normalized[len(typed_prefix) :]
                corrected_tail_norm = self._article_tail_repair(tail)
                if corrected_tail_norm:
                    add(f"{base_prefix}{sun}-{corrected_tail_norm}")

        if normalized.startswith("bix") and "xieraq" in self.dictionary_set:
            tail = normalized[3:]
            if self._word_distance(tail, "xieraq") <= 3:
                add("bix-xieraq")

        if "qot" in normalized:
            candidate = normalized.replace("qot", "qgħod", 1)
            if candidate in self.dictionary_set:
                add(candidate)

        if normalized.startswith("taj"):
            tail = normalized[3:]
            if tail == "at":
                candidates = (f"għajj{tail}", f"tgħajj{tail}", f"tgħaj{tail}")
            else:
                candidates = (f"tgħaj{tail}", f"tgħajj{tail}", f"għajj{tail}")
            for candidate in candidates:
                if candidate in self.dictionary_set:
                    add(candidate)

        possessive_candidates = {
            "tijak": "tiegħek",
            "tijaw": "tiegħu",
        }
        candidate = possessive_candidates.get(normalized)
        if candidate:
            add(candidate)

        return variants

    def _supports_verb_final_gh_repair(self, candidate: str) -> bool:
        normalized = self._normalize_word(candidate)
        if not normalized.endswith("agħhom"):
            return False

        if normalized in self.dictionary_set:
            return True

        for lookup in self._strict_lookup_variants(normalized):
            if lookup in self.dictionary_set:
                return True

        suffix_generator = getattr(self, "suffix_generator", None)
        if suffix_generator is None:
            return False

        generated = suffix_generator.correct_suffix(normalized)
        return bool(generated and self._normalize_word(generated) == normalized)

    def _correct_d_t_then_double(self, word: str) -> str | None:
        orthographic_generator = getattr(self, "orthographic_generator", None)
        doubled_letter_generator = getattr(self, "doubled_letter_generator", None)

        if orthographic_generator is None or doubled_letter_generator is None:
            return None

        for dt_variant in orthographic_generator.substitute_d_t(word):
            variants = [dt_variant]
            variants.extend(
                doubled_letter_generator.missing_double_variants(dt_variant)
            )

            exact = self._try_exact_variants(word, variants)
            if exact:
                return exact

        return None

    # ------------------------------------------------------------------
    # Candidate retrieval
    # ------------------------------------------------------------------

    def _lookup_anchors(self, word: str) -> set[str]:
        variants = self._strict_lookup_variants(word)
        return {self._extract_consonant_anchor(v) for v in variants if v}

    def _get_paradigm_candidates_for_anchor(self, anchor: str) -> list[str]:
        forms = self.anchor_map.get(anchor, [])
        keys: set[str] = set()
        for form in forms:
            keys.update(self.word_tags.get(form, set()))

        candidates: list[str] = []
        for key in keys:
            candidates.extend(self.paradigm_forms.get(key, []))

        return self._deduplicate(candidates)

    def _near_anchor_candidates(
        self, anchors: set[str], max_anchor_distance: int = 1
    ) -> list[str]:
        candidates: list[str] = []

        for known_anchor, words in self.anchor_map.items():
            for anchor in anchors:
                if abs(len(known_anchor) - len(anchor)) > max_anchor_distance:
                    continue
                dist = self._damerau_levenshtein_distance(
                    list(anchor), list(known_anchor)
                )
                if dist <= max_anchor_distance:
                    candidates.extend(words)
                    break

        return candidates

    @lru_cache(maxsize=32768)
    def _get_candidates_cached(self, normalized: str) -> tuple[str, ...]:
        anchors = self._lookup_anchors(normalized)
        candidates: list[str] = []
        seen: set[str] = set()

        def add(candidate: str) -> None:
            if candidate not in seen:
                candidates.append(candidate)
                seen.add(candidate)

        def extend(items: Iterable[str]) -> None:
            for item in items:
                add(item)

        # 1. Same paradigm as exact-anchor words.
        for anchor in anchors:
            extend(self._get_paradigm_candidates_for_anchor(anchor))

        # 2. Exact-anchor surface forms, including untagged words.
        for anchor in anchors:
            extend(self.anchor_map.get(anchor, []))

        # 3. Near-anchor candidates for consonant mistakes.
        if len(candidates) < 8:
            extend(self._near_anchor_candidates(anchors, max_anchor_distance=1))

        # 4. Length fallback, kept narrow and capped hard to avoid
        # whole-dictionary scans dominating one bad guess.
        if len(candidates) < 8:
            word_len = len(self._letter_tokens(normalized))
            max_length_gap = max(1, self._max_distance(normalized))
            initial = normalized[:1]
            fallback_limit = 96

            for candidate in self.dictionary:
                if len(candidates) >= fallback_limit:
                    break
                if abs(self.word_lengths[candidate] - word_len) > max_length_gap:
                    continue
                if initial and candidate[:1] != initial:
                    continue
                add(candidate)

        return tuple(candidates)

    def _get_candidates(self, word: str) -> list[str]:
        normalized = self._normalize_word(word)
        return list(self._get_candidates_cached(normalized))

    @staticmethod
    def _deduplicate(items: Iterable[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for item in items:
            if item not in seen:
                out.append(item)
                seen.add(item)
        return out

    def _is_implausible_vowel_swap(self, typo: str, candidate: str) -> bool:
        typo_norm = self._normalize_word(typo)
        candidate_norm = self._normalize_word(candidate)
        if typo_norm == candidate_norm:
            return False
        if self._violates_ghi_sequence_rule(typo_norm, candidate_norm):
            return True
        if self._extract_consonant_anchor(typo_norm) != self._extract_consonant_anchor(
            candidate_norm
        ):
            return False
        typo_vowels = self._vowel_sequence(self._strip_maltese_shortcuts(typo_norm))
        candidate_vowels = self._vowel_sequence(
            self._strip_maltese_shortcuts(candidate_norm)
        )
        return typo_vowels.replace("ie", "i") != candidate_vowels.replace("ie", "i")

    def _has_explicit_ghi_sequence(self, word: str) -> bool:
        normalized = self._normalize_word(word)
        return "għi" in normalized or "ghi" in normalized

    def _violates_ghi_sequence_rule(self, original: str, candidate: str) -> bool:
        return self._has_explicit_ghi_sequence(
            original
        ) != self._has_explicit_ghi_sequence(candidate)

    def _strip_maltese_shortcuts(self, word: str) -> str:
        normalized = self._normalize_word(word)
        return (
            normalized.replace("għ", "gh")
            .replace("ħ", "h")
            .replace("ċ", "c")
            .replace("ġ", "g")
            .replace("ż", "z")
            .replace("à", "a")
            .replace("è", "e")
            .replace("ì", "i")
            .replace("ò", "o")
            .replace("ù", "u")
        )

    def _collapse_doubles(self, word: str) -> str:
        out: list[str] = []
        for token in self._graphemes(word):
            if out and out[-1] == token and token not in self.VOWELS:
                continue
            out.append(token)
        return self._from_graphemes(out)

    def _is_plausible_whole_word_suggestion(
        self,
        original: str,
        candidate: str,
        *,
        corrected: str | None = None,
    ) -> bool:
        original_norm = self._normalize_word(original)
        candidate_norm = self._normalize_word(candidate)
        corrected_norm = self._normalize_word(corrected or "")

        if not original_norm or not candidate_norm:
            return False
        if candidate_norm == original_norm or candidate_norm == corrected_norm:
            return True
        if self._violates_ghi_sequence_rule(original_norm, candidate_norm):
            return False

        # Genuine article/preposition alternatives are handled by dedicated
        # rules and should survive this whole-word guard. A final apostrophe
        # alone is not enough: jinstema' is not a spelling alternative of
        # tinstema'.
        if any(mark in candidate_norm for mark in ("'", "-")):
            compact_original = re.sub(r"[-'\s]+", "", original_norm)
            compact_candidate = re.sub(r"[-'\s]+", "", candidate_norm)
            if compact_original == compact_candidate and not candidate_norm.endswith("x"):
                return True
            if candidate_norm.startswith(("b'", "f'", "x'", "m'", "s'", "t'", "l-", "'l")):
                return True

        # Keyboard shortcuts and gh -> għ.
        if self._strip_maltese_shortcuts(original_norm) == self._strip_maltese_shortcuts(
            candidate_norm
        ):
            return True

        # i/ie confusion only, not arbitrary i/e/a/o swaps.
        if original_norm.replace("ie", "i") == candidate_norm.replace("ie", "i"):
            return True

        def is_single_confusion(orig_str: str, cand_str: str) -> bool:
            orig_g = self._graphemes(orig_str)
            cand_g = self._graphemes(cand_str)
            if len(orig_g) != len(cand_g) or not orig_g:
                return False
            diffs = [(l, r) for l, r in zip(orig_g, cand_g) if l != r]
            if len(diffs) == 1:
                left, right = diffs[0]
                if {left, right} in (
                    {"t", "d"},
                    {"b", "p"},
                    {"k", "g"},
                    {"s", "ż"},
                    {"c", "ċ"},
                    {"ġ", "ċ"},
                ):
                    return True
                if len(orig_g) == 1 and {left, right} in ({"g", "ġ"}, {"z", "ż"}):
                    return True
                if len(orig_g) == 1 and {left, right} == {"h", "ħ"}:
                    return True
                if len(orig_g) == 1 and orig_g[-1] == left and {left, right} <= {"għ", "h", "ħ"}:
                    return True
            return False

        if is_single_confusion(original_norm, candidate_norm):
            return True

        # Addition/loss of multiple silent letters (h, għ, q), optionally combined
        # with a single-letter phonetic confusion (e.g. ritta -> ridtha, ada -> għadha)
        # or doubled consonant errors.
        def strip_silent(w: str) -> str:
            for t in ("għ", "h", "q"):
                w = w.replace(t, "")
            return self._normalize_word(w)

        orig_silent_stripped = strip_silent(original_norm)
        cand_silent_stripped = strip_silent(candidate_norm)

        if orig_silent_stripped and cand_silent_stripped:
            if orig_silent_stripped == cand_silent_stripped:
                return True
            if is_single_confusion(orig_silent_stripped, cand_silent_stripped):
                return True
            if self._collapse_doubles(orig_silent_stripped) == self._collapse_doubles(cand_silent_stripped):
                return True

        # Doubled/single consonants on the original string (just in case)
        if self._collapse_doubles(original_norm) == self._collapse_doubles(candidate_norm):
            return True
        if self._collapse_doubles(
            self._strip_maltese_shortcuts(original_norm)
        ) == self._collapse_doubles(self._strip_maltese_shortcuts(candidate_norm)):
            return True

        # Missing vowel before object suffixes:
        #   narhom -> narahom / narohom, jarha -> jaraha
        if original_norm.endswith("hom"):
            stem = original_norm[:-3]
            if candidate_norm in {stem + "ahom", stem + "ohom"}:
                return True
        if original_norm.endswith("ha"):
            stem = original_norm[:-2]
            if candidate_norm == stem + "aha":
                return True

        # Missing h before the feminine/object -ha suffix:
        #   habba -> ħabbha
        candidate_ascii = self._strip_maltese_shortcuts(candidate_norm)
        original_ascii = self._strip_maltese_shortcuts(original_norm)
        if candidate_ascii.endswith("ha"):
            without_suffix_h = candidate_ascii[:-2] + "a"
            if without_suffix_h == original_ascii:
                return True

        return False

    # ------------------------------------------------------------------
    # Correction stages
    # ------------------------------------------------------------------

    def _best_ranked_candidate(
        self,
        typo: str,
        stage: str,
        candidate_filter: Callable[[str], bool] | None = None,
        score_limit: float = 0.55,
        max_distance: int | None = None,
    ) -> ScoreRow | None:
        normalized = self._normalize_word(typo)
        max_distance = (
            self._max_distance(normalized) if max_distance is None else max_distance
        )
        typo_len = len(self._letter_tokens(normalized))

        candidates_to_score = set(self._get_candidates(normalized))
        for variant in self._lexicalized_form_variants(normalized):
            candidates_to_score.update(self._get_candidates(variant))

        ortho_gen = getattr(self, "orthographic_generator", None)
        if ortho_gen is not None:
            combined_variants = []
            for t_d in ortho_gen.substitute_d_t(normalized):
                combined_variants.extend(ortho_gen.insert_token_next_to_vowels(t_d, "h"))
                combined_variants.extend(ortho_gen.insert_token_next_to_vowels(t_d, "għ"))
            for b_p in ortho_gen.substitute_b_p(normalized):
                combined_variants.extend(ortho_gen.insert_token_next_to_vowels(b_p, "h"))
                combined_variants.extend(ortho_gen.insert_token_next_to_vowels(b_p, "għ"))
            
            for combined in combined_variants:
                if self._normalize_word(combined) in self.dictionary_set:
                    candidates_to_score.add(combined)

        rows: list[ScoreRow] = []
        for candidate in candidates_to_score:
            if candidate_filter and not candidate_filter(candidate):
                continue
            if self._is_implausible_vowel_swap(normalized, candidate):
                continue
            if (
                abs(
                    self.word_lengths.get(
                        candidate, len(self._letter_tokens(candidate))
                    )
                    - typo_len
                )
                > max_distance + 2
            ):
                continue
            rows.append(self._candidate_score(normalized, candidate, stage))

        if not rows:
            return None

        rows.sort(key=lambda row: (row.score, row.edit_distance, row.candidate))
        best = rows[0]
        if self._is_acceptable_match(best, max_distance, score_limit=score_limit):
            return best
        return None

    def _try_exact_variants(
        self, original_word: str, variants: Iterable[str]
    ) -> str | None:
        for variant in variants:
            for lookup in self._strict_lookup_variants(variant):
                if lookup in self.dictionary_set:
                    return self._match_capitalisation(original_word, lookup)
        return None

    def _try_ranked_from_variants(
        self,
        original_word: str,
        variants: Iterable[str],
        stage: str,
        score_limit: float,
    ) -> str | None:
        max_distance = self._max_distance(original_word)
        rows: list[ScoreRow] = []
        for variant in variants:
            best = self._best_ranked_candidate(
                variant,
                stage=stage,
                score_limit=score_limit,
                max_distance=max_distance,
            )
            if best:
                rows.append(best)

        if not rows:
            return None

        rows.sort(key=lambda row: (row.score, row.edit_distance, row.candidate))
        return self._match_capitalisation(original_word, rows[0].candidate)

    def _dictionary_i_ie_shortcut_variants(self, word: str) -> list[str]:
        orthographic_generator = getattr(self, "orthographic_generator", None)
        if orthographic_generator is None or not hasattr(
            orthographic_generator, "substitute_i_ie"
        ):
            return []

        normalized = self._normalize_word(word)
        variants: list[str] = []

        def add(candidate: str) -> None:
            candidate = self._normalize_word(candidate)
            if candidate and candidate in self.dictionary_set and candidate not in variants:
                variants.append(candidate)

        for i_ie_variant in orthographic_generator.substitute_i_ie(normalized):
            for shortcut_variant in orthographic_generator.shortcut_letter_variants(
                i_ie_variant,
                max_changes=2,
            ):
                add(shortcut_variant)

            shortcut_match = orthographic_generator.correct_shortcut_letters(
                i_ie_variant
            )
            if shortcut_match:
                add(shortcut_match)

        return variants

    def _match_capitalisation(self, original: str, corrected: str) -> str:
        if original.isupper():
            return corrected.upper()
        if len(original) > 1 and original[0].isupper() and original[1:].islower():
            return corrected[:1].upper() + corrected[1:]
        return corrected

    def _match_hyphenated_tail_capitalisation(
        self,
        original_tail: str,
        corrected_phrase: str,
    ) -> str:
        if not self._is_initial_capitalized(original_tail) or "-" not in corrected_phrase:
            return corrected_phrase
        prefix, tail = corrected_phrase.rsplit("-", 1)
        return f"{prefix}-{self._match_capitalisation(original_tail, tail)}"

    def _is_initial_capitalized(self, word: str) -> bool:
        return bool(word) and word[0].isupper()

    def _is_sentence_initial_position(self, text: str, start: int) -> bool:
        cursor = start - 1
        while cursor >= 0 and text[cursor].isspace():
            cursor -= 1
        return cursor < 0 or text[cursor] in ".?!"

    def _limited_candidates_from_pool(
        self,
        typo: str,
        pool: Iterable[str],
        *,
        candidate_limit: int,
    ) -> list[str]:
        normalized = self._normalize_word(typo)
        typo_len = len(self._letter_tokens(normalized))
        max_distance = self._max_distance(normalized)
        initial = normalized[:1]
        candidates: list[str] = []
        seen: set[str] = set()

        for candidate in pool:
            if candidate in seen:
                continue
            if initial and candidate[:1] != initial:
                continue
            if abs(len(self._letter_tokens(candidate)) - typo_len) > max_distance + 2:
                continue
            candidates.append(candidate)
            seen.add(candidate)
            if len(candidates) >= candidate_limit:
                break

        return candidates

    def _best_ranked_candidate_from_pool(
        self,
        typo: str,
        pool: Iterable[str],
        *,
        stage: str,
        score_limit: float,
        candidate_limit: int,
    ) -> str | None:
        normalized = self._normalize_word(typo)
        rows: list[ScoreRow] = []
        max_distance = self._max_distance(normalized)

        for candidate in self._limited_candidates_from_pool(
            normalized,
            pool,
            candidate_limit=candidate_limit,
        ):
            row = self._candidate_score(normalized, candidate, stage)
            if self._is_acceptable_match(
                row,
                max_distance,
                score_limit=score_limit,
            ):
                rows.append(row)

        if not rows:
            return None

        rows.sort(key=lambda row: (row.score, row.edit_distance, row.candidate))
        return rows[0].candidate

    def _correct_place_word(self, word: str) -> str | None:
        normalized = self._normalize_word(word)
        if not normalized or not self.place_words:
            return None

        if normalized in self.place_word_set:
            return self.place_word_display.get(normalized, word)

        orthographic = getattr(self, "orthographic_generator", None)
        if orthographic is not None:
            for variant in orthographic.shortcut_letter_variants(
                normalized,
                max_changes=2,
                max_variants=32,
            ):
                variant = self._normalize_word(variant)
                if variant in self.place_word_set:
                    return self.place_word_display.get(variant, variant)

        if not PLACE_FUZZY_CORRECTION_ENABLED:
            return None

        # ------------------------------------------------------------------
        # PLACE FUZZY CORRECTION BLOCK
        # Remove this block, its two place anchor indexes in __init__, and the
        # matching index population in _load_places_dictionary to remove fuzzy
        # place correction while retaining exact place-name recognition.
        # ------------------------------------------------------------------
        tokens = self._letter_tokens(normalized)
        if not tokens:
            return None
        pool = self._place_anchor_candidates(
            normalized,
            self.place_word_anchor_map,
            self.place_word_anchor_buckets,
        )
        if not pool:
            max_distance = self._max_distance(normalized)
            for length in range(
                max(1, len(tokens) - max_distance),
                len(tokens) + max_distance + 1,
            ):
                pool.extend(self.place_word_buckets.get((tokens[0], length), ()))

        best = self._best_ranked_candidate_from_pool(
            normalized,
            pool,
            stage="place_word",
            score_limit=0.42,
            candidate_limit=max(1, len(pool)),
        )
        if (
            best
            and best[:1] == normalized[:1]
            and abs(
                len(self._letter_tokens(best)) - len(tokens)
            )
            <= self._max_distance(normalized)
            and self._word_distance(normalized, best)
            <= self._max_distance(normalized)
        ):
            return self.place_word_display.get(best, best)

        return None

    def _correct_place_phrase(self, phrase: str) -> str | None:
        normalized = self._normalize_word(phrase)
        if not normalized or not self.place_phrases:
            return None

        if normalized in self.place_phrase_display:
            return self.place_phrase_display[normalized]

        pool = self._place_anchor_candidates(
            normalized,
            self.place_phrase_anchor_map,
            self.place_phrase_anchor_buckets,
        )
        if not pool:
            return None

        best = self._best_ranked_candidate_from_pool(
            normalized,
            pool,
            stage="place_phrase",
            score_limit=0.42,
            candidate_limit=max(1, len(pool)),
        )
        if best:
            return self.place_phrase_display.get(best, best)

        return None

    def _place_anchor_candidates(
        self,
        word: str,
        anchor_map: dict[str, list[str]],
        anchor_buckets: dict[tuple[str, int], list[str]],
    ) -> list[str]:
        anchor = self._extract_consonant_anchor(word)
        if not anchor:
            return []

        exact = anchor_map.get(anchor)
        if exact:
            return list(exact)

        candidates: list[str] = []
        for length in range(max(1, len(anchor) - 1), len(anchor) + 2):
            for known_anchor in anchor_buckets.get((anchor[0], length), ()):
                if (
                    self._damerau_levenshtein_distance(
                        list(anchor),
                        list(known_anchor),
                    )
                    <= 1
                ):
                    candidates.extend(anchor_map[known_anchor])
        return self._deduplicate(candidates)

    def _match_capitalized_place_phrase(
        self,
        text: str,
        word_tokens: list[WordToken],
        matches: list[re.Match[str]],
        index: int,
    ) -> tuple[str, str, int] | None:
        if not self.place_phrases:
            return None

        current = word_tokens[index].text
        current_capitalized = self._is_initial_capitalized(current)
        current_normalized = self._normalize_word(current)

        if current_capitalized:
            max_words = min(8, len(word_tokens) - index)
            for consumed in range(max_words, 1, -1):
                original_phrase = text[
                    matches[index].start() : matches[index + consumed - 1].end()
                ]
                english_key = self._normalize_word(original_phrase)
                if english_key in self.country_english_to_maltese:
                    return (
                        original_phrase,
                        self.country_english_display.get(
                            english_key,
                            original_phrase,
                        ),
                        consumed,
                    )

        # An exact one-word country must be handled as one token. Otherwise a
        # following conjunction can be absorbed by fuzzy place-phrase lookup
        # (for example, "China u" matching an unrelated two-word place).
        if (
            current_normalized in self.country_english_to_maltese
            or current_normalized in self.country_maltese_to_english
        ):
            return None

        if index + 1 < len(word_tokens):
            next_word = word_tokens[index + 1].text
            if current_capitalized or self._is_initial_capitalized(next_word):
                original_phrase = text[
                    matches[index].start() : matches[index + 1].end()
                ]
                corrected = self._correct_place_phrase(original_phrase)
                if corrected:
                    return original_phrase, corrected, 2

        if index + 2 < len(word_tokens) and current_capitalized:
            middle = self._normalize_word(word_tokens[index + 1].text)
            if self._article_like_token(middle):
                original_phrase = text[
                    matches[index].start() : matches[index + 2].end()
                ]
                corrected = self._correct_place_phrase(original_phrase)
                if corrected:
                    return original_phrase, corrected, 3

        return None

    def _correct_sentence_initial_capitalized(self, word: str) -> str:
        normalized = self._normalize_word(word)
        if not normalized:
            return word
        if normalized in self.dictionary_set:
            return word
        if normalized in self.country_english_to_maltese:
            return word

        corrected_place = self._correct_place_word(word)
        if corrected_place and (
            self._normalize_word(corrected_place)
            in self.country_english_to_maltese
            or self._normalize_word(corrected_place)
            in self.country_maltese_to_english
        ):
            return corrected_place

        pattern_repairs = self._pattern_repair_variants(word)
        if pattern_repairs:
            return self._match_capitalisation(word, pattern_repairs[0])

        manual_repairs = self._manual_repair_variants(word)
        if manual_repairs:
            return self._match_capitalisation(word, manual_repairs[0])

        exact = self._try_exact_variants(word, self._strict_lookup_variants(normalized))
        if exact:
            return exact

        ordinary_correction = self.correct_word(word)
        if self._normalize_word(ordinary_correction) != normalized:
            return ordinary_correction

        best = self._best_ranked_candidate_from_pool(
            normalized,
            self._get_candidates(normalized),
            stage="sentence_initial_capitalized",
            score_limit=0.45,
            candidate_limit=self.SENTENCE_INITIAL_CANDIDATE_LIMIT,
        )
        if best:
            return self._match_capitalisation(word, best)

        return word

    def _article_like_token(self, word: str) -> bool:
        normalized = self._normalize_word(word)
        return normalized in {
            "il",
            "l",
            "din",
            "dan",
            "iċ",
            "id",
            "in",
            "ir",
            "is",
            "it",
            "ix",
            "iz",
            "iż",
            "ċ",
            "d",
            "n",
            "r",
            "s",
            "t",
            "x",
            "ż",
            "z",
        } or normalized.startswith(
            (
                "il-",
                "l-",
                "din-",
                "dan-",
                "iċ-",
                "id-",
                "in-",
                "ir-",
                "is-",
                "it-",
                "ix-",
                "iz-",
                "iż-",
                "ċ-",
                "d-",
                "n-",
                "r-",
                "s-",
                "t-",
                "x-",
                "ż-",
                "z-",
            )
        )

    def _word_ends_with_consonant(self, word: str | None) -> bool:
        if not word:
            return False

        graphemes = self._graphemes(self._normalize_word(word))
        for token in reversed(graphemes):
            if not token or not any(ch.isalpha() for ch in token):
                continue
            return token not in self.VOWELS
        return False

    def _word_starts_with_two_consonants(self, word: str) -> bool:
        graphemes = self._graphemes(self._normalize_word(word))
        letters = [
            token for token in graphemes if token and any(ch.isalpha() for ch in token)
        ]
        if len(letters) < 2:
            return False
        return letters[0] not in self.VOWELS and letters[1] not in self.VOWELS

    def _is_future_particle_complement(self, word: str) -> bool:
        normalized = self._normalize_word(word)
        suffix_generator = getattr(self, "suffix_generator", None)
        verb_index = getattr(suffix_generator, "verb_index", None)
        if verb_index is None:
            return False

        records = verb_index.word_records(normalized)
        if not records and suffix_generator is not None:
            records = [
                record
                for candidate in suffix_generator.exact_suffix_matches(normalized)
                for record in verb_index.word_records(candidate.base)
            ]
        return any(record.tense == "MPERF" for record in records)

    def _has_empathetic_i_shape(self, word: str) -> bool:
        normalized = self._normalize_word(word)
        if not normalized.startswith("i") or len(normalized) < 3:
            return False
        return self._word_starts_with_two_consonants(normalized[1:])

    def _apply_empathetic_i(
        self,
        previous_word: str | None,
        corrected_word: str,
    ) -> str:
        normalized = self._normalize_word(corrected_word)
        if (
            not previous_word
            or not normalized
            or " " in corrected_word
            or "-" in corrected_word
            or "'" in corrected_word
            or normalized in self.dictionary_set
            or normalized.startswith("i")
            or normalized.startswith("t")
            or not self._word_ends_with_consonant(previous_word)
            or not self._word_starts_with_two_consonants(normalized)
        ):
            return corrected_word

        return self._match_capitalisation(corrected_word, f"i{normalized}")

    def _apostrophe_tail_variants(
        self,
        original_word: str,
        ordered: list[str],
        limit: int,
    ) -> list[str]:
        original_norm = self._normalize_word(original_word)
        if not original_norm or original_norm.endswith("'"):
            return []

        variants: list[str] = []
        seen = set(ordered)

        for base in ordered:
            if len(variants) + len(ordered) >= limit:
                break
            if not base or base.endswith("'"):
                continue
            if not base[-1] or base[-1] not in self.VOWELS:
                continue

            candidate = f"{base}'"
            if candidate in seen:
                continue
            if not self.meaning_for(candidate):
                continue

            if (
                self._word_distance(original_norm, base)
                <= self._max_distance(original_norm) + 1
            ):
                variants.append(candidate)
                seen.add(candidate)

        return variants

    def _preferred_apostrophe_choice(self, choices: list[dict]) -> str | None:
        for choice in choices:
            word = choice.get("word", "")
            if isinstance(word, str) and word.endswith("'"):
                return word
        return None

    def _correct_inline_article_word(
        self,
        word: str,
        *,
        previous: str | None,
    ) -> str | None:
        normalized = self._normalize_word(word)
        article_rules = getattr(self, "article_phrase_rules", None)
        if article_rules is None or "-" not in normalized:
            return None

        prefix, noun = normalized.split("-", 1)
        if prefix not in {
            "il",
            "l",
            "din",
            "dan",
            "iċ",
            "id",
            "in",
            "ir",
            "is",
            "it",
            "ix",
            "iz",
            "iż",
            "ċ",
            "d",
            "n",
            "r",
            "s",
            "t",
            "x",
            "ż",
            "z",
        }:
            return None

        corrected_noun = self.correct_word(noun)
        corrected_noun_norm = self._normalize_word(corrected_noun)
        if (
            prefix in {"ċ", "d", "n", "r", "s", "t", "x", "ż", "z"}
            and hasattr(article_rules, "_is_article_target")
            and article_rules._is_article_target(corrected_noun_norm)
        ):
            return f"{prefix}-{corrected_noun_norm}"

        candidate = f"{prefix}-{corrected_noun_norm}"
        article_match = article_rules.match_hyphenated_article_after(
            candidate,
            previous=previous,
        )
        if article_match is not None:
            return article_match.corrected
        return None

    def _expand_compact_xi_article(self, word: str) -> str | None:
        normalized = self._normalize_word(word)
        article_rules = getattr(self, "article_phrase_rules", None)
        if article_rules is None:
            return None

        for prefix in self.COMPACT_XI_ARTICLE_PREFIXES:
            if not normalized.startswith(prefix) or len(normalized) <= len(prefix):
                continue
            article_word = normalized[len("xi") :]
            if article_word.startswith("'"):
                article_word = article_word[1:]
            corrected_article = self._correct_inline_article_word(
                article_word,
                previous=None,
            )
            if corrected_article is not None:
                return f"xi {corrected_article}"
            article_match = article_rules.match_hyphenated_article_after(
                article_word,
                previous=None,
            )
            if article_match is not None:
                return f"xi {article_match.corrected}"
        return None

    def _build_missing_h_verb_repairs(self) -> dict[str, str]:
        suffix_generator = getattr(self, "suffix_generator", None)
        verb_index = getattr(suffix_generator, "verb_index", None)

        if verb_index is None:
            return {}

        repairs: dict[str, str] = {}

        for record in verb_index.records:
            if record.form_class != "F1" or record.tense not in {"IMP", "MPERF"}:
                continue

            radicals = verb_index.root_radicals(record)
            if len(radicals) < 3 or radicals[1] != "h":
                continue

            graphemes = self._graphemes(record.word)

            for index, token in enumerate(graphemes):
                if token != "h" or index == 0 or index + 1 >= len(graphemes):
                    continue

                if not (
                    graphemes[index - 1].isalpha() and graphemes[index + 1].isalpha()
                ):
                    continue

                typo = self._from_graphemes(graphemes[:index] + graphemes[index + 1 :])

                if typo != record.word and typo not in repairs:
                    repairs[typo] = record.word

                break

        return repairs

    def _missing_gh_mperf_repair(self, word: str) -> str | None:
        normalized = self._normalize_word(word)
        suffix_generator = getattr(self, "suffix_generator", None)
        verb_index = getattr(suffix_generator, "verb_index", None)
        orthographic = getattr(self, "orthographic_generator", None)
        if verb_index is None or not normalized:
            return None

        lookup = "j" + normalized[1:] if normalized.startswith("i") else normalized
        anchor = verb_index.consonant_anchor(lookup)
        candidate_records = []
        for index in range(len(anchor) + 1):
            candidate_anchor = anchor[:index] + "għ" + anchor[index:]
            candidate_records.extend(verb_index.by_anchor.get(candidate_anchor, ()))

        matches: set[str] = set()
        for record in candidate_records:
            if record.tense != "MPERF" or "għ" not in record.word:
                continue
            graphemes = self._graphemes(record.word)
            for index, token in enumerate(graphemes):
                if token != "għ":
                    continue
                missing_gh = self._from_graphemes(
                    graphemes[:index] + graphemes[index + 1 :]
                )
                forms = [missing_gh]
                if orthographic is not None:
                    forms.extend(orthographic.substitute_i_ie(missing_gh))
                if "ie" in missing_gh:
                    forms.append(missing_gh.replace("ie", "e"))
                forms.extend(
                    "i" + form[1:]
                    for form in list(forms)
                    if form.startswith("j") and len(form) > 2
                )
                if normalized in {
                    self._normalize_word(form) for form in forms
                }:
                    matches.add(record.word)

        return next(iter(matches)) if len(matches) == 1 else None

    def _missing_h_verb_repair(self, word: str) -> str | None:
        normalized = self._normalize_word(word)

        if self._missing_h_verb_repairs is None:
            self._missing_h_verb_repairs = self._build_missing_h_verb_repairs()

        return self._missing_h_verb_repairs.get(normalized)

    def correct_word(self, word: str) -> str:
        if not word:
            return word

        normalized = self._normalize_word(word)
        if not normalized or len(normalized) > MAX_WORD_LENGTH:
            return word
        if normalized == "mid":
            return word

        social_comment_repair = self.SOCIAL_COMMENT_REPAIRS.get(normalized)
        if social_comment_repair:
            return self._match_capitalisation(word, social_comment_repair)

        if normalized in {"l-hawn", "l-hemm"}:
            return self._match_capitalisation(word, normalized[2:])

        if normalized.startswith("l'") and len(normalized) > 2:
            article_tail = normalized[2:]
            corrected_tail = self._article_tail_repair(article_tail)
            if corrected_tail:
                if self._supports_l_apostrophe_tail(corrected_tail):
                    return self._match_capitalisation(
                        word,
                        f"l'{corrected_tail}",
                    )
                return self._match_capitalisation(word, f"l-{corrected_tail}")

        apostrophe_prefix_word = self._valid_apostrophe_prefix_word(normalized)
        if apostrophe_prefix_word is not None:
            return self._match_capitalisation(word, apostrophe_prefix_word)

        # ------------------------------------------------------------------
        # STRICT PRIORITY PIPELINE
        # ------------------------------------------------------------------

        if normalized in self.dictionary_set:
            missing_h_verb_match = self._missing_h_verb_repair(normalized)
            if missing_h_verb_match and normalized not in {"tajru", "m'hemmx"}:
                return self._match_capitalisation(word, missing_h_verb_match)
            if normalized not in {"tajru"}:
                return word

        orthographic_generator = getattr(
            self,
            "orthographic_generator",
            None,
        )

        if orthographic_generator is not None and hasattr(
            orthographic_generator,
            "correct_shortcut_letters",
        ):
            shortcut_match = orthographic_generator.correct_shortcut_letters(word)

            if shortcut_match:
                return self._match_capitalisation(
                    word,
                    shortcut_match,
                )

        # Typed gh is an explicit shortcut for għ. Resolve an exact dictionary
        # match before compact phrase rules can reinterpret the same letters.
        if (
            "gh" in normalized
            and orthographic_generator is not None
            and hasattr(orthographic_generator, "correct_gh_priority")
        ):
            gh_shortcut_match = orthographic_generator.correct_gh_priority(word)
            if gh_shortcut_match:
                return self._match_capitalisation(word, gh_shortcut_match)

        for combined_variant in self._dictionary_i_ie_shortcut_variants(normalized):
            return self._match_capitalisation(word, combined_variant)

        lexicalized_forms = self._lexicalized_form_variants(normalized)
        if lexicalized_forms:
            return self._match_capitalisation(word, lexicalized_forms[0])

        pattern_repairs = self._pattern_repair_variants(normalized)
        if pattern_repairs and (
            normalized not in self.dictionary_set or normalized in {"tajru"}
        ):
            return self._match_capitalisation(word, pattern_repairs[0])

        manual_repairs = self._manual_repair_variants(normalized)
        if manual_repairs and normalized not in self.dictionary_set:
            return self._match_capitalisation(word, manual_repairs[0])

        compact_xi_article = self._expand_compact_xi_article(normalized)
        if compact_xi_article is not None:
            return self._match_capitalisation(word, compact_xi_article)

        fused_preposition_rules = getattr(self, "fused_preposition_rules", None)
        if fused_preposition_rules is not None:
            fused_match = fused_preposition_rules.match(normalized)
            if fused_match is not None:
                return self._match_capitalisation(word, fused_match.corrected)

        article_rules = getattr(self, "article_phrase_rules", None)
        if article_rules is not None:
            compact_article = article_rules.match_compact_preposition_article(
                normalized,
            )
            if compact_article is not None:
                return self._match_capitalisation(
                    word,
                    compact_article.corrected,
                )
            inline_article = self._correct_inline_article_word(
                normalized,
                previous=None,
            )
            if inline_article is not None:
                return self._match_capitalisation(word, inline_article)

        noun_suffix_match = self._correct_noun_possessive_suffix(normalized)
        if noun_suffix_match:
            return self._match_capitalisation(word, noun_suffix_match)

        if article_rules is not None:
            collapsed = article_rules.collapse_three_same_consonants(normalized)
            if collapsed != normalized:
                return self._match_capitalisation(word, collapsed)

        if (
            orthographic_generator is not None
            and len(self._letter_tokens(normalized)) <= 4
            and not any(marker in normalized for marker in ("għ", "gh", "h", "ħ"))
            and hasattr(
                orthographic_generator,
                "correct_i_ie_confusion",
            )
        ):
            i_ie_match = orthographic_generator.correct_i_ie_confusion(word)
            if i_ie_match:
                return self._match_capitalisation(word, i_ie_match)

        # Stage 1: High-priority għ repairs before shortcut letters or
        # broad dictionary scoring can compete.
        if orthographic_generator is not None and hasattr(
            orthographic_generator,
            "correct_gh_priority",
        ):
            gh_priority_match = orthographic_generator.correct_gh_priority(word)

            if gh_priority_match and not self._violates_ghi_sequence_rule(
                normalized,
                gh_priority_match,
            ):
                return self._match_capitalisation(
                    word,
                    gh_priority_match,
                )

        # Keyboard shortcuts for Maltese letters:
        # h -> ħ, c -> ċ, z -> ż, g -> ġ
        if orthographic_generator is not None and hasattr(
            orthographic_generator,
            "correct_shortcut_letters",
        ):
            shortcut_match = orthographic_generator.correct_shortcut_letters(word)

            if shortcut_match:
                return self._match_capitalisation(
                    word,
                    shortcut_match,
                )

            if hasattr(orthographic_generator, "shortcut_letter_variants"):
                for shortcut_variant in orthographic_generator.shortcut_letter_variants(
                    normalized
                ):
                    gh_after_shortcut = orthographic_generator.correct_gh_priority(
                        shortcut_variant
                    )

                    if gh_after_shortcut:
                        return self._match_capitalisation(
                            word,
                            gh_after_shortcut,
                        )

        # d/t confusion for an invalid word.
        # Examples:
        #   rqatt -> rqadt
        #   rmiet remains unchanged here because it was accepted above.
        if orthographic_generator is not None and hasattr(
            orthographic_generator,
            "correct_d_t_confusion",
        ):
            dt_match = orthographic_generator.correct_d_t_confusion(word)

            if dt_match:
                return self._match_capitalisation(
                    word,
                    dt_match,
                )

        # b/p confusion for an invalid word.
        # This can cooperate with strict gh -> għ expansion.
        if orthographic_generator is not None and hasattr(
            orthographic_generator,
            "correct_b_p_confusion",
        ):
            bp_match = orthographic_generator.correct_b_p_confusion(word)

            if bp_match:
                return self._match_capitalisation(
                    word,
                    bp_match,
                )

        dt_double_match = self._correct_d_t_then_double(word)
        if dt_double_match:
            return dt_double_match

        if hasattr(self, "doubled_letter_generator"):
            j_priority_match = self.doubled_letter_generator.correct_j_priority(word)
            if j_priority_match:
                return j_priority_match

        if hasattr(
            self, "suffix_generator"
        ) and self.suffix_generator.exact_suffix_matches(word):
            return word

        if orthographic_generator is not None and hasattr(
            orthographic_generator,
            "correct_extra_double",
        ):
            extra_double_match = orthographic_generator.correct_extra_double(word)

            if extra_double_match:
                return self._match_capitalisation(
                    word,
                    extra_double_match,
                )

        missing_h_verb_match = self._missing_h_verb_repair(normalized)
        if missing_h_verb_match and normalized not in {"m'hemmx"}:
            return self._match_capitalisation(word, missing_h_verb_match)

        if orthographic_generator is not None and hasattr(
            orthographic_generator,
            "correct_missing_h_after_d",
        ):
            missing_h_after_d_match = orthographic_generator.correct_missing_h_after_d(
                word
            )

            if missing_h_after_d_match:
                return self._match_capitalisation(
                    word,
                    missing_h_after_d_match,
                )

        # d/t confusion for an invalid word after whole-word exceptions.
        # This is retained below only for compatibility with older helpers.
        if (
            orthographic_generator is not None
            and hasattr(
                orthographic_generator,
                "correct_d_t_confusion",
            )
            and not hasattr(
                orthographic_generator,
                "correct_b_p_confusion",
            )
        ):
            dt_match = orthographic_generator.correct_d_t_confusion(word)

            if dt_match:
                return self._match_capitalisation(
                    word,
                    dt_match,
                )

        # i/ie confusion for an invalid word.
        # Example:
        #   jin -> jien
        if orthographic_generator is not None and hasattr(
            orthographic_generator,
            "correct_i_ie_confusion",
        ):
            i_ie_match = orthographic_generator.correct_i_ie_confusion(word)

            if i_ie_match:
                return self._match_capitalisation(
                    word,
                    i_ie_match,
                )

        if orthographic_generator is not None and hasattr(
            orthographic_generator,
            "correct_final_aw_to_ghu",
        ):
            aw_ghu_match = orthographic_generator.correct_final_aw_to_ghu(word)

            if aw_ghu_match:
                return self._match_capitalisation(
                    word,
                    aw_ghu_match,
                )

        # Final għ/h/ħ confusion for an invalid word only.
        if orthographic_generator is not None and hasattr(
            orthographic_generator,
            "correct_final_gh_h_hbar_confusion",
        ):
            final_gh_h_hbar_match = (
                orthographic_generator.correct_final_gh_h_hbar_confusion(word)
            )

            if final_gh_h_hbar_match:
                return self._match_capitalisation(
                    word,
                    final_gh_h_hbar_match,
                )

        # Stage 0.5: Missing doubled-letter repair.
        # Example:
        #   basejt -> bassejt
        if hasattr(self, "doubled_letter_generator"):
            doubled_letter = self.doubled_letter_generator.correct_missing_double(word)
            if doubled_letter:
                return doubled_letter

        # Stage 2: Exact safe orthographic variants from helper.
        # Example:
        #   ibghatu -> ibgħatu
        #   ibaght  -> ibgħat
        if hasattr(self, "orthographic_generator"):
            exact_ortho = self.orthographic_generator.correct_strict(word)
        else:
            exact_ortho = self._try_exact_variants(
                word, self._strict_lookup_variants(normalized)
            )

        if exact_ortho:
            return exact_ortho

        suffix_parse_guard = False

        # Stage 1.5: Real-time generated DO/IDO suffix forms.
        # This uses the fast ending-first parser in helpers/suffix_generator.py.
        # Example:
        #   għamilni   -> parsed as possible -ni
        #   għamlilkom -> parsed as possible -lkom / -ilkom
        #   jgħamilhom -> parsed as possible -hom / -lhom
        if hasattr(self, "suffix_generator"):
            liex_lix_match = self._validated_liex_to_lix_repair(normalized)
            if liex_lix_match:
                return self._match_capitalisation(word, liex_lix_match)

            generated_suffix = self.suffix_generator.correct_suffix(word)

            if generated_suffix:
                generated_norm = self._normalize_word(generated_suffix)
                if generated_norm not in self.dictionary_set:
                    generated_row = self._candidate_score(
                        normalized,
                        generated_norm,
                        stage="generated_suffix_compare",
                    )

                    # Strong suffix parses are already narrow and expensive
                    # to produce. Only consult broad dictionary scoring when
                    # the generated form is weak enough to be suspicious.
                    if generated_row.score > 0.36:
                        lexical_best = self._best_ranked_candidate(
                            normalized,
                            stage="pre_suffix_dictionary",
                            score_limit=0.20,
                            max_distance=self._max_distance(normalized),
                        )
                        if (
                            lexical_best
                            and lexical_best.edit_distance <= 1
                            and lexical_best.score + 0.10 < generated_row.score
                        ):
                            return self._match_capitalisation(
                                word,
                                lexical_best.candidate,
                            )
                return self._match_capitalisation(word, generated_suffix)

            # Suffix-looking words can otherwise fall through into broad
            # whole-dictionary scoring, which is slow and usually misleading.
            if self.suffix_generator.has_suffix_parse(word):
                suffix_parse_guard = True

        # Stage 2: Suffix repairs.
        # If exact match fails, we REPLACE the base word with the repaired version.
        # This ensures Stages 3+ (like inserting għ) apply directly to the new suffix!
        suffix_vars = self._suffix_repair_variants(normalized)
        if suffix_vars:
            exact_suffix = self._try_exact_variants(word, suffix_vars)
            if exact_suffix:
                return exact_suffix

            # Find the longest matching suffix to use as our new base word
            for suffix, replacement in self._sorted_suffix_repairs:
                if normalized.endswith(suffix):
                    normalized = normalized[: -len(suffix)] + replacement
                    break  # Stop at the longest match to prevent double-applying

        # Pre-compute metrics needed for the remaining stages
        # (Now using the potentially updated 'normalized' word!)
        max_distance = self._max_distance(normalized)
        target_vowel_count = self._count_vowels(normalized)
        target_vowel_sequence = self._vowel_sequence(normalized)

        # Stage 7: Remove h and check again
        remove_h = self._remove_token(normalized, "h")
        exact_rm_h = self._try_exact_variants(word, remove_h)
        if exact_rm_h:
            return exact_rm_h

        # Stage 2.5: Remove q and check again
        remove_q = self._remove_token(normalized, "q")
        exact_rm_q = self._try_exact_variants(word, remove_q)
        if exact_rm_q:
            return exact_rm_q

        # Stage 3: Insert għ next to vowels (Exact & Ranked)
        insert_gh = self._insert_token_next_to_vowels(normalized, "għ")
        exact_gh = self._try_exact_variants(word, insert_gh)
        if exact_gh:
            return exact_gh

        # Stage 4: Insert h next to vowels (Exact & Ranked)
        insert_h = self._insert_token_next_to_vowels(normalized, "h")
        exact_h = self._try_exact_variants(word, insert_h)
        if exact_h:
            return exact_h

        if suffix_parse_guard:
            return word

        ranked_rm_h = self._try_ranked_from_variants(
            word, remove_h, stage="remove_h", score_limit=0.42
        )
        if ranked_rm_h:
            return ranked_rm_h

        ranked_gh = self._try_ranked_from_variants(
            word, insert_gh, stage="insert_gh_near_vowel", score_limit=0.42
        )
        if ranked_gh:
            return ranked_gh

        ranked_h = self._try_ranked_from_variants(
            word, insert_h, stage="insert_h_near_vowel", score_limit=0.42
        )
        if ranked_h:
            return ranked_h

        # Stage 5: Closest dictionary words with same vowel count
        same_vowel_best = self._best_ranked_candidate(
            normalized,
            stage="same_vowel_count",
            candidate_filter=lambda c: self.word_vowel_counts[c] == target_vowel_count,
            score_limit=0.48,
            max_distance=max_distance,
        )
        if same_vowel_best:
            return self._match_capitalisation(word, same_vowel_best.candidate)

        # Stage 6: Remove għ and check again
        remove_gh = self._remove_token(normalized, "għ")
        exact_rm_gh = self._try_exact_variants(word, remove_gh)
        if exact_rm_gh:
            return exact_rm_gh

        ranked_rm_gh = self._try_ranked_from_variants(
            word, remove_gh, stage="remove_gh", score_limit=0.42
        )
        if ranked_rm_gh:
            return ranked_rm_gh

        ranked_rm_q = self._try_ranked_from_variants(
            word, remove_q, stage="remove_q", score_limit=0.42
        )
        if ranked_rm_q:
            return ranked_rm_q

        # Stage 8: Broader but still conservative composite score
        broad_best = self._best_ranked_candidate(
            normalized,
            stage="broad_score",
            score_limit=0.52,
            max_distance=max_distance,
        )
        if broad_best:
            candidate_seq = self._vowel_sequence(broad_best.candidate)
            if candidate_seq == target_vowel_sequence or broad_best.score <= 0.38:
                return self._match_capitalisation(word, broad_best.candidate)

        return word

    def _insert_token_next_to_vowels(self, word: str, token: str) -> list[str]:
        """
        Ordered insertion variants.

        The real generation lives in helpers/orthographic_generator.py.
        """
        if hasattr(self, "orthographic_generator"):
            return self.orthographic_generator.insert_token_next_to_vowels(word, token)

        normalized = self._normalize_word(word)
        variants: list[str] = []
        g = self._graphemes(normalized)

        def add(candidate: str) -> None:
            if candidate and candidate not in variants:
                variants.append(candidate)

        for i, ch in enumerate(g):
            if ch not in self.VOWELS:
                continue

            before = self._from_graphemes(g[:i] + [token] + g[i:])
            after = self._from_graphemes(g[: i + 1] + [token] + g[i + 1 :])

            if i == 0 or g[i - 1] != token:
                add(before)
            if i + 1 >= len(g) or g[i + 1] != token:
                add(after)

        return variants

    def suggest(
        self, word: str, limit: int = 8, edit_distance_tolerance: int = 1
    ) -> list[str]:
        normalized = self._normalize_word(word)

        if not normalized:
            return []

        if self._is_initial_capitalized(word):
            if normalized in self.place_word_set:
                return [self.place_word_display.get(normalized, word)]
            if normalized in self.dictionary_set:
                return [self._match_capitalisation(word, normalized)]
            corrected_place = self._correct_place_word(word)
            if corrected_place:
                return [corrected_place]
            strict = self._try_exact_variants(
                word,
                self._strict_lookup_variants(normalized),
            )
            return [strict] if strict else []

        suggestions: list[str] = []
        trusted_generated: set[str] = set()
        corrected_hint = ""

        social_comment_repair = self.SOCIAL_COMMENT_REPAIRS.get(normalized)
        if social_comment_repair:
            trusted_generated.add(self._normalize_word(social_comment_repair))

        def add_generated(candidate: str) -> None:
            candidate = self._normalize_word(candidate)

            if not candidate or candidate in suggestions:
                return
            if (
                candidate not in trusted_generated
                and self._is_implausible_vowel_swap(normalized, candidate)
            ):
                return
            if (
                candidate not in trusted_generated
                and not self._is_plausible_whole_word_suggestion(
                    normalized,
                    candidate,
                    corrected=corrected_hint,
                )
            ):
                return
            if (
                candidate not in trusted_generated
                and candidate.endswith("uha")
                and not normalized.endswith(("u", "w"))
            ):
                return
            if normalized.startswith("h") and candidate == normalized[1:]:
                return
            suggestions.append(candidate)

        if social_comment_repair:
            add_generated(social_comment_repair)
            if len(suggestions) >= limit:
                return suggestions[:limit]

        lexicalized_forms = self._lexicalized_form_variants(normalized)
        trusted_generated.update(self._normalize_word(candidate) for candidate in lexicalized_forms)

        force_pre_exact_repair = (
            normalized in self.dictionary_set
            and self._missing_h_verb_repair(normalized) is not None
        )

        if (
            normalized in self.dictionary_set
            and normalized not in {"tajru"}
            and not force_pre_exact_repair
        ):
            add_generated(normalized)
            if lexicalized_forms and self._normalize_word(lexicalized_forms[0]) == normalized:
                for candidate in lexicalized_forms[1:]:
                    add_generated(candidate)
            return suggestions[:limit]

        for candidate in lexicalized_forms:
            add_generated(candidate)

            if len(suggestions) >= limit:
                return suggestions[:limit]

        if lexicalized_forms:
            return suggestions[:limit]

        if "gh" in normalized:
            gh_orthographic = getattr(self, "orthographic_generator", None)
            if (
                gh_orthographic is not None
                and hasattr(gh_orthographic, "correct_gh_priority")
            ):
                gh_shortcut_match = gh_orthographic.correct_gh_priority(normalized)
                if gh_shortcut_match:
                    trusted_generated.add(
                        self._normalize_word(gh_shortcut_match)
                    )
                    add_generated(gh_shortcut_match)
                    return suggestions[:limit]

        pattern_repairs = self._pattern_repair_variants(normalized)
        trusted_generated.update(self._normalize_word(candidate) for candidate in pattern_repairs)
        for candidate in pattern_repairs:
            add_generated(candidate)

            if len(suggestions) >= limit:
                return suggestions[:limit]

        if pattern_repairs:
            return suggestions[:limit]

        manual_repair_variants = self._manual_repair_variants(normalized)
        trusted_generated.update(self._normalize_word(candidate) for candidate in manual_repair_variants)
        for candidate in manual_repair_variants:
            add_generated(candidate)

            if len(suggestions) >= limit:
                return suggestions[:limit]

        if normalized in self.MANUAL_WORD_REPAIRS:
            return suggestions[:limit]

        compact_xi_article = self._expand_compact_xi_article(normalized)
        if compact_xi_article is not None:
            return [compact_xi_article][:limit]

        if normalized in {"tal", "fil", "bil", "lill", "mid"}:
            suggestions.append(f"{normalized}-")
            if normalized == "mid":
                return suggestions[:limit]

        early_orthographic = getattr(self, "orthographic_generator", None)
        if early_orthographic is not None and hasattr(
            early_orthographic, "dictionary_shortcut_variants"
        ):
            shortcut_matches = early_orthographic.dictionary_shortcut_variants(
                normalized
            )
            if shortcut_matches:
                for candidate in shortcut_matches:
                    add_generated(candidate)
                if normalized not in {"habba", "ħabba"}:
                    return suggestions[:limit]

        fused_preposition_rules = getattr(self, "fused_preposition_rules", None)
        if fused_preposition_rules is not None:
            fused_match = fused_preposition_rules.match(normalized)
            if fused_match is not None:
                for choice in fused_match.choices:
                    word_choice = choice.get("word")
                    if word_choice and word_choice not in suggestions:
                        suggestions.append(word_choice)
                if fused_match.corrected not in suggestions:
                    suggestions.insert(0, fused_match.corrected)
                return suggestions[:limit]

        correction_changed = False
        corrected_word = self.correct_word(word)
        corrected_norm = self._normalize_word(corrected_word)
        corrected_hint = corrected_norm
        if corrected_norm != normalized:
            add_generated(corrected_word)
            correction_changed = True
            if normalized.startswith("ghand") and corrected_norm.startswith("għand"):
                return suggestions[:limit]

        noun_suffix_base = self._noun_possessive_base_for_surface(
            corrected_norm
        ) or self._noun_possessive_base_for_surface(normalized)
        if noun_suffix_base:
            add_generated(corrected_norm)
            return suggestions[:limit]

        def add(candidate: str) -> None:
            candidate = self._normalize_word(candidate)

            if (
                candidate
                and candidate in self.dictionary_set
                and candidate not in suggestions
            ):
                if self._is_implausible_vowel_swap(normalized, candidate):
                    return
                if not self._is_plausible_whole_word_suggestion(
                    normalized,
                    candidate,
                    corrected=corrected_norm,
                ):
                    return
                if candidate.endswith("uha") and not normalized.endswith(("u", "w")):
                    return
                if normalized.startswith("h") and candidate == normalized[1:]:
                    return
                if candidate.startswith(("b'", "f'", "t'", "x'", "m'", "s'")):
                    tail = candidate[2:]
                    if (
                        corrected_norm in self.dictionary_set
                        and tail
                        and self._word_distance(tail, corrected_norm) <= 1
                    ):
                        return
                if (
                    corrected_norm != normalized
                    and corrected_norm in self.dictionary_set
                    and candidate != corrected_norm
                    and self._extract_consonant_anchor(candidate)
                    != self._extract_consonant_anchor(corrected_norm)
                    and not candidate.endswith(("ha", "hom"))
                ):
                    return
                if (
                    normalized in self.dictionary_set or corrected_norm == normalized
                ) and self._count_vowels(candidate) != self._count_vowels(normalized):
                    return
                if (
                    corrected_norm != normalized
                    and corrected_norm in self.dictionary_set
                    and self._word_distance(normalized, corrected_norm) <= 2
                    and self._word_distance(corrected_norm, candidate) > 1
                    and not candidate.endswith(("ha", "hom"))
                ):
                    return
                if candidate in {"xil-", "xil", "x'l", "x'l-"}:
                    return
                if (
                    len(self._letter_tokens(candidate)) == 1
                    and len(self._letter_tokens(normalized)) > 1
                ):
                    return
                suggestions.append(candidate)

        # Preserve the original word first when it is already valid.
        if normalized in self.dictionary_set and not force_pre_exact_repair:
            add(normalized)

        orthographic_generator = getattr(
            self,
            "orthographic_generator",
            None,
        )

        for candidate in self._dictionary_i_ie_shortcut_variants(normalized):
            add(candidate)

            if len(suggestions) >= limit:
                return suggestions[:limit]

        # Add high-priority għ dictionary matches. Suggestions may include the
        # reverse għ movement used for alternatives such as għamlu -> agħmlu;
        # automatic correction does not use that reverse movement.
        if orthographic_generator is not None and hasattr(
            orthographic_generator,
            "dictionary_gh_suggestion_variants",
        ):
            for source in (normalized, corrected_norm):
                for (
                    candidate
                ) in orthographic_generator.dictionary_gh_suggestion_variants(source):
                    add(candidate)

                    if len(suggestions) >= limit:
                        return suggestions[:limit]

        # Add h/ħ, c/ċ, z/ż and g/ġ dictionary matches.
        if orthographic_generator is not None and hasattr(
            orthographic_generator,
            "dictionary_shortcut_variants",
        ):
            for candidate in orthographic_generator.dictionary_shortcut_variants(
                normalized
            ):
                add(candidate)

                if len(suggestions) >= limit:
                    return suggestions[:limit]

        # Add d/t dictionary matches.
        if orthographic_generator is not None and hasattr(
            orthographic_generator,
            "dictionary_d_t_variants",
        ):
            for candidate in orthographic_generator.dictionary_d_t_variants(normalized):
                add(candidate)

                if len(suggestions) >= limit:
                    return suggestions[:limit]

        # Add accidental extra-double fixes such as jaffux -> jafux.
        if orthographic_generator is not None and hasattr(
            orthographic_generator,
            "dictionary_remove_extra_double_variants",
        ):
            for (
                candidate
            ) in orthographic_generator.dictionary_remove_extra_double_variants(
                normalized
            ):
                add(candidate)

                if len(suggestions) >= limit:
                    return suggestions[:limit]

        # Add missing h after d fixes such as jidru -> jidhru.
        if orthographic_generator is not None and hasattr(
            orthographic_generator,
            "dictionary_insert_h_after_d_variants",
        ):
            for (
                candidate
            ) in orthographic_generator.dictionary_insert_h_after_d_variants(
                normalized
            ):
                add(candidate)

                if len(suggestions) >= limit:
                    return suggestions[:limit]

        # Add b/p dictionary matches.
        if orthographic_generator is not None and hasattr(
            orthographic_generator,
            "dictionary_b_p_variants",
        ):
            for candidate in orthographic_generator.dictionary_b_p_variants(normalized):
                add(candidate)

                if len(suggestions) >= limit:
                    return suggestions[:limit]

        # Add i/ie dictionary matches.
        if orthographic_generator is not None and hasattr(
            orthographic_generator,
            "dictionary_i_ie_variants",
        ):
            for candidate in orthographic_generator.dictionary_i_ie_variants(
                normalized
            ):
                add(candidate)

                if len(suggestions) >= limit:
                    return suggestions[:limit]

        # Add final ja/ha dictionary matches such as fija -> fiha.
        if orthographic_generator is not None and hasattr(
            orthographic_generator,
            "dictionary_final_j_h_variants",
        ):
            for candidate in orthographic_generator.dictionary_final_j_h_variants(
                normalized
            ):
                add(candidate)

                if len(suggestions) >= limit:
                    return suggestions[:limit]

        # Add final għ/h/ħ dictionary matches.
        if orthographic_generator is not None and hasattr(
            orthographic_generator,
            "dictionary_final_gh_h_hbar_variants",
        ):
            for candidate in orthographic_generator.dictionary_final_gh_h_hbar_variants(
                normalized
            ):
                add(candidate)

                if len(suggestions) >= limit:
                    return suggestions[:limit]

        # A valid word remains the first result, followed by its possible
        # shortcut, d/t, i/ie, and final għ/h/ħ alternatives.
        if normalized in self.dictionary_set and not force_pre_exact_repair:
            return suggestions[:limit]

        priority_variant_groups = [
            self._strict_lookup_variants(normalized),
            self._suffix_repair_variants(normalized),
            self._insert_token_next_to_vowels(
                normalized,
                "għ",
            ),
            self._remove_token(normalized, "għ"),
            self._remove_token(normalized, "h"),
        ]

        for group in priority_variant_groups:
            for variant in group:
                for lookup in self._strict_lookup_variants(variant):
                    add(lookup)

                    if len(suggestions) >= limit:
                        return suggestions[:limit]

        if orthographic_generator is not None and hasattr(
            orthographic_generator,
            "dictionary_final_aw_to_ghu_variants",
        ):
            for candidate in orthographic_generator.dictionary_final_aw_to_ghu_variants(
                normalized
            ):
                add(candidate)

                if len(suggestions) >= limit:
                    return suggestions[:limit]

        noun_suffix_match = self._correct_noun_possessive_suffix(normalized)
        if noun_suffix_match:
            add_generated(noun_suffix_match)

            if len(suggestions) >= limit:
                return suggestions[:limit]

        if hasattr(self, "suffix_generator"):
            liex_lix_match = self._validated_liex_to_lix_repair(normalized)
            if liex_lix_match:
                add_generated(liex_lix_match)
                generated_from_lix = self.suffix_generator.correct_suffix(
                    liex_lix_match
                )
                if generated_from_lix:
                    add_generated(generated_from_lix)

                if len(suggestions) >= limit:
                    return suggestions[:limit]

            if self.suffix_generator.exact_suffix_matches(word):
                add_generated(normalized)
                return suggestions[:limit]

            generated_suffix = self.suffix_generator.correct_suffix(word)

            if generated_suffix:
                add_generated(generated_suffix)

                if hasattr(self.suffix_generator, "suggest_suffixes"):
                    for suffix_suggestion in self.suffix_generator.suggest_suffixes(
                        word,
                        limit=limit,
                    ):
                        add_generated(suffix_suggestion)

                        if len(suggestions) >= limit:
                            return suggestions[:limit]

                if len(suggestions) >= limit:
                    return suggestions[:limit]

        if correction_changed and corrected_norm in self.dictionary_set:
            return suggestions[:limit]

        if correction_changed:
            return suggestions[:limit]

        candidates_to_score = set(self._get_candidates(normalized))
        for variant in lexicalized_forms:
            candidates_to_score.update(self._get_candidates(variant))

        ortho_gen = getattr(self, "orthographic_generator", None)
        if ortho_gen is not None:
            combined_variants = []
            for t_d in ortho_gen.substitute_d_t(normalized):
                combined_variants.extend(ortho_gen.insert_token_next_to_vowels(t_d, "h"))
                combined_variants.extend(ortho_gen.insert_token_next_to_vowels(t_d, "għ"))
            for b_p in ortho_gen.substitute_b_p(normalized):
                combined_variants.extend(ortho_gen.insert_token_next_to_vowels(b_p, "h"))
                combined_variants.extend(ortho_gen.insert_token_next_to_vowels(b_p, "għ"))
            
            for combined in combined_variants:
                if self._normalize_word(combined) in self.dictionary_set:
                    candidates_to_score.add(combined)

        rows = [
            self._candidate_score(
                normalized,
                candidate,
                stage="suggest",
            )
            for candidate in candidates_to_score
            if not self._is_implausible_vowel_swap(normalized, candidate)
        ]

        rows.sort(
            key=lambda row: (
                row.score,
                row.edit_distance,
                row.candidate,
            )
        )

        max_distance = self._max_distance(normalized) + (edit_distance_tolerance - 1)

        for row in rows:
            if self._is_acceptable_match(
                row,
                max_distance,
                score_limit=0.58,
            ):
                add(row.candidate)

                if len(suggestions) >= limit:
                    break

        return suggestions[:limit]

    def debug_word(
        self,
        word: str,
        limit: int = 12,
        edit_distance_tolerance: int = 1,
    ) -> dict:
        """
        Return diagnostic information for one word.

        Includes:
        - shortcut-letter substitutions;
        - d/t substitutions;
        - general ranked candidates;
        - the final selected correction.
        """
        normalized = self._normalize_word(word)
        manual_repair_variants = self._manual_repair_variants(normalized)

        orthographic_generator = getattr(
            self,
            "orthographic_generator",
            None,
        )

        shortcut_variants: list[str] = []
        shortcut_dictionary_matches: list[str] = []
        shortcut_auto_correction: str | None = None

        gh_priority_dictionary_matches: list[str] = []
        gh_priority_auto_correction: str | None = None
        d_t_variants: list[str] = []
        d_t_dictionary_matches: list[str] = []
        d_t_auto_correction: str | None = None
        b_p_variants: list[str] = []
        b_p_dictionary_matches: list[str] = []
        b_p_auto_correction: str | None = None
        i_ie_variants: list[str] = []
        i_ie_dictionary_matches: list[str] = []
        i_ie_auto_correction: str | None = None
        final_gh_h_hbar_variants: list[str] = []
        final_gh_h_hbar_dictionary_matches: list[str] = []
        final_gh_h_hbar_auto_correction: str | None = None

        if orthographic_generator is not None:
            if hasattr(
                orthographic_generator,
                "shortcut_letter_variants",
            ):
                shortcut_variants = orthographic_generator.shortcut_letter_variants(
                    normalized
                )

            if hasattr(
                orthographic_generator,
                "dictionary_shortcut_variants",
            ):
                shortcut_dictionary_matches = (
                    orthographic_generator.dictionary_shortcut_variants(normalized)
                )

            if hasattr(
                orthographic_generator,
                "correct_shortcut_letters",
            ):
                shortcut_auto_correction = (
                    orthographic_generator.correct_shortcut_letters(normalized)
                )

            if hasattr(
                orthographic_generator,
                "dictionary_gh_priority_variants",
            ):
                gh_priority_dictionary_matches = (
                    orthographic_generator.dictionary_gh_priority_variants(normalized)
                )

            if hasattr(
                orthographic_generator,
                "correct_gh_priority",
            ):
                gh_priority_auto_correction = (
                    orthographic_generator.correct_gh_priority(normalized)
                )

            if hasattr(
                orthographic_generator,
                "substitute_d_t",
            ):
                d_t_variants = orthographic_generator.substitute_d_t(normalized)

            if hasattr(
                orthographic_generator,
                "dictionary_d_t_variants",
            ):
                d_t_dictionary_matches = orthographic_generator.dictionary_d_t_variants(
                    normalized
                )

            if hasattr(
                orthographic_generator,
                "correct_d_t_confusion",
            ):
                d_t_auto_correction = orthographic_generator.correct_d_t_confusion(
                    normalized
                )

            if hasattr(
                orthographic_generator,
                "substitute_b_p",
            ):
                b_p_variants = orthographic_generator.substitute_b_p(normalized)

            if hasattr(
                orthographic_generator,
                "dictionary_b_p_variants",
            ):
                b_p_dictionary_matches = orthographic_generator.dictionary_b_p_variants(
                    normalized
                )

            if hasattr(
                orthographic_generator,
                "correct_b_p_confusion",
            ):
                b_p_auto_correction = orthographic_generator.correct_b_p_confusion(
                    normalized
                )

            if hasattr(
                orthographic_generator,
                "substitute_i_ie",
            ):
                i_ie_variants = orthographic_generator.substitute_i_ie(normalized)

            if hasattr(
                orthographic_generator,
                "dictionary_i_ie_variants",
            ):
                i_ie_dictionary_matches = (
                    orthographic_generator.dictionary_i_ie_variants(normalized)
                )

            if hasattr(
                orthographic_generator,
                "correct_i_ie_confusion",
            ):
                i_ie_auto_correction = orthographic_generator.correct_i_ie_confusion(
                    normalized
                )

            if hasattr(
                orthographic_generator,
                "substitute_final_gh_h_hbar",
            ):
                final_gh_h_hbar_variants = (
                    orthographic_generator.substitute_final_gh_h_hbar(normalized)
                )

            if hasattr(
                orthographic_generator,
                "dictionary_final_gh_h_hbar_variants",
            ):
                final_gh_h_hbar_dictionary_matches = (
                    orthographic_generator.dictionary_final_gh_h_hbar_variants(
                        normalized
                    )
                )

            if hasattr(
                orthographic_generator,
                "correct_final_gh_h_hbar_confusion",
            ):
                final_gh_h_hbar_auto_correction = (
                    orthographic_generator.correct_final_gh_h_hbar_confusion(normalized)
                )

        corrected = self.correct_word(word)
        suggestions = self.suggest(
            word, limit=limit, edit_distance_tolerance=edit_distance_tolerance
        )

        suggestion_details: list[dict] = []

        for suggestion in suggestions:
            suggestion_normalized = self._normalize_word(suggestion)

            suggestion_details.append(
                {
                    "word": suggestion,
                    "meaning": self.meaning_for(suggestion_normalized),
                    "in_dictionary": (suggestion_normalized in self.dictionary_set),
                    "distance": self._word_distance(
                        normalized,
                        suggestion_normalized,
                    ),
                }
            )

        rows = [
            self._candidate_score(
                normalized,
                candidate,
                stage="debug",
            )
            for candidate in self._get_candidates(normalized)
        ]

        rows.sort(
            key=lambda row: (
                row.score,
                row.edit_distance,
                row.candidate,
            )
        )

        return {
            "input": word,
            "normalized": normalized,
            "in_dictionary": (normalized in self.dictionary_set),
            "corrected": corrected,
            "changed": (self._normalize_word(corrected) != normalized),
            "manual_repair_variants": manual_repair_variants,
            "shortcut_variants": shortcut_variants,
            "shortcut_dictionary_matches": (shortcut_dictionary_matches),
            "shortcut_auto_correction": (shortcut_auto_correction),
            "gh_priority_dictionary_matches": (gh_priority_dictionary_matches),
            "gh_priority_auto_correction": (gh_priority_auto_correction),
            "d_t_variants": d_t_variants,
            "d_t_dictionary_matches": (d_t_dictionary_matches),
            "d_t_auto_correction": (d_t_auto_correction),
            "b_p_variants": b_p_variants,
            "b_p_dictionary_matches": (b_p_dictionary_matches),
            "b_p_auto_correction": (b_p_auto_correction),
            "i_ie_variants": i_ie_variants,
            "i_ie_dictionary_matches": (i_ie_dictionary_matches),
            "i_ie_auto_correction": (i_ie_auto_correction),
            "final_gh_h_hbar_variants": final_gh_h_hbar_variants,
            "final_gh_h_hbar_dictionary_matches": (final_gh_h_hbar_dictionary_matches),
            "final_gh_h_hbar_auto_correction": (final_gh_h_hbar_auto_correction),
            "suggestions": suggestions,
            "suggestion_details": suggestion_details,
            "strict_lookup_variants": (self._strict_lookup_variants(normalized)),
            "vowel_count": self._count_vowels(normalized),
            "vowel_sequence": self._vowel_sequence(normalized),
            "consonant_anchor": (self._extract_consonant_anchor(normalized)),
            "top_candidates": [asdict(row) for row in rows[:limit]],
        }

    def correct_text(self, text: str) -> str:
        if not text:
            return text
        return self.correct_text_rich(text)["corrected_text"]

    @lru_cache(maxsize=65536)
    def meaning_for(self, word: str) -> str:
        normalized = self._normalize_word(word)
        analytic_base = self.LEXICALIZED_ANALYTIC_MEANING_BASES.get(normalized)
        if analytic_base:
            if normalized in {"l-anqas", "l-inqas"}:
                return "the least, the smallest"
            base_meaning = meaning_index.meaning_for(analytic_base)
            if base_meaning:
                return base_meaning

        if normalized.startswith("l-") and len(normalized) > 2:
            superlative_tail = normalized[2:]
            if self._is_adjective_tagged_word(superlative_tail):
                adjective_meaning = meaning_index.meaning_for(superlative_tail)
                article_rules = getattr(self, "article_phrase_rules", None)
                if adjective_meaning and article_rules is not None:
                    superlative = article_rules._superlative_meaning(
                        adjective_meaning
                    )
                    if superlative:
                        return superlative

        if normalized.startswith("l'") and len(normalized) > 2:
            relative_tail = normalized[2:]
            relative_meaning = meaning_index.meaning_for(relative_tail)
            if not relative_meaning:
                relative_meaning = self.meaning_for(relative_tail)
            if relative_meaning:
                if self._is_adjective_tagged_word(relative_tail):
                    return f"which is {relative_meaning}"
                if self._is_verb_tagged_word(relative_tail):
                    if relative_meaning.casefold().startswith("to "):
                        relative_meaning = relative_meaning[3:]
                    for pronoun_prefix in ("he ", "she ", "it "):
                        if relative_meaning.casefold().startswith(pronoun_prefix):
                            relative_meaning = relative_meaning[len(pronoun_prefix):]
                            break
                    return f"which {relative_meaning}"
                if self._is_adverb_tagged_word(relative_tail) or self._is_preposition_tagged_word(
                    relative_tail
                ):
                    return f"which {relative_meaning}"

        noun_base = self._noun_possessive_base_for_surface(word)
        if noun_base:
            noun_meaning = meaning_index.meaning_for(noun_base)
            if noun_meaning:
                return noun_meaning

        direct_meaning = meaning_index.meaning_for(word)
        if direct_meaning:
            return direct_meaning

        suffix_generator = getattr(self, "suffix_generator", None)
        if suffix_generator is None or not hasattr(
            suffix_generator,
            "candidates_for_surface",
        ):
            return ""

        meanings: list[str] = []

        for candidate in suffix_generator.candidates_for_surface(word):
            meaning = format_suffix_candidate_meaning(
                candidate,
                fallback_gloss=meaning_index.meaning_for(candidate.base),
            )

            if meaning and meaning not in meanings:
                meanings.append(meaning)

        return " / ".join(meanings)

    def ambiguity_choices(
        self,
        original_word: str,
        corrected_word: str,
        limit: int = 2,
        edit_distance_tolerance: int = 1,
    ) -> list[dict]:

        original_norm = self._normalize_word(original_word)
        corrected_norm = self._normalize_word(corrected_word)

        limit = self._ambiguity_choice_limit(original_word, corrected_word, limit)

        suggestions = self.suggest(
            original_word, limit=8, edit_distance_tolerance=edit_distance_tolerance
        )

        ordered: list[str] = []

        def add(word: str) -> None:
            norm = self._normalize_word(word)
            if norm and norm not in ordered:
                ordered.append(norm)

        add(corrected_norm)

        if original_norm in self.dictionary_set and corrected_norm == original_norm:
            return [
                {
                    "word": self._match_capitalisation(original_word, corrected_norm),
                    "meaning": self.meaning_for(corrected_norm),
                }
            ][:limit]

        # Show dictionary-valid Maltese-letter alternatives.
        if hasattr(self, "orthographic_generator"):
            if hasattr(
                self.orthographic_generator,
                "dictionary_gh_suggestion_variants",
            ):
                for (
                    alternative
                ) in self.orthographic_generator.dictionary_gh_suggestion_variants(
                    corrected_norm
                ):
                    add(alternative)

                    if len(ordered) >= limit:
                        break

            shortcut_alternatives = (
                self.orthographic_generator.dictionary_shortcut_variants(original_norm)
            )

            for alternative in shortcut_alternatives:
                add(alternative)

                if len(ordered) >= limit:
                    break

        # A valid word may still be a d/t misspelling of another valid word.
        # Keep the original as the default, but show exact dictionary alternatives.
        if hasattr(self, "orthographic_generator"):
            for alternative in self.orthographic_generator.dictionary_d_t_variants(
                original_norm
            ):
                add(alternative)

                if len(ordered) >= limit:
                    break

        for suggestion in suggestions:
            suggestion_norm = self._normalize_word(suggestion)

            if suggestion_norm == corrected_norm:
                continue

            # "Very quickly found" = close edit distance.
            # This prevents random weak suggestions from appearing.
            if (
                self._word_distance(original_norm, suggestion_norm)
                <= self._max_distance(original_norm) + edit_distance_tolerance
            ):
                add(suggestion_norm)

            if len(ordered) >= limit:
                break

        choices = []

        for word in ordered[:limit]:
            displayed_word = self._match_capitalisation(original_word, word)

            choices.append(
                {
                    "word": displayed_word,
                    "meaning": self.meaning_for(word),
                }
            )

        return choices

    def _ambiguity_choice_limit(
        self,
        original_word: str,
        corrected_word: str,
        default_limit: int,
    ) -> int:
        original_norm = self._normalize_word(original_word)
        corrected_norm = self._normalize_word(corrected_word)
        if original_norm.startswith("l'") or corrected_norm.startswith("l'"):
            return max(default_limit, 4)
        return default_limit

    def correct_text_rich(self, text: str, edit_distance_tolerance: int = 1) -> dict:
        """
        Corrects text while also returning token-level ambiguity data for the frontend.

        Returns:
            {
                "corrected_text": "...",
                "tokens": [...]
            }
        """
        if not text:
            return {"corrected_text": text, "tokens": []}

        tokens: list[dict] = []
        corrected_parts: list[str] = []
        quote_matches = list(self.ENGLISH_QUOTES_PATTERN.finditer(text))

        word_matches = []
        for m in self.WORD_PATTERN.finditer(text):
            overlaps_quote = any(
                m.start() < q.end() and m.end() > q.start() for q in quote_matches
            )
            if not overlaps_quote:
                word_matches.append(m)

        matches = []
        for q in quote_matches:
            matches.append(
                UnifiedMatch(q.start(), q.end(), q.group(0), True, q.group("inner"))
            )
        for w in word_matches:
            matches.append(UnifiedMatch(w.start(), w.end(), w.group(0), False))

        matches.sort(key=lambda x: x.start())

        word_tokens = [
            WordToken(
                # For quoted phrases, expose inner_text so grammar rules
                # (article contractions etc.) see the real first consonant.
                text=(
                    match.inner_text
                    if getattr(match, "is_quote", False)
                    else match.group(0)
                ),
                start=match.start(),
                end=match.end(),
            )
            for match in matches
        ]

        article_rules = getattr(self, "article_phrase_rules", None)
        fused_preposition_rules = getattr(self, "fused_preposition_rules", None)
        previous_surface_word: str | None = None

        def token_choice_state(
            choices: list[dict],
            *,
            force_crucial: bool = False,
        ) -> tuple[bool, bool]:
            normalized_words = []
            has_literal_article_alternative = False
            for choice in choices:
                word = self._normalize_word(choice.get("word", ""))
                if word and word not in normalized_words:
                    normalized_words.append(word)
                if word.startswith(("'il ", "'il-", "'l ", "'l-")):
                    has_literal_article_alternative = True
            multi = len(normalized_words) >= 2
            return multi, bool(force_crucial and multi)

        last_end = 0
        index = 0

        while index < len(matches):
            match = matches[index]
            # Add punctuation/spacing before the word.
            if match.start() > last_end:
                raw_text = text[last_end : match.start()]
                tokens.append(
                    {
                        "type": "text",
                        "text": raw_text,
                    }
                )
                corrected_parts.append(raw_text)

            if getattr(match, "is_quote", False):
                inner_text = match.inner_text
                maltese_suggestion = self.ENGLISH_MAPPINGS.get(
                    inner_text.lower().strip()
                )

                tokens.append(
                    {
                        "type": "english_phrase",
                        "original": match.group(0),
                        "corrected": inner_text,
                        "inner_text": inner_text,
                        "maltese_suggestion": maltese_suggestion,
                    }
                )
                corrected_parts.append(inner_text)
                previous_surface_word = None
                last_end = match.end()
                index += 1
                continue

            original_word = match.group(0)
            original_norm = self._normalize_word(original_word)

            if index + 1 < len(matches) and not getattr(
                matches[index + 1], "is_quote", False
            ):
                next_word_for_phrase = word_tokens[index + 1].text
                next_norm_for_phrase = self._normalize_word(next_word_for_phrase)
                if original_norm == "di" and next_norm_for_phrase.startswith("l-"):
                    corrected_phrase = "dil-" + next_norm_for_phrase.split("-", 1)[1]
                    tokens.append(
                        {
                            "type": "phrase",
                            "original": text[
                                matches[index].start() : matches[index + 1].end()
                            ],
                            "corrected": corrected_phrase,
                            "ambiguous": False,
                            "crucial": False,
                            "choices": [],
                            "name_like": False,
                        }
                    )
                    corrected_parts.append(corrected_phrase)
                    previous_surface_word = self._normalize_word(corrected_phrase)
                    last_end = matches[index + 1].end()
                    index += 2
                    continue

                spaced_prepositions = {
                    "fl", "bl", "sal", "tal", "fil", "bil", "mil", "mill",
                    "mid", "lil", "lill", "ghall", "għall", "mic", "miċ",
                }
                article_rules = getattr(self, "article_phrase_rules", None)
                if original_norm in spaced_prepositions and article_rules:
                    corrected_tail = article_rules._strict_dictionary_tail(
                        next_norm_for_phrase
                    )
                    if corrected_tail is None:
                        corrected_place = self._correct_place_word(
                            next_word_for_phrase
                        )
                        corrected_tail = (
                            self._normalize_word(corrected_place)
                            if corrected_place
                            else None
                        )
                    if original_norm == "mil" and corrected_tail == "bidu":
                        corrected_phrase = self._match_capitalisation(
                            original_word,
                            "mil-bidu",
                        )
                    else:
                        corrected_phrase = (
                            article_rules.preposition_article_form(
                                original_norm,
                                corrected_tail,
                            )
                            if corrected_tail
                            else None
                        )
                else:
                    corrected_phrase = None
                if corrected_phrase:
                    corrected_phrase = self._match_hyphenated_tail_capitalisation(
                        next_word_for_phrase,
                        corrected_phrase,
                    )
                    tokens.append(
                        {
                            "type": "phrase",
                            "original": text[
                                matches[index].start() : matches[index + 1].end()
                            ],
                            "corrected": corrected_phrase,
                            "ambiguous": False,
                            "crucial": False,
                            "choices": [],
                            "name_like": False,
                        }
                    )
                    corrected_parts.append(corrected_phrase)
                    previous_surface_word = self._normalize_word(corrected_phrase)
                    last_end = matches[index + 1].end()
                    index += 2
                    continue

            token_repairs = []
            social_comment_repair = self.SOCIAL_COMMENT_REPAIRS.get(original_norm)
            if social_comment_repair:
                token_repairs = [social_comment_repair]
            elif original_norm not in self.dictionary_set:
                token_repairs = self._lexicalized_form_variants(
                    original_norm
                ) or self._pattern_repair_variants(
                    original_norm
                ) or self._manual_repair_variants(original_norm)
            elif original_norm in {"tajru"}:
                token_repairs = self._pattern_repair_variants(original_norm)
            if token_repairs:
                corrected_word = self._match_capitalisation(
                    original_word,
                    token_repairs[0],
                )
                choices = [
                    {
                        "word": suggestion,
                        "meaning": self.meaning_for(suggestion),
                    }
                    for suggestion in self.suggest(
                        original_word,
                        limit=3,
                        edit_distance_tolerance=edit_distance_tolerance,
                    )
                ]
                is_ambiguous = len(choices) >= 2 and self._normalize_word(
                    choices[0]["word"]
                ) != self._normalize_word(choices[1]["word"])
                tokens.append(
                    {
                        "type": "word",
                        "original": original_word,
                        "corrected": corrected_word,
                        "ambiguous": is_ambiguous,
                        "choices": choices if is_ambiguous else [],
                        "name_like": self._is_initial_capitalized(corrected_word),
                    }
                )
                corrected_parts.append(corrected_word)
                previous_surface_word = self._normalize_word(
                    corrected_word.split()[-1]
                )
                last_end = match.end()
                index += 1
                continue

            # If the next token is a quoted English phrase, handle grammar rules
            # based on the phrase's first consonant (exposed via word_tokens inner_text).
            # We try the real article contraction rule; if it fires we extract just the
            # article prefix (e.g. "il-", "l-", "tal-") and emit it as its own token,
            # consuming the separator so the next iteration handles the english_phrase cleanly.
            if index + 1 < len(matches) and getattr(
                matches[index + 1], "is_quote", False
            ):
                next_quote = matches[index + 1]
                article_prefix = None
                article_choices = []
                is_ambiguous_art = False
                is_crucial_art = False

                # We bypass the dictionary-based article_rules here because English phrases
                # are not in the Maltese dictionary. The rule might falsely match just the
                # standalone preposition (e.g. "tal") and return it without a dash, completely
                # breaking the quote attachment. Instead, we rely entirely on the robust
                # phonological fallback below.

                # Phonological fallback: if the article rule didn't fire (e.g. English phrase
                # not in the Maltese dictionary), but the word is a bare article form or
                # fused preposition, apply the standard rule: consonant-initial → base-,
                # vowel-initial → base without i/l (where applicable), sun-letter → assimilated.
                if article_prefix is None:
                    current_norm = self._normalize_word(original_word)

                    SUN_CONSONANTS = {"d", "n", "r", "s", "t", "x", "z", "ċ", "ż", "c"}
                    BASE_PREPS = {
                        "għa": "għall",
                        "mi": "mill",
                        "sa": "sal",
                        "ma": "mal",
                        "ta": "tal",
                        "bi": "bil",
                        "fi": "fil",
                        "bħa": "bħall",
                        "gha": "ghall",
                        "bha": "bhall",
                    }
                    ALL_FORMS = {"il", "l", "'il", "'l", "\u2019il", "\u2019l"}
                    ALL_FORMS.update(BASE_PREPS.values())
                    for stem in BASE_PREPS.keys():
                        ALL_FORMS.update(stem + c for c in SUN_CONSONANTS)
                    ALL_FORMS.update("i" + c for c in SUN_CONSONANTS)

                    if current_norm in ALL_FORMS:
                        MALTESE_VOWELS = set("aeiouàèìòùáéíóúâêîôû")
                        inner = next_quote.inner_text.strip()
                        first_char = inner[0].lower() if inner else ""

                        prefix_char = ""
                        if current_norm.startswith("'") or current_norm.startswith(
                            "\u2019"
                        ):
                            prefix_char = current_norm[0]
                            current_norm_check = current_norm[1:]
                        else:
                            current_norm_check = current_norm

                        word_stem = ""
                        for s in BASE_PREPS:
                            if current_norm_check.startswith(s):
                                word_stem = s
                                break

                        MALTESE_STEMS = {"gha": "għa", "bha": "bħa"}
                        MALTESE_PREPS = {"gha": "għall", "bha": "bħall"}

                        true_stem = MALTESE_STEMS.get(word_stem, word_stem)
                        true_first_char = "ċ" if first_char == "c" else first_char

                        if first_char in MALTESE_VOWELS:
                            base_prep = MALTESE_PREPS.get(
                                word_stem, BASE_PREPS.get(word_stem)
                            )
                            expected = base_prep if word_stem else "l"
                        elif first_char in SUN_CONSONANTS:
                            expected = (
                                true_stem + true_first_char
                                if true_stem
                                else "i" + true_first_char
                            )
                        else:
                            base_prep = MALTESE_PREPS.get(
                                word_stem, BASE_PREPS.get(word_stem)
                            )
                            expected = base_prep if word_stem else "il"

                        article_prefix = (
                            self._match_capitalisation(
                                original_word, prefix_char + expected
                            )
                            + "-"
                        )
                        is_crucial_art = True
                        article_choices = [
                            {"word": article_prefix, "meaning": ""},
                        ]
                        if not word_stem and expected == "il":
                            article_choices.append(
                                {
                                    "word": self._match_capitalisation(
                                        original_word, prefix_char + "'il"
                                    )
                                    + "-",
                                    "meaning": "",
                                }
                            )

                corrected_word_out = (
                    article_prefix if article_prefix is not None else original_word
                )
                tokens.append(
                    {
                        "type": "word",
                        "original": original_word,
                        "corrected": corrected_word_out,
                        "ambiguous": is_ambiguous_art,
                        "crucial": is_crucial_art,
                        "choices": article_choices,
                        "name_like": False,
                    }
                )
                corrected_parts.append(corrected_word_out)
                # If the article rule fired, consume the separator (space/dash) so the
                # quote follows immediately. Otherwise preserve it as raw text.
                last_end = (
                    next_quote.start() if article_prefix is not None else match.end()
                )
                previous_surface_word = self._normalize_word(original_word)
                index += 1
                continue

            sentence_initial = self._is_sentence_initial_position(text, match.start())

            capitalized_place_phrase = self._match_capitalized_place_phrase(
                text,
                word_tokens,
                matches,
                index,
            )
            if capitalized_place_phrase is not None:
                original_phrase, corrected_phrase, consumed = capitalized_place_phrase
                tokens.append(
                    {
                        "type": "phrase",
                        "original": original_phrase,
                        "corrected": corrected_phrase,
                        "ambiguous": False,
                        "choices": [],
                        "name_like": True,
                    }
                )
                corrected_parts.append(corrected_phrase)
                previous_surface_word = self._normalize_word(
                    corrected_phrase.split()[-1]
                )
                last_end = matches[index + consumed - 1].end()
                index += consumed
                continue

            # Catch split prefix forms before the generic word fixer runs.
            if index + 1 < len(matches) and not getattr(
                matches[index + 1], "is_quote", False
            ):
                next_word = word_tokens[index + 1].text
                current_norm = self._normalize_word(original_word)
                next_norm = self._normalize_word(next_word)

                if current_norm == "ma" and next_norm in {"lahhar", "l-aħħar"}:
                    corrected_phrase = self._match_capitalisation(
                        original_word,
                        "ma l-aħħar",
                    )
                    tokens.append(
                        {
                            "type": "phrase",
                            "original": text[
                                matches[index].start() : matches[index + 1].end()
                            ],
                            "corrected": corrected_phrase,
                            "ambiguous": False,
                            "crucial": False,
                            "choices": [],
                            "name_like": False,
                        }
                    )
                    corrected_parts.append(corrected_phrase)
                    previous_surface_word = self._normalize_word("l-aħħar")
                    last_end = matches[index + 1].end()
                    index += 2
                    continue

                if (
                    current_norm == "ta"
                    and next_norm in {"l", "il"}
                    and index + 2 < len(matches)
                    and not getattr(matches[index + 2], "is_quote", False)
                ):
                    article_noun = self.correct_word(word_tokens[index + 2].text)
                    article_noun_norm = self._normalize_word(article_noun)
                    if article_noun_norm:
                        corrected_phrase = f"tal-{article_noun_norm}"
                        tokens.append(
                            {
                                "type": "phrase",
                                "original": text[
                                    matches[index].start() : matches[index + 2].end()
                                ],
                                "corrected": corrected_phrase,
                                "ambiguous": False,
                                "crucial": False,
                                "choices": [],
                                "name_like": False,
                            }
                        )
                        corrected_parts.append(corrected_phrase)
                        previous_surface_word = self._normalize_word(corrected_phrase)
                        last_end = matches[index + 2].end()
                        index += 3
                        continue

                if current_norm == "f" and index + 1 < len(matches):
                    between = text[matches[index].end() : matches[index + 1].start()]
                    if "4" in between and next_norm == "snin":
                        corrected_phrase = "f'4 snin"
                        tokens.append(
                            {
                                "type": "phrase",
                                "original": text[
                                    matches[index].start() : matches[index + 1].end()
                                ],
                                "corrected": corrected_phrase,
                                "ambiguous": False,
                                "crucial": False,
                                "choices": [],
                                "name_like": False,
                            }
                        )
                        corrected_parts.append(corrected_phrase)
                        previous_surface_word = self._normalize_word("snin")
                        last_end = matches[index + 1].end()
                        index += 2
                        continue

                if current_norm == "di" and next_norm.startswith("l-"):
                    corrected_phrase = "dil-" + next_norm.split("-", 1)[1]
                    tokens.append(
                        {
                            "type": "phrase",
                            "original": text[
                                matches[index].start() : matches[index + 1].end()
                            ],
                            "corrected": corrected_phrase,
                            "ambiguous": False,
                            "crucial": False,
                            "choices": [],
                            "name_like": False,
                        }
                    )
                    corrected_parts.append(corrected_phrase)
                    previous_surface_word = self._normalize_word(corrected_phrase)
                    last_end = matches[index + 1].end()
                    index += 2
                    continue

                if current_norm == "in" and next_norm == "nies":
                    corrected_phrase = "in-nies"
                    tokens.append(
                        {
                            "type": "phrase",
                            "original": text[
                                matches[index].start() : matches[index + 1].end()
                            ],
                            "corrected": corrected_phrase,
                            "ambiguous": False,
                            "crucial": False,
                            "choices": [],
                            "name_like": False,
                        }
                    )
                    corrected_parts.append(corrected_phrase)
                    previous_surface_word = self._normalize_word(corrected_phrase)
                    last_end = matches[index + 1].end()
                    index += 2
                    continue

                if current_norm == "mil" and next_norm == "bidu":
                    corrected_phrase = self._match_capitalisation(
                        original_word,
                        "mil-bidu",
                    )
                    tokens.append(
                        {
                            "type": "phrase",
                            "original": text[
                                matches[index].start() : matches[index + 1].end()
                            ],
                            "corrected": corrected_phrase,
                            "ambiguous": False,
                            "crucial": False,
                            "choices": [],
                            "name_like": False,
                        }
                    )
                    corrected_parts.append(corrected_phrase)
                    previous_surface_word = self._normalize_word(corrected_phrase)
                    last_end = matches[index + 1].end()
                    index += 2
                    continue

                if current_norm == "xi" and self._article_like_token(next_norm):
                    corrected_article = self._correct_inline_article_word(
                        next_word,
                        previous=word_tokens[index - 1].text if index > 0 else None,
                    )
                    if corrected_article is not None:
                        corrected_phrase = f"xi {corrected_article}"
                        tokens.append(
                            {
                                "type": "phrase",
                                "original": text[
                                    matches[index].start() : matches[index + 1].end()
                                ],
                                "corrected": corrected_phrase,
                                "ambiguous": False,
                                "crucial": False,
                                "choices": [],
                                "name_like": False,
                            }
                        )
                        corrected_parts.append(corrected_phrase)
                        previous_surface_word = self._normalize_word(
                            corrected_phrase.split()[-1]
                        )
                        last_end = matches[index + 1].end()
                        index += 2
                        continue

                if current_norm == "kemm" and self._article_like_token(next_norm):
                    corrected_noun = None
                    consumed = 2

                    if next_norm in {"il", "l"} and index + 2 < len(matches):
                        corrected_noun = self._normalize_word(
                            self.correct_word(word_tokens[index + 2].text)
                        )
                        consumed = 3
                    elif "-" in next_norm:
                        corrected_noun = next_norm.split("-", 1)[1]
                        if article_rules is not None:
                            corrected_noun = (
                                article_rules._strict_dictionary_tail(corrected_noun)
                                or corrected_noun
                            )

                    if corrected_noun:
                        corrected_phrase = f"kemm-il {corrected_noun}"
                        tokens.append(
                            {
                                "type": "phrase",
                                "original": text[
                                    matches[index].start() : matches[index + consumed - 1].end()
                                ],
                                "corrected": corrected_phrase,
                                "ambiguous": False,
                                "crucial": False,
                                "choices": [
                                    {
                                        "word": "kemm-il",
                                        "meaning": self.meaning_for("kemm-il"),
                                    }
                                ],
                                "name_like": False,
                            }
                        )
                        corrected_parts.append(corrected_phrase)
                        previous_surface_word = self._normalize_word(
                            corrected_phrase.split()[-1]
                        )
                        last_end = matches[index + consumed - 1].end()
                        index += consumed
                        continue

                if (
                    current_norm == "m"
                    and next_norm not in {"l", "il"}
                    and not next_norm.startswith(("l-", "il-"))
                ):
                    corrected_next = self.correct_word(next_word)
                    if self._is_verb_tagged_word(corrected_next):
                        corrected_phrase = f"{current_norm}'{corrected_next}"
                        phrase_choices = [
                            {
                                "word": corrected_phrase,
                                "meaning": self.meaning_for(corrected_next),
                            },
                            {
                                "word": f"{current_norm} {corrected_next}",
                                "meaning": self.meaning_for(corrected_next),
                            },
                        ]
                        is_ambiguous, is_crucial = token_choice_state(
                            phrase_choices,
                            force_crucial=True,
                        )
                        tokens.append(
                            {
                                "type": "phrase",
                                "original": text[
                                    matches[index].start() : matches[index + 1].end()
                                ],
                                "corrected": corrected_phrase,
                                "ambiguous": is_ambiguous,
                                "crucial": is_crucial,
                                "choices": phrase_choices,
                            }
                        )
                        corrected_parts.append(corrected_phrase)
                        previous_surface_word = self._normalize_word(
                            corrected_phrase.split()[-1]
                        )
                        last_end = matches[index + 1].end()
                        index += 2
                        continue

                if current_norm in {"ma", "ma'"} and not (
                    self._article_like_token(next_norm)
                    and ("-" in next_norm or index + 2 < len(matches))
                ):
                    negative_imperative = self._negative_imperative_form(
                        next_word
                    )
                    corrected_next = self.correct_word(next_word)
                    if negative_imperative is None:
                        negative_imperative = self._negative_imperative_form(
                            corrected_next
                        )
                    if negative_imperative:
                        display_ma = self._match_capitalisation(
                            original_word,
                            "ma",
                        )
                        corrected_phrase = f"{display_ma} {negative_imperative}"
                        tokens.append(
                            {
                                "type": "phrase",
                                "original": text[
                                    matches[index].start() : matches[index + 1].end()
                                ],
                                "corrected": corrected_phrase,
                                "ambiguous": False,
                                "crucial": True,
                                "choices": [
                                    {
                                        "word": corrected_phrase,
                                        "meaning": self.meaning_for(
                                            negative_imperative
                                        ),
                                    }
                                ],
                            }
                        )
                        corrected_parts.append(corrected_phrase)
                        previous_surface_word = self._normalize_word(
                            negative_imperative
                        )
                        last_end = matches[index + 1].end()
                        index += 2
                        continue

                    is_verb = self._is_verb_tagged_word(corrected_next)

                    target_ma = "ma" if is_verb else "ma'"

                    if (
                        current_norm != target_ma
                        or self._normalize_word(corrected_next) != next_norm
                        or (
                            is_verb
                            and self._contract_negative_ma(
                                f"ma {corrected_next}"
                            )
                            != f"ma {corrected_next}"
                        )
                    ):
                        display_ma = self._match_capitalisation(
                            original_word, target_ma
                        )
                        corrected_phrase = f"{display_ma} {corrected_next}"
                        if is_verb:
                            corrected_phrase = self._contract_negative_ma(
                                corrected_phrase
                            )
                        phrase_choices = [
                            {
                                "word": corrected_phrase,
                                "meaning": self.meaning_for(corrected_next),
                            },
                        ]

                        if current_norm != target_ma:
                            other_ma = "ma'" if is_verb else "ma"
                            display_other = self._match_capitalisation(
                                original_word, other_ma
                            )
                            phrase_choices.append(
                                {
                                    "word": self._contract_negative_ma(
                                        f"{display_other} {corrected_next}"
                                    ),
                                    "meaning": self.meaning_for(corrected_next),
                                }
                            )

                        is_ambiguous, is_crucial = token_choice_state(
                            phrase_choices,
                            force_crucial=True if current_norm != target_ma else False,
                        )
                        tokens.append(
                            {
                                "type": "phrase",
                                "original": text[
                                    matches[index].start() : matches[index + 1].end()
                                ],
                                "corrected": corrected_phrase,
                                "ambiguous": is_ambiguous,
                                "crucial": is_crucial,
                                "choices": phrase_choices,
                            }
                        )
                        corrected_parts.append(corrected_phrase)
                        previous_surface_word = self._normalize_word(
                            corrected_phrase.split()[-1]
                        )
                        last_end = matches[index + 1].end()
                        index += 2
                        continue

                if current_norm in {"min", "minn"} and not (
                    next_norm.split("-", 1)[0]
                    in {
                        "il", "l", "ic", "iċ", "id", "in", "ir", "is",
                        "it", "ix", "iz", "iż",
                    }
                    and ("-" in next_norm or index + 2 < len(matches))
                ):
                    corrected_next = self.correct_word(next_word)

                    # ADD MORE WORDS HERE manually as needed
                    SPECIAL_MINN_WORDS = {"xiex", "meta"}
                    HEMM_HAWN_WORDS = {"hemm", "hawn", "hemmhekk", "hawnhekk"}

                    is_name = self._is_initial_capitalized(corrected_next)
                    next_norm_corrected = self._normalize_word(corrected_next)

                    if next_norm_corrected in HEMM_HAWN_WORDS:
                        remainder = text[matches[index].start() :]
                        match_punct = re.search(r"[.?!]", remainder)
                        is_question = match_punct and match_punct.group(0) == "?"

                        if is_question:
                            target_min = current_norm  # Both are valid, accept what the user wrote
                        else:
                            target_min = "minn"  # Only "minn" is valid for statements
                    else:
                        target_min = (
                            "minn"
                            if (is_name or next_norm_corrected in SPECIAL_MINN_WORDS)
                            else "min"
                        )

                    if current_norm != target_min or next_norm_corrected != next_norm:
                        display_min = self._match_capitalisation(
                            original_word, target_min
                        )
                        corrected_phrase = f"{display_min} {corrected_next}"
                        phrase_choices = [
                            {
                                "word": corrected_phrase,
                                "meaning": self.meaning_for(corrected_next),
                            },
                        ]

                        if current_norm != target_min:
                            other_min = "min" if target_min == "minn" else "minn"
                            display_other = self._match_capitalisation(
                                original_word, other_min
                            )
                            phrase_choices.append(
                                {
                                    "word": f"{display_other} {corrected_next}",
                                    "meaning": self.meaning_for(corrected_next),
                                }
                            )

                        is_ambiguous, is_crucial = token_choice_state(
                            phrase_choices,
                            force_crucial=True if current_norm != target_min else False,
                        )
                        tokens.append(
                            {
                                "type": "phrase",
                                "original": text[
                                    matches[index].start() : matches[index + 1].end()
                                ],
                                "corrected": corrected_phrase,
                                "ambiguous": is_ambiguous,
                                "crucial": is_crucial,
                                "choices": phrase_choices,
                            }
                        )
                        corrected_parts.append(corrected_phrase)
                        previous_surface_word = self._normalize_word(
                            corrected_phrase.split()[-1]
                        )
                        last_end = matches[index + 1].end()
                        index += 2
                        continue

                if current_norm in {"ghar", "għar"}:
                    corrected_next = self.correct_word(next_word)
                    corrected_next_norm = self._normalize_word(corrected_next)
                    if corrected_next_norm.startswith("r") and self._is_probable_noun(
                        corrected_next_norm
                    ):
                        corrected_phrase = f"għar-{corrected_next_norm}"
                        tokens.append(
                            {
                                "type": "phrase",
                                "original": text[
                                    matches[index].start() : matches[index + 1].end()
                                ],
                                "corrected": corrected_phrase,
                                "ambiguous": False,
                                "crucial": False,
                                "choices": [],
                                "name_like": False,
                            }
                        )
                        corrected_parts.append(corrected_phrase)
                        previous_surface_word = corrected_next_norm
                        last_end = matches[index + 1].end()
                        index += 2
                        continue

            if article_rules is not None and not (
                index + 1 < len(matches)
                and getattr(matches[index + 1], "is_quote", False)
            ):
                article_match = article_rules.match_preposition_article_contraction(
                    word_tokens,
                    index,
                )

                if article_match is not None:
                    consumed = article_match.end - article_match.start
                    original_article_tail = word_tokens[
                        index + consumed - 1
                    ].text.split("-", 1)[-1]
                    article_corrected = (
                        self._match_hyphenated_tail_capitalisation(
                            original_article_tail,
                            article_match.corrected,
                        )
                    )
                    original_phrase = text[
                        matches[index].start() : matches[index + consumed - 1].end()
                    ]
                    is_ambiguous, is_crucial = token_choice_state(
                        article_match.choices,
                        force_crucial=True,
                    )

                    tokens.append(
                        {
                            "type": "phrase",
                            "original": original_phrase,
                            "corrected": article_corrected,
                            "ambiguous": is_ambiguous,
                            "crucial": is_crucial,
                            "choices": article_match.choices,
                        }
                    )

                    corrected_parts.append(article_corrected)
                    previous_surface_word = self._normalize_word(
                        article_corrected.split()[-1]
                    )
                    last_end = matches[index + consumed - 1].end()
                    index += consumed
                    continue

                article_match = article_rules.match_split_article(
                    word_tokens,
                    index,
                )

                if article_match is not None:
                    original_phrase = text[
                        matches[index].start() : matches[index + 1].end()
                    ]
                    is_ambiguous, is_crucial = token_choice_state(
                        article_match.choices,
                        force_crucial=True,
                    )

                    tokens.append(
                        {
                            "type": "phrase",
                            "original": original_phrase,
                            "corrected": article_match.corrected,
                            "ambiguous": is_ambiguous,
                            "crucial": is_crucial,
                            "choices": article_match.choices,
                        }
                    )

                    corrected_parts.append(article_match.corrected)
                    previous_surface_word = self._normalize_word(
                        article_match.corrected.split()[-1]
                    )
                    last_end = matches[index + 1].end()
                    index += 2
                    continue

            # Catch split prefix forms before the generic word fixer runs.
            if index + 1 < len(matches) and not getattr(
                matches[index + 1], "is_quote", False
            ):
                next_word = word_tokens[index + 1].text
                current_norm = self._normalize_word(original_word)
                next_norm = self._normalize_word(next_word)

                if current_norm == "l" and self._starts_vowel_gh_or_h(next_norm):
                    corrected_next = self.correct_word(next_word)
                    corrected_phrase = f"l-{corrected_next}"
                    phrase_choices = [
                        {
                            "word": corrected_phrase,
                            "meaning": self.meaning_for(corrected_next),
                        },
                    ]
                    is_ambiguous, is_crucial = token_choice_state(
                        phrase_choices,
                        force_crucial=True,
                    )
                    tokens.append(
                        {
                            "type": "phrase",
                            "original": text[
                                matches[index].start() : matches[index + 1].end()
                            ],
                            "corrected": corrected_phrase,
                            "ambiguous": is_ambiguous,
                            "crucial": is_crucial,
                            "choices": phrase_choices,
                        }
                    )
                    corrected_parts.append(corrected_phrase)
                    previous_surface_word = self._normalize_word(
                        corrected_phrase.split()[-1]
                    )
                    last_end = matches[index + 1].end()
                    index += 2
                    continue

                if (
                    current_norm in {"f", "b"}
                    and next_norm not in {"l", "il"}
                    and not next_norm.startswith(("l-", "il-"))
                ):
                    corrected_next = self.correct_word(next_word)
                    if self._is_probable_noun(corrected_next):
                        corrected_phrase = f"{current_norm}'{corrected_next}"
                        phrase_choices = [
                            {
                                "word": corrected_phrase,
                                "meaning": self.meaning_for(corrected_next),
                            },
                            {
                                "word": f"{current_norm} {corrected_next}",
                                "meaning": self.meaning_for(corrected_next),
                            },
                        ]
                        is_ambiguous, is_crucial = token_choice_state(
                            phrase_choices,
                            force_crucial=True,
                        )
                        tokens.append(
                            {
                                "type": "phrase",
                                "original": text[
                                    matches[index].start() : matches[index + 1].end()
                                ],
                                "corrected": corrected_phrase,
                                "ambiguous": is_ambiguous,
                                "crucial": is_crucial,
                                "choices": phrase_choices,
                            }
                        )
                        corrected_parts.append(corrected_phrase)
                        previous_surface_word = self._normalize_word(
                            corrected_phrase.split()[-1]
                        )
                        last_end = matches[index + 1].end()
                        index += 2
                        continue

                # Special case: fi xħin → fi x'ħin
                if current_norm == "fi" and next_norm in {"xhin", "xħin"}:
                    corrected_phrase = "fi x'ħin"
                    phrase_choices = [
                        {"word": corrected_phrase, "meaning": self.meaning_for("x'ħin")}
                    ]
                    tokens.append(
                        {
                            "type": "phrase",
                            "original": text[
                                matches[index].start() : matches[index + 1].end()
                            ],
                            "corrected": corrected_phrase,
                            "ambiguous": False,
                            "crucial": False,
                            "choices": phrase_choices,
                        }
                    )
                    corrected_parts.append(corrected_phrase)
                    previous_surface_word = self._normalize_word("x'ħin")
                    last_end = matches[index + 1].end()
                    index += 2
                    continue

                if current_norm in {"xi", "bi", "fi"}:
                    corrected_next = next_word
                    if fused_preposition_rules is not None:
                        remainder_candidates = (
                            fused_preposition_rules.strict_remainder_candidates(
                                next_word
                            )
                        )
                        if remainder_candidates:
                            corrected_next = remainder_candidates[0]
                    else:
                        corrected_next = self.correct_word(next_word)

                    if self._normalize_word(corrected_next) != next_norm:
                        corrected_phrase = f"{current_norm} {corrected_next}"
                        phrase_choices = [
                            {
                                "word": corrected_phrase,
                                "meaning": self.meaning_for(corrected_next),
                            },
                        ]
                        is_ambiguous, is_crucial = token_choice_state(
                            phrase_choices,
                            force_crucial=False,
                        )
                        tokens.append(
                            {
                                "type": "phrase",
                                "original": text[
                                    matches[index].start() : matches[index + 1].end()
                                ],
                                "corrected": corrected_phrase,
                                "ambiguous": is_ambiguous,
                                "crucial": is_crucial,
                                "choices": phrase_choices,
                            }
                        )
                        corrected_parts.append(corrected_phrase)
                        previous_surface_word = self._normalize_word(
                            corrected_phrase.split()[-1]
                        )
                        last_end = matches[index + 1].end()
                        index += 2
                        continue

                if (
                    current_norm == "x"
                    and next_norm not in {"l", "il"}
                    and not next_norm.startswith(("l-", "il-"))
                ):
                    corrected_next = self.correct_word(next_word)

                    is_target_pos = (
                        self._is_verb_tagged_word(corrected_next)
                        or self._is_pronoun_tagged_word(corrected_next)
                        or self._is_adverb_tagged_word(corrected_next)
                        or self._is_noun_tagged_word(corrected_next)
                    )

                    if is_target_pos:
                        primary_form = self._xi_form_for_word(corrected_next)

                        # If preceded by a verb or pronoun, 'xi' takes priority
                        if previous_surface_word and self._is_verb_or_pronoun_tagged(
                            previous_surface_word
                        ):
                            primary_form = "xi"

                        xi_phrase = f"xi {corrected_next}"
                        x_apos_phrase = f"x'{corrected_next}"

                        if primary_form == "xi":
                            corrected_phrase = xi_phrase
                            alt_phrase = x_apos_phrase
                        else:
                            corrected_phrase = x_apos_phrase
                            alt_phrase = xi_phrase

                        phrase_choices = [
                            {
                                "word": corrected_phrase,
                                "meaning": self.meaning_for(corrected_next),
                            },
                            {
                                "word": alt_phrase,
                                "meaning": self.meaning_for(corrected_next),
                            },
                        ]

                        is_ambiguous, is_crucial = token_choice_state(
                            phrase_choices,
                            force_crucial=True,
                        )
                        tokens.append(
                            {
                                "type": "phrase",
                                "original": text[
                                    matches[index].start() : matches[index + 1].end()
                                ],
                                "corrected": corrected_phrase,
                                "ambiguous": is_ambiguous,
                                "crucial": is_crucial,
                                "choices": phrase_choices,
                            }
                        )
                        corrected_parts.append(corrected_phrase)
                        previous_surface_word = self._normalize_word(
                            corrected_phrase.split()[-1]
                        )
                        last_end = matches[index + 1].end()
                        index += 2
                        continue

                if (
                    current_norm == "m"
                    and next_norm not in {"l", "il"}
                    and not next_norm.startswith(("l-", "il-"))
                ):
                    corrected_next = self.correct_word(next_word)
                    if self._is_verb_tagged_word(corrected_next):
                        corrected_phrase = f"{current_norm}'{corrected_next}"
                        phrase_choices = [
                            {
                                "word": corrected_phrase,
                                "meaning": self.meaning_for(corrected_next),
                            },
                            {
                                "word": f"{current_norm} {corrected_next}",
                                "meaning": self.meaning_for(corrected_next),
                            },
                        ]
                        is_ambiguous, is_crucial = token_choice_state(
                            phrase_choices,
                            force_crucial=True,
                        )
                        tokens.append(
                            {
                                "type": "phrase",
                                "original": text[
                                    matches[index].start() : matches[index + 1].end()
                                ],
                                "corrected": corrected_phrase,
                                "ambiguous": is_ambiguous,
                                "crucial": is_crucial,
                                "choices": phrase_choices,
                            }
                        )
                        corrected_parts.append(corrected_phrase)
                        previous_surface_word = self._normalize_word(
                            corrected_phrase.split()[-1]
                        )
                        last_end = matches[index + 1].end()
                        index += 2
                        continue

            if article_rules is not None:
                compact_xi_article = self._expand_compact_xi_article(original_word)
                if compact_xi_article is not None:
                    tokens.append(
                        {
                            "type": "phrase",
                            "original": original_word,
                            "corrected": compact_xi_article,
                            "ambiguous": False,
                            "crucial": False,
                            "choices": [],
                            "name_like": False,
                        }
                    )

                    corrected_parts.append(compact_xi_article)
                    previous_surface_word = self._normalize_word(
                        compact_xi_article.split()[-1]
                    )
                    last_end = match.end()
                    index += 1
                    continue

                article_match = article_rules.match_compact_preposition_article(
                    original_word,
                )

                if article_match is not None:
                    compact_corrected = self._match_hyphenated_tail_capitalisation(
                        original_word.split("-", 1)[-1],
                        article_match.corrected,
                    )
                    is_ambiguous, is_crucial = token_choice_state(
                        article_match.choices,
                        force_crucial=True,
                    )
                    tokens.append(
                        {
                            "type": "phrase",
                            "original": original_word,
                            "corrected": compact_corrected,
                            "ambiguous": is_ambiguous,
                            "crucial": is_crucial,
                            "choices": article_match.choices,
                        }
                    )

                    corrected_parts.append(compact_corrected)
                    previous_surface_word = self._normalize_word(
                        compact_corrected.split()[-1]
                    )
                    last_end = match.end()
                    index += 1
                    continue

                previous_word = word_tokens[index - 1].text if index > 0 else None
                article_match = article_rules.match_hyphenated_article_after(
                    original_word,
                    previous=previous_word,
                )

                if article_match is not None:
                    hyphenated_corrected = (
                        self._match_hyphenated_tail_capitalisation(
                            original_word.split("-", 1)[-1],
                            article_match.corrected,
                        )
                    )
                    is_ambiguous, is_crucial = token_choice_state(
                        article_match.choices,
                        force_crucial=True,
                    )
                    tokens.append(
                        {
                            "type": "phrase",
                            "original": original_word,
                            "corrected": hyphenated_corrected,
                            "ambiguous": is_ambiguous,
                            "crucial": is_crucial,
                            "choices": article_match.choices,
                        }
                    )

                    corrected_parts.append(hyphenated_corrected)
                    previous_surface_word = self._normalize_word(
                        hyphenated_corrected.split()[-1]
                    )
                    last_end = match.end()
                    index += 1
                    continue

                if "-" in self._normalize_word(original_word):
                    corrected_inline_article = self._correct_inline_article_word(
                        original_word,
                        previous=previous_word,
                    )
                    if corrected_inline_article is not None:
                        tokens.append(
                            {
                                "type": "word",
                                "original": original_word,
                                "corrected": corrected_inline_article,
                                "ambiguous": False,
                                "choices": [],
                                "name_like": False,
                            }
                        )
                        corrected_parts.append(corrected_inline_article)
                        previous_surface_word = self._normalize_word(
                            corrected_inline_article
                        )
                        last_end = match.end()
                        index += 1
                        continue

            if fused_preposition_rules is not None:
                fused_match = fused_preposition_rules.match(original_word)

                if fused_match is not None:
                    fused_corrected = self._contract_negative_ma(
                        fused_match.corrected
                    )
                    fused_choices = [
                        {
                            **choice,
                            "word": self._contract_negative_ma(
                                choice.get("word", "")
                            ),
                        }
                        for choice in fused_match.choices
                    ]
                    is_ambiguous, is_crucial = token_choice_state(
                        fused_choices,
                        force_crucial=True,
                    )
                    tokens.append(
                        {
                            "type": "word",
                            "original": original_word,
                            "corrected": fused_corrected,
                            "ambiguous": is_ambiguous,
                            "crucial": is_crucial,
                            "choices": fused_choices,
                        }
                    )

                    corrected_parts.append(fused_corrected)
                    previous_surface_word = self._normalize_word(
                        fused_corrected.split()[-1]
                    )
                    last_end = match.end()
                    index += 1
                    continue

            if self._is_initial_capitalized(original_word) and not sentence_initial:
                corrected_place_word = self._correct_place_word(original_word)
                if corrected_place_word:
                    final_capitalized = corrected_place_word
                else:
                    strict_capitalized = self._try_exact_variants(
                        original_word,
                        self._strict_lookup_variants(original_norm),
                    )
                    final_capitalized = strict_capitalized
                    if final_capitalized is None:
                        diacritic_candidate = self.correct_word(original_word)
                        if (
                            self._normalize_word(diacritic_candidate)
                            in self.dictionary_set
                            and self._strip_maltese_shortcuts(original_norm)
                            == self._strip_maltese_shortcuts(diacritic_candidate)
                        ):
                            final_capitalized = diacritic_candidate
                    if final_capitalized is None:
                        final_capitalized = original_word
                tokens.append(
                    {
                        "type": "word",
                        "original": original_word,
                        "corrected": final_capitalized,
                        "ambiguous": False,
                        "choices": [],
                        "name_like": True,
                    }
                )
                corrected_parts.append(final_capitalized)
                previous_surface_word = self._normalize_word(final_capitalized)
                last_end = match.end()
                index += 1
                continue

            corrected_word = (
                self._correct_sentence_initial_capitalized(original_word)
                if self._is_initial_capitalized(original_word) and sentence_initial
                else self.correct_word(original_word)
            )
            corrected_word = self._contract_negative_ma(corrected_word)
            preserved_empathetic_i = False
            if (
                previous_surface_word
                and self._word_ends_with_consonant(previous_surface_word)
                and self._has_empathetic_i_shape(original_word)
            ):
                corrected_word = self._match_capitalisation(
                    original_word,
                    self._normalize_word(original_word),
                )
                preserved_empathetic_i = True
            choices = self.ambiguity_choices(
                original_word,
                corrected_word,
                limit=3,
                edit_distance_tolerance=edit_distance_tolerance,
            )
            if (
                sentence_initial
                and self._is_initial_capitalized(original_word)
                and self._normalize_word(corrected_word)
                == self._normalize_word(original_word)
            ):
                choices = []
            preferred_apostrophe = self._preferred_apostrophe_choice(choices)
            if preferred_apostrophe and self._normalize_word(
                preferred_apostrophe
            ) != self._normalize_word(corrected_word):
                corrected_word = preferred_apostrophe
                choices = self.ambiguity_choices(
                    original_word,
                    corrected_word,
                    limit=3,
                    edit_distance_tolerance=edit_distance_tolerance,
                )

            surface_word = self._apply_empathetic_i(
                previous_surface_word, corrected_word
            )

            if preserved_empathetic_i or self._normalize_word(
                surface_word
            ) != self._normalize_word(corrected_word):
                choices = []

            is_ambiguous = len(choices) >= 2 and self._normalize_word(
                choices[0]["word"]
            ) != self._normalize_word(choices[1]["word"])

            tokens.append(
                {
                    "type": "word",
                    "original": original_word,
                    "corrected": surface_word,
                    "ambiguous": is_ambiguous,
                    "choices": choices if is_ambiguous else [],
                    "name_like": self._is_initial_capitalized(surface_word),
                }
            )

            corrected_parts.append(surface_word)
            previous_surface_word = self._normalize_word(surface_word)
            last_end = match.end()
            index += 1

        # Add trailing punctuation/spacing.
        if last_end < len(text):
            raw_text = text[last_end:]
            tokens.append(
                {
                    "type": "text",
                    "text": raw_text,
                }
            )
            corrected_parts.append(raw_text)

        corrected_text = "".join(corrected_parts)
        self._add_country_translation_choices(tokens)

        return {
            "corrected_text": corrected_text,
            "tokens": tokens,
        }


# -----------------------------------------------------------------------------
# Flask app
# -----------------------------------------------------------------------------

app = Flask(__name__)

spellchecker = UniversalMalteseSpellchecker(dictionary_files=DICTIONARY_FILES)
meaning_index = MeaningIndex([*DICTIONARY_FILES, EU_COUNTRIES_DIC])
article_phrase_rules = MalteseArticlePhraseRules(
    dictionary_files=DICTIONARY_FILES,
    meaning_index=meaning_index,
    normalizer=spellchecker._normalize_word,
)
article_phrase_rules.spellchecker = spellchecker
spellchecker.article_phrase_rules = article_phrase_rules

orthographic_generator = MalteseOrthographicGenerator(spellchecker=spellchecker)
spellchecker.orthographic_generator = orthographic_generator

doubled_letter_generator = MalteseDoubledLetterGenerator(spellchecker=spellchecker)
spellchecker.doubled_letter_generator = doubled_letter_generator

suffix_generator = MalteseSuffixGenerator(
    spellchecker=spellchecker,
    verbs_file=[
        BASE_DIR / "finaldics/verbmt_semitic.dic",
        BASE_DIR / "finaldics/verbmt_nonsemitic.dic",
    ],
)

spellchecker.suffix_generator = suffix_generator

fused_preposition_rules = MalteseFusedPrepositionRules(
    spellchecker=spellchecker,
    article_rules=article_phrase_rules,
    meaning_index=meaning_index,
)
spellchecker.fused_preposition_rules = fused_preposition_rules


ENABLE_DEV_TOOLS = False
SHOW_STATUS_MESSAGES = False

@app.get("/")
def home():
    html_path = BASE_DIR / "index.html"
    try:
        with open(html_path, encoding="utf-8") as f:
            html = f.read()
    except FileNotFoundError:
        return "index.html not found", 404

    # Inject configuration variables into the HTML
    html = html.replace(
        '"REPLACE_ME_ENABLE_DEV_TOOLS" === "True"',
        "true" if ENABLE_DEV_TOOLS else "false"
    )
    html = html.replace(
        '"REPLACE_ME_SHOW_STATUS_MESSAGES" === "True"',
        "true" if SHOW_STATUS_MESSAGES else "false"
    )
    return html


@app.get("/devtoy.js")
def devtoy_js():
    return send_from_directory(BASE_DIR, "devtoy.js")


@app.get("/assets/<path:filename>")
def assets(filename):
    return send_from_directory(BASE_DIR / "assets", filename)


@app.get("/devtoy-assets/<filename>")
def devtoy_assets(filename):
    return send_from_directory(BASE_DIR / "assets" / "devtoys", filename)


@app.get("/health")
def health():
    suffix_info = {}

    if hasattr(spellchecker, "suffix_generator"):
        suffix_info = {
            "suffix_parser_mode": "ending-first",
            "suffix_verb_records": len(
                spellchecker.suffix_generator.verb_index.records
            ),
        }

    return jsonify(
        {
            "ok": True,
            "dictionary_words": len(spellchecker.dictionary),
            "paradigms": len(spellchecker.paradigm_forms),
            **suffix_info,
        }
    )


@app.post("/check-text")
def check_text():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "")

    if not isinstance(text, str):
        return jsonify({"error": "text must be a string."}), 400

    if not text.strip():
        return jsonify({"error": "Please write some Maltese text first."}), 400

    if len(text) > MAX_TEXT_LENGTH:
        return (
            jsonify(
                {
                    "error": (
                        f"Text is too long. Maximum length is "
                        f"{MAX_TEXT_LENGTH} characters."
                    )
                }
            ),
            413,
        )

    edit_distance_tolerance = int(data.get("edit_distance_tolerance", 1))
    result = spellchecker.correct_text_rich(
        text, edit_distance_tolerance=edit_distance_tolerance
    )
    corrected_text = result["corrected_text"]

    return jsonify(
        {
            "original_text": text,
            "corrected_text": corrected_text,
            "changed": corrected_text != text,
            "tokens": result["tokens"],
        }
    )


@app.post("/suggest-word")
def suggest_word():
    data = request.get_json(silent=True) or {}
    word = data.get("word", "")

    if not isinstance(word, str):
        return jsonify({"error": "word must be a string."}), 400

    if not word.strip():
        return jsonify({"error": "Please write a word first."}), 400

    if len(word) > MAX_WORD_LENGTH:
        return (
            jsonify(
                {
                    "error": (
                        f"Word is too long. Maximum length is "
                        f"{MAX_WORD_LENGTH} characters."
                    )
                }
            ),
            413,
        )

    edit_distance_tolerance = int(data.get("edit_distance_tolerance", 1))

    suggestions = spellchecker.suggest(
        word,
        limit=10,
        edit_distance_tolerance=edit_distance_tolerance,
    )

    return jsonify(
        {
            "word": word,
            "suggestions": (
                meaning_index.enrich_choices(
                    [
                        {
                            "word": suggestion,
                            "meaning": spellchecker.meaning_for(suggestion),
                        }
                        for suggestion in suggestions
                    ]
                )
            ),
        }
    )


@app.post("/debug-word")
def debug_word():
    data = request.get_json(silent=True) or {}
    word = data.get("word", "")

    if not isinstance(word, str):
        return jsonify({"error": "word must be a string."}), 400
    if not word.strip():
        return jsonify({"error": "Please write a word first."}), 400
    if len(word) > MAX_WORD_LENGTH:
        return (
            jsonify(
                {
                    "error": f"Word is too long. Maximum length is {MAX_WORD_LENGTH} characters."
                }
            ),
            413,
        )

    edit_distance_tolerance = int(data.get("edit_distance_tolerance", 1))
    return jsonify(
        spellchecker.debug_word(word, edit_distance_tolerance=edit_distance_tolerance)
    )


@app.post("/debug-orthographic")
def debug_orthographic():
    data = request.get_json(silent=True) or {}
    word = data.get("word", "")

    if not isinstance(word, str):
        return jsonify({"error": "word must be a string."}), 400
    if not word.strip():
        return jsonify({"error": "Please write a word first."}), 400
    if len(word) > MAX_WORD_LENGTH:
        return (
            jsonify(
                {
                    "error": f"Word is too long. Maximum length is {MAX_WORD_LENGTH} characters."
                }
            ),
            413,
        )

    return jsonify(orthographic_generator.debug(word))


@app.post("/debug-suffix")
def debug_suffix():
    data = request.get_json(silent=True) or {}
    word = data.get("word", "")

    if not isinstance(word, str):
        return jsonify({"error": "word must be a string."}), 400

    if not word.strip():
        return jsonify({"error": "Please write a word first."}), 400

    if len(word) > MAX_WORD_LENGTH:
        return (
            jsonify(
                {
                    "error": f"Word is too long. Maximum length is {MAX_WORD_LENGTH} characters."
                }
            ),
            413,
        )

    if not hasattr(spellchecker, "suffix_generator"):
        return jsonify({"error": "suffix generator is not attached."}), 500

    return jsonify(spellchecker.suffix_generator.debug_suffix(word))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)
