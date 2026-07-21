import json
import os
import re
import threading
import time
import unicodedata
import uuid
from Essentials.dictionary_meanings import (
    MeaningIndex,
    extract_meaning_from_payload,
    format_suffix_candidate_meaning,
    is_invalid_imperative_suffix_combination,
)
from collections import defaultdict
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterable
from Essentials.helpers.article_phrase_rules import MalteseArticlePhraseRules, WordToken
from Essentials.helpers.spellchecker_types import ScoreRow, UnifiedMatch
from Essentials.helpers.fused_preposition_rules import MalteseFusedPrepositionRules
from Essentials.helpers.suffix_generator import MalteseSuffixGenerator
from Essentials.helpers.orthographic_generator import MalteseOrthographicGenerator
from Essentials.helpers.doubled_letter_generator import MalteseDoubledLetterGenerator
from Essentials.helpers.context_analyzer import OptionalSentenceContextAnalyzer
from Essentials.grammar import MalteseGrammarRuleEngine
from Essentials.helpers.performance_logging import (
    RequestProfiler,
    current_profiler,
    log_spellcheck_event,
    reset_current_profiler,
    rss_mb,
    set_current_profiler,
)
from Essentials.helpers.symspell_index import MalteseSymSpellIndex
from Other.tools.repair_mojibake import repair_mojibake_text
from flask import Flask, jsonify, request, send_from_directory


@dataclass(slots=True)
class TokenAnalysis:
    normalized: str
    corrected: str = ""
    candidates: tuple[str, ...] = ()
    is_deterministic: bool = False
    x_candidates: tuple[str, ...] = ()
    basic_candidates: tuple[str, ...] = ()
    complex_candidates: tuple[str, ...] = ()
    phase: str = ""


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

FINAL_DICS_DIR = BASE_DIR / "finaldics"
EU_COUNTRIES_DIC = FINAL_DICS_DIR / "eu_countries.dic"
NAMES_DIC = FINAL_DICS_DIR / "names.dic"
PLACES_DIC = FINAL_DICS_DIR / "places.dic"
NO_POSSESSION_NOUNS_DIC = FINAL_DICS_DIR / "nopossessionnouns.dic"
PROTECTED_NAMES_DIC = FINAL_DICS_DIR / "protected_names.dic"
USAGE_LOG_FILE = Path(
    os.environ.get(
        "SPELLCHECK_USAGE_LOG_FILE",
        str(BASE_DIR.parent / "spellcheck_usage_log.txt"),
    )
)
USAGE_LOG_ENABLED = os.environ.get("SPELLCHECK_USAGE_LOG_ENABLED", "true").lower() not in {
    "0",
    "false",
    "no",
}
USAGE_LOG_LOCK = threading.Lock()
ENABLE_SYMSPELL_CANDIDATES = os.environ.get(
    "SPELLCHECK_ENABLE_SYMSPELL",
    "false",
).lower() in {"1", "true", "yes", "on"}
SYMSPELL_SHADOW_MODE = os.environ.get(
    "SPELLCHECK_SYMSPELL_SHADOW",
    "true",
).lower() in {"1", "true", "yes", "on"}
SYMSPELL_MAX_EDIT_DISTANCE = int(os.environ.get("SPELLCHECK_SYMSPELL_DISTANCE", "2"))
SYMSPELL_MAX_RESULTS = int(os.environ.get("SPELLCHECK_SYMSPELL_MAX_RESULTS", "64"))

ENABLE_SENTENCE_CONTEXT_ANALYZER = os.environ.get(
    "SPELLCHECK_ENABLE_CONTEXT_ANALYZER",
    "false",
).lower() in {"1", "true", "yes", "on"}
SENTENCE_CONTEXT_BACKEND = os.environ.get(
    "SPELLCHECK_CONTEXT_BACKEND",
    "stanza",
)

DICTIONARY_FILES = sorted(
    path
    for path in FINAL_DICS_DIR.glob("*.dic")
    if path.name
    not in {
        NAMES_DIC.name,
        PLACES_DIC.name,
        EU_COUNTRIES_DIC.name,
        NO_POSSESSION_NOUNS_DIC.name,
        PROTECTED_NAMES_DIC.name,
    }
)
VERB_DICTIONARY_NAMES = {"verbmt_semitic.dic", "verbmt_nonsemitic.dic"}
MEANING_DICTIONARY_FILES = [
    path for path in DICTIONARY_FILES if path.name not in VERB_DICTIONARY_NAMES
]

MAX_TEXT_LENGTH = 10_000
MAX_WORD_LENGTH = 100
LONG_TEXT_CHAR_THRESHOLD = 180
LONG_TEXT_WORD_THRESHOLD = 28

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

    # Paired quotes may mark English phrases. They are accepted only when the
    # enclosed text is an exact English entry; otherwise the quote marks are
    # discarded and the enclosed words follow the normal Maltese pipeline.
    # The boundary guards stop Maltese apostrophes like ma', t'ommkom, and
    # ssibu'x from being mistaken for English quotes.
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
        "peanuts": ["karawett"],
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
        "car": ["karozza"],
        "max": ["mg\u0127ax"],
        "seat": ["siġġu"],
        "security": ["sigurtà"],
        "toilet": ["tojlit"],
        "velcro": ["velkro"],
        "shark": ["kelb il-baħar"]
    }

    ACCEPTED_ENGLISH_WORDS = {
        "a",
        "about",
        "and",
        "at",
        "automatic",
        "air",
        "car",
        "beta",
        "block",
        "basket",
        "day",
        "down",
        "end",
        "fan",
        "gas",
        "hello",
        "hi",
        "iphone",
        "max",
        "mobile",
        "morning",
        "owner",
        "of",
        "ok",
        "okay",
        "partner",
        "please",
        "phone",
        "panic",
        "plus",
        "pro",
        "seat",
        "security",
        "support",
        "teacher",
        "transport",
        "container",
        "overpopulation",
        "supermarket",
        "the",
        "toilet",
        "trolley",
        "trolly",
        "update",
        "velcro",
        "wolt",
        "euro",
    }

    ACCEPTED_ENGLISH_PHRASES = {
        "at the end of the day",
        "air b n b",
        "for granted",
        "good morning",
        "south area",
    }

    # Product spellings may be accepted in a spaced user form while the
    # checker returns the established English spelling after quotes are
    # removed. These are intentionally exact aliases, not English fuzzing.
    ENGLISH_CANONICAL_PHRASES = {
        "i phone 17 pro max": "iphone 17 pro max",
    }

    ENGLISH_DISPLAY_NOTES = {
        "max": "il-kelma massimu bl-Ingli\u017c",
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
        "xej": "xejn",
        "qas": "lanqas",
        "qass": "lanqas",
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
        "nowm": "ngħum",
        "nowmu": "ngħumu",
        "towm": "tgħum",
        "towmu": "tgħumu",
        "xahad": "xi ħadd",
        "qiedgha": "qiegħda",
        "qedgha": "qiegħda",
        "qedgħa": "qiegħda",
        "ghet": "qed",
        "et": "qed",
        "ett": "qed",
        "ed": "qed",
        "qet": "qed",
        "qett": "qed",
        "skond": "skont",
        "solitu": "soltu",
        "ghidt": "għedt",
        "ghidtila": "għedtilha",
        "ghidtlek": "għedtlek",
        "ergajt": "erġajt",
        "gitni": "ġietni",
        "hallitni": "ħallietni",
        "tieghak": "tiegħek",
        "tieghek": "tiegħek",
        "tiegħak": "tiegħek",
        "fuqa": "fuqha",
        "inkellima": "inkellimha",
        "qaletli": "qaltli",
        "qaletlek": "qaltlek",
        "qaletlu": "qaltlu",
        "qalitli": "qaltli",
        "qalitlek": "qaltlek",
        "qalitlu": "qaltlu",
        "qaltilha": "qaltilha",
        "prezentuz": "preżentuż",
        "presentuz": "preżentuż",
        "preżentuz": "preżentuż",
        "presentuż": "preżentuż",
        "prezentuż": "preżentuż",
        "prezentuza": "preżentuża",
        "presentuza": "preżentuża",
        "preżentuza": "preżentuża",
        "presentuża": "preżentuża",
        "prezentuża": "preżentuża",
        "prezentuzi": "preżentużi",
        "presentuzi": "preżentużi",
        "preżentuzi": "preżentużi",
        "presentużi": "preżentużi",
        "prezentużi": "preżentużi",
        "ala": "għala",
        "anna": "għandna",
        "alla": "għala",
        "ghanna": "għandna",
        "għanna": "għandna",
        "il bierah": "ilbieraħ",
        "il bieraħ": "ilbieraħ",
        "baskitbol": "basketball",
        "l-baskitbol": "l-basketball",
        "il-baskitbol": "il-basketball",
        "bejsbol": "baseball",
        "jowm": "tgħum",
        "jowmu": "tgħumu",
    }

    # The verb ``ta`` has a compact direct-object paradigm which cannot be
    # recovered by the ordinary suffix splitter.  These are lexical verb
    # surfaces, not correction replacements; i-/j- variants are retained as
    # distinct context-selected forms.
    TA_DIRECT_OBJECT_GLOSSES = {
        "ntik": "I give you", "intik": "I give you",
        "ntih": "I give him", "intih": "I give him",
        "ntiha": "I give her", "intiha": "I give her",
        "ntikom": "I give you all", "intikom": "I give you all",
        "ntihom": "I give them", "intihom": "I give them",
        "ttini": "you give me", "ttih": "you give him",
        "ttiha": "you give her", "ttina": "you give us",
        "ttihom": "you give them",
        "jtini": "he gives me", "itini": "he gives me",
        "jtik": "he gives you", "itik": "he gives you",
        "jtiha": "he gives her", "itiha": "he gives her",
        "jtina": "he gives us", "itina": "he gives us",
        "jtikom": "he gives you all", "itikom": "he gives you all",
        "jtihom": "he gives them", "itihom": "he gives them",
        "ittini": "she gives me", "ittik": "she gives you",
        "ittih": "she gives him", "ittiha": "she gives her",
        "ittina": "she gives us", "ittihom": "she gives them",
        "ntuh": "we give him", "intuh": "we give him",
        "ntuha": "we give her", "intuha": "we give her",
        "ntukom": "we give you all", "intukom": "we give you all",
        "ntuhom": "we give them", "intuhom": "we give them",
        "ttuna": "you all give us", "ittuna": "you all give us",
        "ttuh": "you all give him", "ittuh": "you all give him",
        "ttuha": "you all give her", "ittuha": "you all give her",
        "ttuhom": "you all give them", "ittuhom": "you all give them",
        "jtuna": "they give us", "ituna": "they give us",
        "jtuh": "they give him", "ituh": "they give him",
        "jtuha": "they give her", "ituha": "they give her",
        "jtukom": "they give you all", "itukom": "they give you all",
        "jtuhom": "they give them", "ituhom": "they give them",
    }

    QIEGHED_SPELLING_VARIANTS = frozenset(
        {"qieghad", "qiegħad", "qieghat", "qiegħat", "qijat", "qijad"}
    )

    MANUAL_WORD_SUGGESTIONS = {
        "ilbierah": ("ilbieraħ",),
        "ajru": ("ajruplan",),
        "qalb": ("qalb", "qalbi"),
        "mhabba": ("imħabba",),
        "tbatija": ("tbatija",),
        "tana": ("tana", "tagħna"),
        "għajjejt": ("għajejt",),
        "ghajjejt": ("għajejt",),
        "ajjejt": ("għajjejt", "għajejt"),
        "għar": ("agħar",),
        "agħar": ("għar",),
        "ghar": ("għar", "agħar"),
        "aghar": ("agħar", "għar"),
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

    CAPITALIZED_PLACE_CANDIDATE_LIMIT = 5
    SENTENCE_INITIAL_CANDIDATE_LIMIT = 10
    TIME_EXPRESSION_WORDS = {
        "filghodu": "filgħodu",
        "filodu": "filgħodu",
        "filgħaxija": "filgħaxija",
        "dalghodu": "dalgħodu",
        "nofsinhar": "nofsinhar",
        "waranofsinhar": "waranofsinhar",
        "lum": "llum",
        "llum": "illum",
        "illum": "illum",
        "bierah": "bieraħ",
        "lbierah": "ilbieraħ",
        "ilbierah": "ilbieraħ",
        "lbirahtlula": "ilbiraħtlula",
        "ilbirahtlula": "ilbiraħtlula",
        "llajs": "llajs",
        "illajs": "llajs",
        "llahwa": "llaħwa",
        "illahwa": "llaħwa",
        "llami": "llami",
        "illami": "llami",
        "llejla": "llejla",
        "billejl": "billejl",
    }
    CARDINAL_TO_SHORT_ATTNUM = {
        "ewġ": "żewġ",
        "ewġt": "żewġ",
        "wieħed": "wieħed",
        "waħda": "waħda",
        "tlieta": "tliet",
        "tlitt": "tliet",
        "erbgħa": "erba'",
        "erbat": "erba'",
        "ħamsa": "ħames",
        "ħamest": "ħames",
        "sitta": "sitt",
        "sebgħa": "seba'",
        "sebat": "seba'",
        "tmienja": "tmien",
        "tmint": "tmien",
        "disgħa": "disa'",
        "disat": "disa'",
        "għaxra": "għaxar",
        "għaxart": "għaxar",
        "ġiex": "ġiex",
        "ġixt": "ġiex",
        "ħdax": "ħdax-il",
        "ħdax-il": "ħdax-il",
        "għoxrin": "għoxrin",
    }
    SHORT_ATTNUM_TO_LONG = {
        "żewġ": "żewġt",
        "wieħed": "wieħed",
        "waħda": "waħda",
        "tliet": "tlitt",
        "erba'": "erbat",
        "ħames": "ħamest",
        "sitt": "sitt",
        "seba'": "sebat",
        "tmien": "tmint",
        "disa'": "disat",
        "għaxar": "għaxart",
        "ġiex": "ġixt",
        "ħdax-il": "ħdax-il",
        "għoxrin": "għoxrin",
    }
    SINGULAR_NUMBER_WORDS = {
        "ħdax",
        "ħdax-il",
        "tnax",
        "tnax-il",
        "tlettax",
        "erbatax",
        "ħmistax",
        "sittax",
        "sbatax",
        "tmintax",
        "dsatax",
        "għoxrin",
    }
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
        "jowm": "tgħum",
        "jowmu": "tgħumu",
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
        "verita": "verità",
        "inteligenti": "intelliġenti",
        "tantx": "tantx",
        "mhawx": "m'hawnx",
        "m'hawx": "m'hawnx",
        "m’hawx": "m'hawnx",
        "m’hawnx": "m'hawnx",
        "m'hawnx": "m'hawnx",
        "ala": "għala",
        "hadd": "ħadd",
        "alla": "għala",
        "principju": "prinċipju",
        "x'taghmel": "x'tagħmel",
        "faqqalu": "faqqagħlu",
        "malta": "Malta",
        "qazzist": "qażżiżt",
        "qazzisti": "qażżiżti",
    }

    EXACT_SUGGESTION_OVERRIDES = {
        "biex": (),
        "xhin": ("xħin", "x'ħin"),
        "xħin": ("x'ħin",),
        "x'ħin": ("xħin",),
        "għalaq": ("agħlaq",),
        "ghalaq": ("agħlaq",),
        "waranofsinhar": ("wara nofsinhar",),
        "llejla": ("il-lejla",),
        "min": ("minn",),
        "qalu": ("qallu",),
        "hu": ("ħu", "u"),
        "hi": ("ħi",),
        "ghal": ("għall-",),
        "għal": ("għall-",),
        "filwaqt": ("fil-waqt",),
        "kif": ("kief",),
        "kief": ("kif",),
        "nixxi": ("nixtri",),
        "tixxi": ("tixtri",),
        "jixxi": ("jixtri",),
        "nixxu": ("nixtru",),
        "tixxu": ("tixtru",),
        "jixxu": ("jixtru",),
        "ixxi": ("ixtri",),
        "ixxu": ("ixtru",),
        "għajjejt": ("għajejt",),
        "ghajjejt": ("għajjejt", "għajejt"),
        "ajjejt": ("għajjejt", "għajejt"),
    }

    INITIAL_VOWEL_NOUN_EXCEPTIONS = {
        "uċuħ": ("wċuħ",),
        "wċuħ": ("uċuħ",),
        "uġigħ": ("wġigħ",),
        "wġigħ": ("uġigħ",),
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
        self.instance_id = uuid.uuid4().hex[:12]
        self._local = threading.local()
        self._no_possession_nouns: frozenset[str] | None = None
        self._protected_names: frozenset[str] | None = None
        self._given_names: frozenset[str] | None = None
        self._surnames: frozenset[str] | None = None
        self._noun_number_index: dict[str, dict[str, tuple[str, ...]]] | None = None
        self._noun_number_prefix_index: dict[str, dict[str, tuple[str, ...]]] | None = None
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
        self.raw_entries: list[tuple[str, str | None]] = []

        # paradigm key -> all surface forms in that paradigm
        self.paradigm_forms: dict[str, list[str]] = defaultdict(list)

        # consonant anchor -> surface forms
        self.anchor_map: dict[str, list[str]] = defaultdict(list)
        self.anchor_letters: tuple[str, ...] = ()

        # Cached metadata
        self.word_lengths: dict[str, int] = {}
        self.word_vowel_counts: dict[str, int] = {}
        self.word_anchors: dict[str, str] = {}
        self._missing_h_verb_repairs: dict[str, str] | None = None
        self.symspell_index: MalteseSymSpellIndex | None = None

        raw_entries: list[tuple[str, str | None]] = []

        if dictionary_words:
            raw_entries.extend((word, None) for word in dictionary_words)

        if dictionary_files:
            raw_entries.extend(self._load_dictionary_files(list(dictionary_files)))
        raw_entries.extend(self._load_eu_single_word_entries(EU_COUNTRIES_DIC))
        raw_entries.extend(self._load_eu_single_word_entries(PLACES_DIC))
        self.raw_entries = raw_entries
        calendar_terms = {
            "january", "february", "march", "april", "may", "june",
            "july", "august", "september", "october", "november",
            "december", "monday", "tuesday", "wednesday", "thursday",
            "friday", "saturday", "sunday",
        }
        self._calendar_proper_names = frozenset(
            self._normalize_word(word)
            for word, tag in raw_entries
            if word[:1].isupper()
            and tag
            and any(term in str(tag).casefold() for term in calendar_terms)
        )

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
                runtime_tag = str(tag).split()[0]
                self.word_tags[normalized].add(str(tag))
                if (
                    self._is_paradigm_tag(runtime_tag)
                    and normalized
                    not in seen_paradigm_forms[self._parse_paradigm_key(runtime_tag)]
                ):
                    paradigm_key = self._parse_paradigm_key(runtime_tag)
                    self.paradigm_forms[paradigm_key].append(normalized)
                    seen_paradigm_forms[paradigm_key].add(normalized)

        self.dictionary_set = set(self.dictionary)
        self._load_country_place_index(FINAL_DICS_DIR / "eu_countries.json")
        self._load_places_dictionary(PLACES_DIC)
        self._build_word_metadata()
        self._build_anchor_index()
        self.word_anchors.clear()
        if ENABLE_SYMSPELL_CANDIDATES:
            self.symspell_index = MalteseSymSpellIndex(
                normalizer=self._normalize_word,
                token_key=self._letter_tokens,
                max_edit_distance=SYMSPELL_MAX_EDIT_DISTANCE,
            )
            symspell_stats = self.symspell_index.build(self.dictionary)
            log_spellcheck_event(
                event="SYMSPELL_BUILD",
                instance_id=self.instance_id,
                enabled=True,
                words=symspell_stats.words,
                delete_keys=symspell_stats.delete_keys,
                elapsed_ms=symspell_stats.elapsed_ms,
                rss_mb=rss_mb(),
            )

        print(
            f"Loaded {len(self.dictionary)} dictionary words, "
            f"{len(self.paradigm_forms)} paradigms."
        )

    def tagged_words_with_marker(self, marker: str) -> set[str]:
        marker = str(marker or "").upper()
        if not marker:
            return set()
        out: set[str] = set()
        for word, tags in self.word_tags.items():
            if any(marker in tag.split("-", 1)[0].upper() for tag in tags):
                out.add(word)
        return out

    # ------------------------------------------------------------------
    # Normalisation/tokenisation
    # ------------------------------------------------------------------

    def _normalize_word(self, word: str) -> str:
        """Lower-case, NFC-normalise, and unify apostrophe variants."""
        return self._normalize_word_cached(str(word))

    @staticmethod
    @lru_cache(maxsize=8192)
    def _normalize_word_cached(word: str) -> str:
        return (
            unicodedata.normalize("NFC", str(word).strip().lower())
            .replace("\u2019", "'")
            .replace("\u2018", "'")
            .replace("\u02bc", "'")
        )

    def _english_key(self, text: str) -> str:
        return " ".join(str(text).casefold().strip().split())

    def _accepted_exact_english(self, text: str) -> bool:
        key = self._english_key(text)
        if not key:
            return False
        if key in self.ENGLISH_CANONICAL_PHRASES:
            return True
        if " " in key:
            return key in self.ACCEPTED_ENGLISH_PHRASES or key in self.ENGLISH_MAPPINGS
        return key in self.ACCEPTED_ENGLISH_WORDS or key in self.ENGLISH_MAPPINGS

    def _canonical_english_text(self, text: str) -> str:
        key = self._english_key(text)
        return self.ENGLISH_CANONICAL_PHRASES.get(key, text)

    def _correct_quoted_english_text(self, text: str) -> tuple[str, bool]:
        """Keep only exact English text inside paired quotation marks."""
        key = self._english_key(text)
        if not key:
            return text, False
        if self._accepted_exact_english(key):
            return self._canonical_english_text(text), True
        return text, False

    def _accepted_article_english(self, word: str) -> bool:
        """Whether a single exact English word may take a Maltese article."""
        key = self._english_key(word)
        return bool(key and " " not in key and self._accepted_exact_english(key))

    def _near_accepted_english_word(self, text: str) -> bool:
        key = self._english_key(text)
        if " " in key or len(key) < 6 or not key.isalpha():
            return False
        if (
            key in self.dictionary_set
            or key in self._no_possession_noun_set()
            or key in self.country_english_to_maltese
            or key in self.country_maltese_to_english
            or self._exact_place_word(key)
        ):
            return False

        accepted_words = set(self.ACCEPTED_ENGLISH_WORDS)
        accepted_words.update(
            word for word in self.ENGLISH_MAPPINGS if " " not in word and len(word) >= 6
        )
        for accepted in accepted_words:
            if (
                len(accepted) >= 6
                and key[:1] == accepted[:1]
                and abs(len(key) - len(accepted)) <= 2
                and self._word_distance(key, accepted) <= 2
            ):
                return True
        return False

    def _english_token(
        self,
        *,
        original: str,
        corrected: str,
        inner_text: str,
    ) -> dict:
        key = self._english_key(inner_text)
        mapped = self.ENGLISH_MAPPINGS.get(key, [])
        if isinstance(mapped, str):
            mapped = [mapped]
        suggestions = list(mapped)
        if not suggestions:
            for suggestion in self._english_fixed_noun_suggestions(key):
                if suggestion not in suggestions:
                    suggestions.append(suggestion)
        token = {
            "type": "english_phrase",
            "original": original,
            "corrected": corrected,
            "inner_text": inner_text,
            "maltese_suggestion": suggestions or None,
        }
        note = self.ENGLISH_DISPLAY_NOTES.get(key)
        if note:
            token["english_note"] = note
        return token

    def _exact_unquoted_english_phrase_at(
        self,
        text: str,
        matches: list[UnifiedMatch | re.Match[str]],
        index: int,
    ) -> tuple[str, int] | None:
        """Return an exact approved English phrase beginning at ``index``.

        Phrase detection runs before individual words so an entry such as
        ``for granted`` cannot be passed as two separate Maltese candidates.
        It accepts only whitespace between terms; punctuation deliberately
        ends the phrase.
        """
        for phrase in sorted(
            (
                value
                for value in (*self.ACCEPTED_ENGLISH_PHRASES, *self.ENGLISH_MAPPINGS)
                if " " in value
            ),
            key=lambda value: len(value.split()),
            reverse=True,
        ):
            words = phrase.split()
            end_index = index + len(words) - 1
            if end_index >= len(matches):
                continue
            if any(
                getattr(matches[position], "is_quote", False)
                for position in range(index, end_index + 1)
            ):
                continue
            original_phrase = text[matches[index].start() : matches[end_index].end()]
            if self._english_key(original_phrase) == phrase:
                return original_phrase, len(words)
        return None

    def _graphemes(self, word: str) -> list[str]:
        """Split a word into rough graphemes, keeping għ as one item."""
        return list(self._graphemes_cached(self._normalize_word(word)))

    @staticmethod
    @lru_cache(maxsize=4096)
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

    @lru_cache(maxsize=4096)
    def _letter_tokens_raw(self, word: str) -> tuple[str, ...]:
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
        return tuple(tokens)

    def _letter_tokens(self, word: str) -> tuple[str, ...]:
        normalized = self._normalize_word(word)
        return self._letter_tokens_raw(normalized)

    def _reset_request_token_cache(self) -> None:
        self._local.token_cache = {}
        self._local.suggestion_cache = {}

    def _request_token_cache(self) -> dict[str, TokenAnalysis]:
        cache = getattr(self._local, "token_cache", None)
        if cache is None:
            cache = {}
            self._local.token_cache = cache
        return cache

    def _request_suggestion_cache(
        self,
    ) -> dict[tuple[str, int, int], tuple[str, ...]]:
        cache = getattr(self._local, "suggestion_cache", None)
        if cache is None:
            cache = {}
            self._local.suggestion_cache = cache
        return cache

    def _get_token_analysis(self, word: str) -> TokenAnalysis | None:
        return self._request_token_cache().get(word)

    def _store_token_analysis(
        self,
        word: str,
        *,
        corrected: str | None = None,
        candidates: Iterable[str] | None = None,
        is_deterministic: bool | None = None,
    ) -> TokenAnalysis:
        cache = self._request_token_cache()
        analysis = cache.get(word)
        if analysis is None:
            analysis = TokenAnalysis(normalized=self._normalize_word(word))
            cache[word] = analysis
        if corrected is not None:
            analysis.corrected = corrected
        if candidates is not None:
            unique_candidates: list[str] = []
            for candidate in candidates:
                candidate = str(candidate)
                if candidate and candidate not in unique_candidates:
                    unique_candidates.append(candidate)
            analysis.candidates = tuple(unique_candidates)
        if is_deterministic is not None:
            analysis.is_deterministic = is_deterministic
        return analysis

    def _allows_guttural_candidate(self, source: str, candidate: str) -> bool:
        """Reject unsafe initial or medial guttural substitutions.

        A typed ``h`` may still represent ``ħ`` and a typed final guttural may
        be repaired through its dedicated path.  This guard only prevents the
        generic generator from inventing ``ħ`` or swapping ``għ`` and ``ħ``
        within a word.
        """
        source_norm = self._normalize_word(source)
        candidate_norm = self._normalize_word(candidate)
        if not source_norm or source_norm == candidate_norm:
            return True

        # Do not turn an inserted plain h into a new ħ. A typed h remains a
        # valid shortcut source because deleting ħ would not recover it.
        if "ħ" in candidate_norm and candidate_norm.replace("ħ", "") == source_norm:
            return False

        source_has_gh = "għ" in source_norm or "gh" in source_norm
        candidate_has_gh = "għ" in candidate_norm
        source_has_hbar = "ħ" in source_norm
        candidate_has_hbar = "ħ" in candidate_norm
        if source_has_gh and candidate_has_hbar and not candidate_has_gh:
            return False
        if source_has_hbar and candidate_has_gh and not candidate_has_hbar:
            return False
        return True

    def _qieged_spelling_repair(self, word: str) -> str | None:
        normalized = self._normalize_word(word)
        if normalized in self.QIEGHED_SPELLING_VARIANTS and "qiegħed" in self.dictionary_set:
            return "qiegħed"
        return None

    def _ta_direct_object_gloss(self, word: str) -> str:
        return self.TA_DIRECT_OBJECT_GLOSSES.get(self._normalize_word(word), "")

    def _phase_x_collect_candidates(self, word: str) -> TokenAnalysis:
        normalized = self._normalize_word(word)
        analysis = self._store_token_analysis(word)
        analysis.phase = "X"
        analysis.normalized = normalized

        if not normalized:
            return analysis

        basic_candidates: list[str] = []
        complex_candidates: list[str] = []
        priority_basic_candidate: str | None = None
        priority_basic_locked = False

        def add_unique(target: list[str], candidate: str) -> None:
            candidate = self._normalize_word(candidate)
            if (
                candidate
                and self._allows_guttural_candidate(normalized, candidate)
                and candidate not in target
            ):
                target.append(candidate)

        def is_exact_y_surface(candidate: str) -> bool:
            """Accept direct dictionary words and generated noun possessives."""
            return (
                candidate in self.dictionary_set
                or self._correct_noun_possessive_suffix(candidate) == candidate
            )

        if normalized in self.dictionary_set:
            add_unique(basic_candidates, normalized)
            analysis.corrected = self._match_capitalisation(word, normalized)
            analysis.is_deterministic = True

        manual_repair = self.MANUAL_WORD_REPAIRS.get(normalized)
        if manual_repair:
            add_unique(basic_candidates, manual_repair)
            priority_basic_candidate = manual_repair
            priority_basic_locked = True
            analysis.corrected = self._match_capitalisation(word, manual_repair)
            analysis.is_deterministic = True

        if normalized.startswith("l'") and len(normalized) > 2:
            tail = normalized[2:]
            if tail and tail[0] in self.VOWELS and self._is_probable_noun(tail):
                article_surface = "l-" + tail
                add_unique(basic_candidates, article_surface)
                analysis.corrected = self._match_capitalisation(word, article_surface)
                analysis.is_deterministic = True

        # Explicitly approved lexical exception: unlike the unrelated verb
        # ``qass``, this discourse particle always expands to ``lanqas``.
        if normalized in {"qas", "qass"}:
            add_unique(basic_candidates, "lanqas")
            analysis.corrected = self._match_capitalisation(word, "lanqas")
            analysis.is_deterministic = True

        direct_object_h_repairs = self._restore_direct_object_h_vowel(normalized)
        if direct_object_h_repairs:
            add_unique(basic_candidates, direct_object_h_repairs[0])
            analysis.corrected = self._match_capitalisation(
                word,
                direct_object_h_repairs[0],
            )
            analysis.is_deterministic = True

        qieged_repair = self._qieged_spelling_repair(normalized)
        if qieged_repair is not None:
            add_unique(basic_candidates, qieged_repair)
            analysis.corrected = self._match_capitalisation(word, qieged_repair)
            analysis.is_deterministic = True

        if self._ta_direct_object_gloss(normalized):
            add_unique(basic_candidates, normalized)
            analysis.corrected = self._match_capitalisation(word, normalized)
            analysis.is_deterministic = True

        # An initial i may be an epenthetic surface over a misspelled verb.
        # Re-run the bounded basic operations on the lexical stem, then admit
        # only a genuine verb result. This composes i-removal, a shortcut
        # consonant, gemination, and ordinary suffix recovery without a
        # replacement table (inhalasa -> nħallasha).
        if (
            normalized.startswith("i")
            and len(normalized) > 4
            and not self._valid_initial_vowel_surface_word(normalized)
            and not normalized.startswith(("il", "it", "is"))
        ):
            stem = normalized[1:]
            orthographic = getattr(self, "orthographic_generator", None)
            doubled = getattr(self, "doubled_letter_generator", None)
            suffix_generator = getattr(self, "suffix_generator", None)
            stem_seeds = [stem]
            if orthographic is not None:
                stem_seeds.extend(
                    orthographic.shortcut_letter_variants(
                        stem,
                        max_changes=1,
                        max_variants=16,
                    )
                )
            for seed in stem_seeds:
                candidate_seeds = [seed]
                if doubled is not None:
                    candidate_seeds.extend(doubled.missing_double_variants(seed))
                for candidate_seed in candidate_seeds:
                    recovered = (
                        suffix_generator.correct_suffix(candidate_seed)
                        if suffix_generator is not None
                        else None
                    )
                    for candidate in (candidate_seed, recovered):
                        if candidate and self._is_verb_tagged_word(candidate):
                            surface_candidate = candidate
                            ji_candidate = f"j{candidate}"
                            if (
                                not candidate.startswith(("i", "j"))
                                and self._is_imperfect_surface_candidate(ji_candidate)
                            ):
                                surface_candidate = ji_candidate
                            add_unique(basic_candidates, surface_candidate)

        # ``x'`` is a Maltese contraction, not an English quote. Correct its
        # lexical tail through the same basic pipeline before phrase parsing.
        if normalized.startswith("x'") and len(normalized) > 2:
            corrected_tail = self.correct_word(normalized[2:])
            if self._normalize_word(corrected_tail) != normalized[2:]:
                apostrophe_candidate = f"x'{corrected_tail}"
                add_unique(basic_candidates, apostrophe_candidate)
                analysis.corrected = self._match_capitalisation(
                    word,
                    apostrophe_candidate,
                )
                analysis.is_deterministic = True

        if normalized == "x'imkien":
            add_unique(complex_candidates, "xi mkien")
            analysis.corrected = self._match_capitalisation(word, "xi mkien")
            analysis.is_deterministic = True

        # Function words are often typed together with their following verb.
        # Split only when the recovered tail is a verified verb surface.
        for compact_prefix, expanded_prefix in (
            ("ma", "ma"),
            ("ha", "ħa"),
            ("se", "se"),
            ("sa", "sa"),
        ):
            if not normalized.startswith(compact_prefix) or len(normalized) <= len(compact_prefix) + 1:
                continue
            compact_tail = normalized[len(compact_prefix):]
            corrected_tail = self.correct_word(compact_tail)
            corrected_tail_norm = self._normalize_word(corrected_tail)
            if not self._is_verb_tagged_word(corrected_tail_norm):
                continue
            compact_phrase = f"{expanded_prefix} {corrected_tail_norm}"
            add_unique(complex_candidates, compact_phrase)
            analysis.corrected = self._match_capitalisation(word, compact_phrase)
            analysis.is_deterministic = True
            break

        # The compact interrogative forms xikun/x'ikun retain the long xi
        # when their tail resolves to the imperfect jkun surface.
        xi_tail = None
        if normalized.startswith("xi") and len(normalized) > 2:
            xi_tail = normalized[2:]
        elif normalized.startswith("x'") and normalized[2:] == "ikun":
            xi_tail = normalized[2:]
        if xi_tail:
            corrected_tail = self.correct_word(xi_tail)
            corrected_tail_norm = self._normalize_word(corrected_tail)
            initial_j_tail = self._initial_i_j_surface_repair(corrected_tail_norm)
            if initial_j_tail is not None:
                corrected_tail_norm = initial_j_tail
            ji_tail = f"j{corrected_tail_norm}"
            if self._is_verb_tagged_word(ji_tail):
                corrected_tail_norm = ji_tail
            if self._is_verb_tagged_word(corrected_tail_norm):
                add_unique(complex_candidates, f"xi {corrected_tail_norm}")
                analysis.corrected = self._match_capitalisation(
                    word,
                    f"xi {corrected_tail_norm}",
                )
                analysis.is_deterministic = True

        # A leading l can belong to a contracted l' word whose initial għ/h
        # was omitted. Rebuild only a verified lexical tail.
        if (
            normalized not in self.dictionary_set
            and normalized.startswith("l")
            and len(normalized) > 2
            and normalized[1] in self.VOWELS
        ):
            l_tail = normalized[1:]
            l_candidates = [f"għ{l_tail}", f"h{l_tail}"]
            for l_candidate in tuple(l_candidates):
                l_candidates.extend(self._dictionary_verified_basic_repairs(l_candidate))
            for l_candidate in l_candidates:
                if self._is_recognized_surface(l_candidate):
                    add_unique(complex_candidates, f"l'{l_candidate}")
                    analysis.corrected = self._match_capitalisation(
                        word,
                        f"l'{l_candidate}",
                    )
                    analysis.is_deterministic = True
                    break

        # Unhyphenated definite articles can attach to a verified noun tail.
        # Restore their normal assimilated surface before suffix logic sees a
        # misleading word-initial double consonant (ittoqba -> it-toqba).
        article_rules = getattr(self, "article_phrase_rules", None)
        if article_rules is not None:
            for prefix in ("il", "it", "is", "ir", "in", "id", "iċ", "ic", "ix", "iż", "iz"):
                if not normalized.startswith(prefix) or len(normalized) <= len(prefix):
                    continue
                # inm- is the initial-vowel surface of an mm- verb, not an
                # in- article followed by a nominal tail.
                if (
                    prefix == "in"
                    and normalized.startswith("inm")
                ):
                    continue
                noun_tail = normalized[len(prefix):]
                tail_options = [noun_tail]
                tail_options.extend(self._dictionary_verified_basic_repairs(noun_tail))
                tail_options.append(self._normalize_word(self.correct_word(noun_tail)))
                corrected_noun_tail = next(
                    (
                        option
                        for option in tail_options
                        if self._is_noun_tagged_word(option)
                        or self._is_adjective_tagged_word(option)
                    ),
                    "",
                )
                if not corrected_noun_tail:
                    continue
                article_surface = (
                    f"{article_rules.assimilate('il', corrected_noun_tail)}"
                    f"{corrected_noun_tail}"
                )
                add_unique(complex_candidates, article_surface)
                analysis.corrected = self._match_capitalisation(word, article_surface)
                analysis.is_deterministic = True
                break

        # Compact bi-/fi- plus the assimilated x article may arrive without
        # its hyphen. Validate the restored adjective/noun tail first.
        for compact_prefix in ("bix", "fix"):
            if normalized.startswith(compact_prefix) and len(normalized) > len(compact_prefix):
                tail = normalized[len(compact_prefix):]
                tail_candidates = [f"x{tail}"]
                orthographic = getattr(self, "orthographic_generator", None)
                if orthographic is not None:
                    tail_candidates.extend(
                        orthographic.substitute_i_ie(tail_candidates[0])
                    )
                for tail_candidate in tail_candidates:
                    restored_tail = self.correct_word(tail_candidate)
                    restored_norm = self._normalize_word(restored_tail)
                    if restored_norm.startswith("x") and self._is_recognized_surface(restored_norm):
                        compact_candidate = f"{compact_prefix}-{restored_norm}"
                        add_unique(complex_candidates, compact_candidate)
                        analysis.corrected = self._match_capitalisation(
                            word,
                            compact_candidate,
                        )
                        analysis.is_deterministic = True
                        break

        # Compact assimilated b-/f- prepositions retain their own sun-letter
        # prefix while their noun tail follows the normal correction path.
        for compact_prefix in (
            "bic", "biċ", "bid", "bin", "bir", "bis", "bit", "bix", "biz", "biż",
            "fic", "fiċ", "fid", "fin", "fir", "fis", "fit", "fix", "fiz", "fiż",
        ):
            if compact_prefix in {"bix", "fix"}:
                continue
            if not normalized.startswith(compact_prefix) or len(normalized) <= len(compact_prefix):
                continue
            compact_tail = normalized[len(compact_prefix):]
            corrected_tail = self.correct_word(compact_tail)
            corrected_tail_norm = self._normalize_word(corrected_tail)
            if not corrected_tail_norm or not self._is_recognized_surface(corrected_tail_norm):
                continue
            add_unique(complex_candidates, f"{compact_prefix}-{corrected_tail_norm}")
            analysis.corrected = self._match_capitalisation(
                word,
                f"{compact_prefix}-{corrected_tail_norm}",
            )
            analysis.is_deterministic = True
            break

        # Possessive noun surfaces are generated from tagged dictionary nouns.
        # Feed their validated correction into X so the Y resolver can use it
        # just like a direct dictionary spelling.
        possessive_surface = self._correct_noun_possessive_suffix(normalized)
        if possessive_surface is not None:
            add_unique(basic_candidates, possessive_surface)
            if possessive_surface != normalized:
                analysis.corrected = self._match_capitalisation(
                    word,
                    possessive_surface,
                )
                analysis.is_deterministic = True

        # Keep lexical compounds ahead of the generic repair machinery.  A
        # form such as ``filodu`` is an orthographic spelling of one lexical
        # expression, not an invitation for the article parser to create
        # ``fl-għodu``.
        fixed_time = self._fixed_time_expression_word(normalized)
        if fixed_time is not None:
            add_unique(basic_candidates, fixed_time)

        # Lexicalized expressions have their own bounded spelling space.
        # Admit a verified canonical form before generic anchors score an
        # unrelated word.
        lexicalized_forms = self._lexicalized_form_variants(normalized)
        if lexicalized_forms:
            add_unique(basic_candidates, lexicalized_forms[0])
            if lexicalized_forms[0] != normalized:
                analysis.corrected = self._match_capitalisation(
                    word,
                    lexicalized_forms[0],
                )
                analysis.is_deterministic = True

        contracted_fb = self._repair_contracted_fb_word(word)
        if contracted_fb is not None:
            contracted_word, _tail = contracted_fb
            add_unique(basic_candidates, contracted_word)
            analysis.corrected = self._match_capitalisation(word, contracted_word)
            analysis.is_deterministic = True

        terminal_apostrophe = self._strict_terminal_apostrophe_match(normalized)
        if terminal_apostrophe is not None:
            add_unique(basic_candidates, terminal_apostrophe)

        # This is an explicitly approved lexical completion rather than a
        # fuzzy replacement: the clipped form always expands to ``xejn``.
        if normalized == "xej":
            add_unique(basic_candidates, "xejn")
        if normalized == "aw":
            add_unique(basic_candidates, "hawn")

        missing_gh_priority = self._missing_gh_mperf_repair(normalized)
        if missing_gh_priority is not None:
            add_unique(basic_candidates, missing_gh_priority)
            priority_basic_candidate = missing_gh_priority

        # Existing pattern rules remain useful only when they terminate on a
        # real lexical or generated verb surface.  Keeping that verification
        # here lets Phase Y use them without reopening fuzzy correction.
        pattern_priority: str | None = None
        for candidate in self._pattern_repair_variants(normalized):
            candidate_parts = candidate.split()
            verified_pattern_candidate = (
                candidate in self.dictionary_set
                or self._is_verb_tagged_word(candidate)
                or (
                    len(candidate_parts) == 2
                    and candidate_parts[0] in {"ħa", "se", "sa", "ma"}
                    and self._is_verb_tagged_word(candidate_parts[1])
                )
            )
            if verified_pattern_candidate:
                add_unique(basic_candidates, candidate)
                if len(candidate_parts) == 2:
                    analysis.corrected = self._match_capitalisation(word, candidate)
                    analysis.is_deterministic = True
                if pattern_priority is None and (
                    normalized == "tajru" or (
                        "għ" in candidate
                        and any(sequence in normalized for sequence in ("aj", "ej"))
                    )
                ):
                    pattern_priority = candidate

        # Phase Y may compose a small number of independently safe spelling
        # operations, but it only keeps dictionary or generated-surface hits.
        # This is intentionally not a replacement table: it applies the same
        # bounded operations to every unrecognised word.
        if fixed_time is None and self._has_basic_repair_cue(normalized):
            verified_basic_repairs = self._dictionary_verified_basic_repairs(normalized)
            for candidate in verified_basic_repairs:
                add_unique(basic_candidates, candidate)
            if verified_basic_repairs:
                current_norm = self._normalize_word(analysis.corrected)
                first_repair = self._normalize_word(verified_basic_repairs[0])
                current_possessive_base = (
                    self._noun_possessive_base_for_surface(current_norm)
                    if current_norm
                    else None
                )
                current_is_possessive = bool(
                    current_possessive_base
                    and len(self._letter_tokens(current_possessive_base)) >= 2
                )
                if (
                    not current_norm
                    or (
                        not current_is_possessive
                        and (
                            current_norm not in self.dictionary_set
                            or first_repair in self.dictionary_set
                        )
                    )
                ):
                    analysis.corrected = self._match_capitalisation(
                        word,
                        verified_basic_repairs[0],
                    )
                    analysis.is_deterministic = True

        # A missing għ can be the only error in a short verb form (alaqt ->
        # għalaqt), so do this small exact lookup even when the generic cue
        # detector finds nothing.  The candidate remains dictionary-gated.
        orthographic = getattr(self, "orthographic_generator", None)
        if (
            normalized not in self.dictionary_set
            and orthographic is not None
            and len(self._letter_tokens(normalized)) <= 18
        ):
            for candidate in orthographic.insert_token_next_to_vowels(
                normalized,
                "għ",
            ):
                if candidate in self.dictionary_set or self._is_verb_tagged_word(candidate):
                    add_unique(basic_candidates, candidate)

        for candidate in self._strict_lookup_variants(normalized):
            add_unique(basic_candidates, candidate)
            if (
                candidate != normalized
                and (
                    candidate in self.dictionary_set
                    or self._is_recognized_surface(candidate)
                    or self._is_verb_tagged_word(candidate)
                )
                and not priority_basic_locked
                and (
                    priority_basic_candidate is None
                    or self._word_distance(normalized, candidate)
                    <= self._word_distance(normalized, priority_basic_candidate)
                )
            ):
                priority_basic_candidate = candidate
        final_accent = self._dictionary_final_vowel_accent(normalized)
        if final_accent:
            add_unique(basic_candidates, final_accent)
        guttural_owm = self._guttural_owm_variant(normalized)
        if guttural_owm:
            add_unique(basic_candidates, guttural_owm)
        for candidate in self._initial_i_variants(normalized):
            add_unique(basic_candidates, candidate)
        # Initial n/m is a phonological assimilation: ``nmut`` and
        # ``inmut`` may be the i-/non-i surfaces of a genuine mm- verb.
        # The result remains dictionary/paradigm-gated before Phase Y uses it.
        nm_variants: list[str] = []
        if normalized.startswith("nm"):
            nm_variants.append("mm" + normalized[2:])
        elif normalized.startswith("inm"):
            nm_variants.append("imm" + normalized[3:])
        if "nb" in normalized:
            nm_variants.append(normalized.replace("nb", "mb", 1))
        for candidate in nm_variants:
            add_unique(basic_candidates, candidate)
            analysis.corrected = self._match_capitalisation(word, candidate)
            analysis.is_deterministic = True
        initial_i_imperfect = self._initial_i_imperfect_spelling_repair(normalized)
        if initial_i_imperfect is not None:
            add_unique(basic_candidates, initial_i_imperfect)
        initial_i_form7 = self._initial_i_form7_surface_repair(normalized)
        if initial_i_form7 is not None:
            add_unique(basic_candidates, initial_i_form7)
        for candidate in self._suffix_repair_variants(normalized):
            add_unique(basic_candidates, candidate)
        orthographic = getattr(self, "orthographic_generator", None)
        if orthographic is not None:
            for candidate in orthographic.shortcut_letter_variants(
                normalized,
                max_changes=1,
                max_variants=16,
            ):
                if (
                    candidate != normalized
                    and (
                        candidate in self.dictionary_set
                        or self._is_recognized_surface(candidate)
                        or self._is_verb_tagged_word(candidate)
                    )
                ):
                    add_unique(basic_candidates, candidate)
                    if (
                        not priority_basic_locked
                        and (
                        priority_basic_candidate is None
                        or self._word_distance(normalized, candidate)
                        < self._word_distance(normalized, priority_basic_candidate)
                        )
                    ):
                        priority_basic_candidate = candidate
                    break
            for candidate in orthographic.substitute_i_ie(normalized):
                if (
                    candidate != normalized
                    and (
                        candidate in self.dictionary_set
                        or self._is_recognized_surface(candidate)
                        or self._is_verb_tagged_word(candidate)
                    )
                ):
                    add_unique(basic_candidates, candidate)
                    if (
                        not priority_basic_locked
                        and (
                        priority_basic_candidate is None
                        or self._word_distance(normalized, candidate)
                        < self._word_distance(normalized, priority_basic_candidate)
                        )
                    ):
                        priority_basic_candidate = candidate
                    break
        # F8 verbs with a final -a elide that vowel before the direct-object
        # h suffix (xtra + h -> xtrah). Treat the compact surface as a valid
        # generated form before the generic suffix generator restores -a.
        if normalized.endswith("h") and len(normalized) > 3:
            compact_base = normalized[:-1] + "a"
            if any("-F8-" in tag for tag in self.word_tags.get(compact_base, ())):
                add_unique(basic_candidates, normalized)
                analysis.corrected = self._match_capitalisation(word, normalized)
                analysis.is_deterministic = True
        suffix_generator = getattr(self, "suffix_generator", None)
        if suffix_generator is not None:
            suffix_surface = suffix_generator.correct_suffix(normalized)
            if (
                suffix_surface is not None
                and self._suffix_anchor_compatible(normalized, suffix_surface)
            ):
                add_unique(basic_candidates, suffix_surface)
                if suffix_surface != normalized and priority_basic_candidate is None:
                    priority_basic_candidate = suffix_surface
        direct_object_accent = self._direct_object_accent_surface(normalized)
        if direct_object_accent is not None:
            add_unique(basic_candidates, direct_object_accent)

        # Dictionary-verified doubled consonants are a basic orthographic
        # repair. Record them in Phase X so rich-text phrase paths and direct
        # correction share the same deterministic result.
        article_prefixes = {
            "il", "l", "tal", "mal", "bil", "fil", "fis", "fir",
            "mil", "mid", "mill", "lil", "lill", "sal",
            "ghal", "għal", "ghall", "għall", "ghat", "għat",
        }
        if (
            normalized not in self.dictionary_set
            and normalized not in article_prefixes
            and len(self._letter_tokens(normalized)) >= 4
            and hasattr(self, "doubled_letter_generator")
        ):
            doubled = self.doubled_letter_generator.correct_missing_double(normalized)
            if doubled is not None:
                add_unique(basic_candidates, doubled)

        # Compose the common basic repairs in their phonological order. Each
        # combination is admitted only when it is an exact dictionary form:
        # wicna -> wiċna -> wiċċna; hemmek -> hemmekk -> hemmhekk.
        doubled_generator = getattr(self, "doubled_letter_generator", None)
        if (
            normalized not in self.dictionary_set
            and normalized not in article_prefixes
            and orthographic is not None
            and doubled_generator is not None
        ):
            seeds = [normalized]
            seeds.extend(
                orthographic.shortcut_letter_variants(
                    normalized,
                    max_changes=1,
                    max_variants=16,
                )
            )
            for seed in seeds:
                # Preserve an exact one-operation shortcut repair before
                # considering any compound follow-up transformation.
                if is_exact_y_surface(seed):
                    add_unique(basic_candidates, seed)
                for doubled_seed in doubled_generator.missing_double_variants(seed):
                    if is_exact_y_surface(doubled_seed):
                        add_unique(basic_candidates, doubled_seed)

                # A missing h can occur before a vowel, then a following
                # consonant may still need doubling (hemmek -> hemmhekk).
                # Both operations remain dictionary-gated.
                for h_seed in orthographic.insert_token_next_to_vowels(seed, "h"):
                    if is_exact_y_surface(h_seed):
                        add_unique(basic_candidates, h_seed)
                    for doubled_seed in doubled_generator.missing_double_variants(h_seed):
                        if is_exact_y_surface(doubled_seed):
                            add_unique(basic_candidates, doubled_seed)

        compact_long = self._expand_compact_long_preposition_phrase(normalized)
        if compact_long is not None:
            add_unique(complex_candidates, compact_long)

        compact_xi = self._expand_compact_xi_article(normalized)
        if compact_xi is not None:
            add_unique(complex_candidates, compact_xi)

        attached_l = self._attached_l_apostrophe_repair(normalized)
        if attached_l is not None:
            add_unique(complex_candidates, attached_l)

        repaired_x = self._repair_x_apostrophe_word(word)
        if repaired_x is not None:
            add_unique(complex_candidates, repaired_x)

        apostrophe_prefix = self._valid_apostrophe_prefix_word(normalized)
        if apostrophe_prefix is not None:
            add_unique(complex_candidates, apostrophe_prefix)

        initial_u_w = self._initial_u_to_w_repair(normalized)
        if initial_u_w is not None:
            add_unique(basic_candidates, initial_u_w)

        # The aj/ej verb-family rules encode a targeted missing-guttural
        # pattern. Keep that result ahead of a later generic doubling repair
        # such as tajar -> tajjar.
        if pattern_priority is not None:
            analysis.corrected = self._match_capitalisation(word, pattern_priority)
            analysis.is_deterministic = True

        if priority_basic_candidate is not None:
            analysis.corrected = self._match_capitalisation(
                word,
                priority_basic_candidate,
            )
            analysis.is_deterministic = True

        # Direct dictionary matches are the baseline correction. Later basic
        # branches may still contribute alternatives, but they must not rewrite
        # a valid lexical surface merely because an i/ie or suffix variant also
        # exists (qisu -> qiesu was one such regression).
        if (
            normalized in self.dictionary_set
            and normalized not in self.MANUAL_WORD_REPAIRS
            and normalized not in self.SOCIAL_COMMENT_REPAIRS
            and not self._ta_direct_object_gloss(normalized)
            and not normalized.startswith("taj")
        ):
            analysis.corrected = self._match_capitalisation(word, normalized)
            analysis.is_deterministic = True

        x_candidates = []
        for bucket in (basic_candidates, complex_candidates):
            for candidate in bucket:
                if candidate not in x_candidates:
                    x_candidates.append(candidate)

        # Prefer a tagged lexical entry over an untagged generated surface
        # where the orthographic cost is equal (for example ``idea`` before
        # the possessive surface ``idejja`` for the input ``ideja``).
        basic_candidates.sort(
            key=lambda c: (
                # Preserve a typed initial h when another dictionary-gated
                # repair can explain the word.  This keeps a structural
                # h+double repair (hemmek -> hemmhekk) ahead of an unrelated
                # first-letter h -> ħ substitution.
                1
                if len(self._letter_tokens(normalized)) >= 5
                and normalized.startswith("h")
                and c.startswith("ħ")
                else 0,
                self._word_distance(normalized, c),
                0 if self.word_tags.get(c) else 1,
                len(c),
            )
        )

        analysis.basic_candidates = tuple(basic_candidates)
        analysis.complex_candidates = tuple(complex_candidates)
        analysis.x_candidates = tuple(x_candidates)
        if x_candidates:
            analysis.candidates = tuple(x_candidates)
        return analysis

    def _vowel_digraph_repair_variants(self, word: str) -> list[str]:
        """Return narrow, reversible vowel-digraph spelling variants.

        These are spelling operations rather than lexical exceptions.  They
        are admitted only after a dictionary/paradigm lookup by the caller.
        """
        normalized = self._normalize_word(word)
        variants: list[str] = []

        def add(candidate: str) -> None:
            candidate = self._normalize_word(candidate)
            if candidate and candidate != normalized and candidate not in variants:
                variants.append(candidate)

        for index in range(max(0, len(normalized) - 1)):
            if normalized.startswith(("aj", "ej"), index):
                add(normalized[:index] + "għi" + normalized[index + 2 :])
            if normalized.startswith("eja", index):
                add(normalized[:index] + "ea" + normalized[index + 3 :])
            if normalized.startswith("ea", index):
                add(normalized[:index] + "eja" + normalized[index + 2 :])
            if (
                normalized[index:index + 1] == "i"
                and index + 1 < len(normalized)
                and normalized[index + 1] in self.VOWELS
            ):
                add(normalized[: index + 1] + "j" + normalized[index + 1 :])
            # A typed semivowel can stand for the short i of a verb surface.
            # Restrict this to a two-letter vowel sequence so ordinary y-like
            # consonants are never rewritten.
            if normalized.startswith(("ey", "ay"), index):
                add(normalized[:index] + "i" + normalized[index + 2 :])

        return variants

    def _has_basic_repair_cue(self, word: str) -> bool:
        normalized = self._normalize_word(word)
        graphemes = self._graphemes(normalized)
        return bool(
            "gh" in normalized
            or any(letter in normalized for letter in ("h", "c", "z", "g"))
            or any(sequence in normalized for sequence in ("eja", "ea", "ey", "ay", "io"))
            # A short unrecognised CVC... surface can be missing a silent għ
            # even when it contains no keyboard-shortcut character
            # (minajr -> mingħajr).  Candidate admission remains exact-
            # dictionary-gated in Phase Y.
            or bool(re.search(r"n[aeiouàèìòù]", normalized))
            or any(
                left == right and left not in self.VOWELS
                for left, right in zip(graphemes, graphemes[1:])
            )
        )

    def _dictionary_verified_basic_repairs(self, word: str) -> list[str]:
        """Return bounded, dictionary-verified basic repair compositions."""
        normalized = self._normalize_word(word)
        if (
            not normalized
            or normalized in self.dictionary_set
            or len(self._letter_tokens(normalized)) > 18
        ):
            return []

        orthographic = getattr(self, "orthographic_generator", None)
        doubled = getattr(self, "doubled_letter_generator", None)
        accepted: list[str] = []
        seeds: list[str] = [normalized]
        seen: set[str] = {normalized}

        def is_valid(candidate: str) -> bool:
            return bool(
                candidate in self.dictionary_set
                or self._is_verb_tagged_word(candidate)
            )

        def add_seed(candidate: str) -> None:
            candidate = self._normalize_word(candidate)
            if not candidate or candidate in seen or len(seen) >= 64:
                return
            seen.add(candidate)
            seeds.append(candidate)
            for lookup in self._strict_lookup_variants(candidate):
                lookup = self._normalize_word(lookup)
                if is_valid(lookup):
                    accepted.append(lookup)

        for candidate in self._strict_lookup_variants(normalized)[1:]:
            add_seed(candidate)
        for candidate in self._vowel_digraph_repair_variants(normalized):
            add_seed(candidate)

        if orthographic is not None:
            for candidate in orthographic.shortcut_letter_variants(
                normalized,
                max_changes=2,
                max_variants=12,
            ):
                add_seed(candidate)
            for candidate in orthographic.substitute_i_ie(normalized):
                add_seed(candidate)
            for candidate in orthographic.insert_token_next_to_vowels(normalized, "għ"):
                add_seed(candidate)
            for candidate in orthographic.insert_token_next_to_vowels(normalized, "h"):
                add_seed(candidate)

            # Apply a second, bounded surface pass after de-gemination.  This
            # keeps ordinary compound typos compositional: kullhadd ->
            # kulhadd -> kulħadd, and jerga -> jerġa -> jerġa'.
            for candidate in orthographic.remove_extra_double_variants(
                normalized
            ):
                add_seed(candidate)
            for seed in tuple(seeds):
                for candidate in orthographic.shortcut_letter_variants(
                    seed,
                    max_changes=2,
                    max_variants=16,
                ):
                    add_seed(candidate)
                if seed and self._word_ends_with_vowel(seed):
                    add_seed(seed + "'")
                final_accent = self._dictionary_final_vowel_accent(seed)
                if final_accent:
                    add_seed(final_accent)

        # A one-operation dictionary hit is the intended result in almost all
        # cases.  Keep exploring only where a typed ey/ay sequence can still
        # contract before a final consonant-doubling repair.
        if accepted and not any(sequence in normalized for sequence in ("ey", "ay")):
            return list(dict.fromkeys(accepted))

        # One follow-up operation is enough for the intended compound cases:
        # għeyni -> għini -> għinni, and hemmek -> hemmhekk.
        for seed in tuple(seeds):
            follow_ups = self._vowel_digraph_repair_variants(seed)
            if doubled is not None and len(self._letter_tokens(seed)) >= 4:
                follow_ups.extend(doubled.missing_double_variants(seed))
            if orthographic is not None and "għ" in seed:
                follow_ups.extend(orthographic.insert_token_next_to_vowels(seed, "h"))
            if orthographic is not None:
                # A typed plain h can be silent. Removing it is safe only
                # when the resulting form (or its ordinary double reduction)
                # is an exact lexical or verb surface.
                follow_ups.extend(orthographic.remove_token(seed, "h"))
            for candidate in follow_ups:
                candidate = self._normalize_word(candidate)
                if not candidate or candidate in seen or len(seen) >= 96:
                    continue
                seen.add(candidate)
                if is_valid(candidate):
                    accepted.append(candidate)
                elif orthographic is not None:
                    for reduced in orthographic.remove_extra_double_variants(candidate):
                        if is_valid(reduced):
                            accepted.append(reduced)

        # A final doubling after a vowel-digraph repair covers short verb
        # surfaces without recursively expanding every spelling path.
        if doubled is not None and len(self._letter_tokens(normalized)) >= 4:
            for seed in tuple(seeds):
                for vowel_seed in self._vowel_digraph_repair_variants(seed):
                    for candidate in doubled.missing_double_variants(vowel_seed):
                        candidate = self._normalize_word(candidate)
                        if candidate and is_valid(candidate):
                            accepted.append(candidate)

        return list(dict.fromkeys(accepted))

    def _direct_object_accent_surface(self, word: str) -> str | None:
        """Return the canonical DO-suffixed surface for an obsolete ie form."""
        normalized = self._normalize_word(word)
        if "ie" not in normalized:
            return None
        suffix_generator = getattr(self, "suffix_generator", None)
        if suffix_generator is None:
            return None
        candidate = suffix_generator.correct_suffix(normalized)
        if candidate and candidate != normalized and "ie" not in candidate:
            return candidate
        return None

    def _phase_y_basic_resolution(
        self,
        word: str,
        analysis: TokenAnalysis,
    ) -> str | None:
        normalized = analysis.normalized or self._normalize_word(word)
        if not normalized:
            return None
        if analysis.is_deterministic:
            return analysis.corrected or word
        repaired_i = self._initial_i_surface_repair(normalized)
        if repaired_i and repaired_i != normalized:
            repaired = self._match_capitalisation(word, repaired_i)
            if word[:1].isupper():
                repaired = self._capitalize_first_letter(repaired)
            return repaired
        # An input i- surface can be the context-sensitive counterpart of a
        # dictionary ji- imperfect.  Keep it intact here; Phase Z selects i-
        # or ji- from the preceding corrected word.
        if (
            normalized.startswith("i")
            and self._is_verb_tagged_word(f"j{normalized[1:]}")
        ):
            return word
        if normalized in self.dictionary_set:
            return word

        def double_signature(surface: str) -> tuple[tuple[int, str], ...]:
            simplified = self._normalize_word(surface).translate(
                str.maketrans({"ġ": "g", "ħ": "h", "ċ": "c", "ż": "z"})
            )
            graphemes = self._graphemes(simplified)
            return tuple(
                (index, token)
                for index, token in enumerate(graphemes[:-1])
                if token == graphemes[index + 1] and token not in self.VOWELS
            )

        direct_object_accent = self._direct_object_accent_surface(normalized)
        valid_basic_candidates = []
        for candidate in analysis.basic_candidates:
            if candidate == normalized:
                possessive_base = self._noun_possessive_base_for_surface(normalized)
                if possessive_base and len(self._letter_tokens(possessive_base)) >= 2:
                    return self._match_capitalisation(word, candidate)
                continue
            if (
                candidate in self.dictionary_set
                or self._is_verb_tagged_word(candidate)
                or self._is_recognized_surface(candidate)
                or self._correct_noun_possessive_suffix(candidate) == candidate
                or candidate == direct_object_accent
            ):
                valid_basic_candidates.append(candidate)

        source_doubles = double_signature(normalized)
        if source_doubles:
            valid_basic_candidates.sort(
                key=lambda candidate: (
                    0 if double_signature(candidate) == source_doubles else 1,
                    analysis.basic_candidates.index(candidate),
                )
            )
        if valid_basic_candidates:
            return self._match_capitalisation(word, valid_basic_candidates[0])

        for candidate in analysis.complex_candidates:
            if candidate == normalized:
                return self._match_capitalisation(word, candidate)
            if candidate in self.dictionary_set or self._is_verb_tagged_word(candidate):
                return self._match_capitalisation(word, candidate)

        if normalized in self.dictionary_set:
            return word

        return None

    def _phase_w_seed_suggestions(
        self,
        word: str,
        analysis: TokenAnalysis,
        *,
        limit: int,
    ) -> list[str]:
        seeded: list[str] = []
        seen: set[str] = set()
        source_norm = analysis.normalized or self._normalize_word(word)

        def add(candidate: str) -> None:
            candidate = self._normalize_word(candidate)
            if (
                candidate.endswith("ħek")
                and not any(marker in source_norm for marker in ("h", "ħ", "għ", "gh"))
            ):
                return
            if (
                source_norm in self.dictionary_set
                and source_norm.endswith("u")
                and candidate == source_norm[:-1] + "għu"
            ):
                return
            if candidate and candidate not in seen:
                seen.add(candidate)
                seeded.append(candidate)

        add(analysis.corrected or analysis.normalized)
        for candidate in analysis.x_candidates:
            add(candidate)
        # Phase X keeps the canonical lexical form as its deterministic
        # correction. Phase W is where the explicitly listed equivalent
        # spellings belong, so expose them alongside that correction.
        for candidate in self._lexicalized_form_variants(analysis.normalized):
            add(candidate)
        if len(seeded) < limit:
            for candidate in self._deterministic_suggestion_variants(
                word,
                analysis,
                limit=limit,
            ):
                add(candidate)
                if len(seeded) >= limit:
                    break
        return seeded[:limit]

    def _phase_z_finalize_surface_word(
        self,
        original_word: str,
        corrected_word: str,
        *,
        previous_surface_word: str | None,
        sentence_initial: bool,
        prefer_initial_vowel_surface: bool,
    ) -> tuple[str, str]:
        final_ghat = self._final_ghat_to_ghat_e_repair(corrected_word)
        if final_ghat is not None:
            corrected_word = final_ghat

        participial_initial_i = self._initial_i_participial_adjective_surface(
            corrected_word
        )
        if sentence_initial and participial_initial_i is not None:
            surface_word = participial_initial_i
        else:
            vowel_options, plain_options = self._initial_vowel_surface_options(
                corrected_word
            )
            surface_word = self._apply_initial_vowel_surface(
                original_word,
                corrected_word,
                prefer_vowel_surface=(
                    not self._word_ends_with_vowel(previous_surface_word)
                    if previous_surface_word is not None
                    and (
                        self._is_form7_perf_or_imp_surface(corrected_word)
                        or vowel_options
                        or plain_options
                    )
                    else prefer_initial_vowel_surface
                ),
            )
        if sentence_initial:
            surface_word = self._capitalize_first_letter(surface_word)
            corrected_word = self._capitalize_first_letter(corrected_word)
        return surface_word, corrected_word

    def _feminine_imperfect_continuation(
        self,
        original_word: str,
        corrected_word: str,
        previous_surface_word: str | None,
    ) -> str:
        """Prefer the 3SF imperfect after an immediately preceding 3SF perfect."""
        original = self._normalize_word(original_word)
        corrected = self._normalize_word(corrected_word)
        previous = self._normalize_word(previous_surface_word or "")
        if not (original.startswith("j") and corrected.startswith("j") and previous):
            return corrected_word

        previous_records = self._verb_records_for_surface(previous)
        if not any(
            record.person == "3SF" and record.tense == "PERF"
            for record in previous_records
        ):
            return corrected_word

        feminine = f"t{corrected[1:]}"
        if any(
            record.person == "3SF" and record.tense == "MPERF"
            for record in self._verb_records_for_surface(feminine)
        ):
            return self._match_capitalisation(corrected_word, feminine)
        return corrected_word

    def _cached_correct_word(self, word: str) -> str | None:
        analysis = self._get_token_analysis(word)
        if analysis and analysis.corrected:
            return analysis.corrected
        return None

    def _store_correct_word_result(
        self,
        word: str,
        corrected: str,
        *,
        is_deterministic: bool = False,
        candidates: Iterable[str] | None = None,
    ) -> str:
        normalized_orig = self._normalize_word(word)
        normalized_corr = self._normalize_word(corrected)
        if normalized_orig and normalized_corr and " " not in normalized_corr:
            if normalized_orig.startswith(("i", "u")) and normalized_corr.startswith(("j", "w")):
                prefer_vowel, prefer_plain = self._initial_vowel_surface_options(normalized_corr)
                matching_vowel = next(
                    (
                        option
                        for option in prefer_vowel
                        if self._normalize_word(option) == normalized_orig
                    ),
                    None,
                )
                if matching_vowel:
                    corrected = self._match_capitalisation(word, matching_vowel)
            elif normalized_orig.startswith(("j", "w")) and normalized_corr.startswith(("i", "u")):
                prefer_vowel, prefer_plain = self._initial_vowel_surface_options(normalized_corr)
                if prefer_plain:
                    corrected = self._match_capitalisation(word, prefer_plain[0])

        candidate_list = list(candidates or ())
        if not candidate_list:
            candidate_list = [corrected]
        self._store_token_analysis(
            word,
            corrected=corrected,
            candidates=candidate_list,
            is_deterministic=is_deterministic,
        )
        return corrected

    def _store_suggest_result(
        self,
        word: str,
        limit: int,
        edit_distance_tolerance: int,
        suggestions: list[str],
    ) -> list[str]:
        normalized = self._normalize_word(word)
        result = tuple(suggestions[:limit])
        self._request_suggestion_cache()[(word, limit, edit_distance_tolerance)] = result
        analysis = self._get_token_analysis(word)
        if analysis is None:
            self._store_token_analysis(
                word,
                corrected=result[0] if result else "",
                candidates=result,
                is_deterministic=False,
            )
        else:
            if result and not analysis.corrected:
                analysis.corrected = result[0]
            if result and not analysis.candidates:
                analysis.candidates = result
            if not analysis.normalized:
                analysis.normalized = normalized
        return list(result)

    def _deterministic_suggestion_variants(
        self,
        word: str,
        analysis: TokenAnalysis,
        *,
        limit: int,
    ) -> list[str]:
        normalized = self._normalize_word(word)
        ordered: list[str] = []
        seen_canonical: set[str] = set()

        def add(candidate: str) -> None:
            candidate = self._normalize_word(candidate)
            canonical = self._canonical_suggestion_key(candidate)
            # A valid final -u form must not acquire a speculative final għ
            # merely because a different paradigm happens to contain -għu.
            if (
                normalized in self.dictionary_set
                and normalized.endswith("u")
                and candidate == normalized[:-1] + "għu"
            ):
                return
            if (
                candidate
                and candidate not in ordered
                and canonical not in seen_canonical
            ):
                ordered.append(candidate)
                seen_canonical.add(canonical)

        add(analysis.corrected)
        for candidate in analysis.candidates:
            add(candidate)

        for candidate in self._medial_guttural_vowel_restoration_variants(word):
            add(candidate)
            if len(ordered) >= limit:
                return ordered[:limit]

        for candidate in self._missing_medial_guttural_variants(word):
            add(candidate)
            if len(ordered) >= limit:
                return ordered[:limit]

        for candidate in self._missing_h_before_r_verb_variants(word):
            add(candidate)
            if len(ordered) >= limit:
                return ordered[:limit]

        orthographic_generator = getattr(self, "orthographic_generator", None)
        if orthographic_generator is not None:
            sources = [word, analysis.corrected]

            for method_name in (
                "dictionary_gh_suggestion_variants",
                "dictionary_shortcut_variants",
                "dictionary_d_t_variants",
                "dictionary_b_p_variants",
                "dictionary_i_ie_variants",
                "dictionary_final_gh_h_hbar_variants",
            ):
                if not hasattr(orthographic_generator, method_name):
                    continue
                generator = getattr(orthographic_generator, method_name)
                for source in sources:
                    for candidate in generator(source):
                        add(candidate)
                        if len(ordered) >= limit:
                            return ordered[:limit]

            if hasattr(
                orthographic_generator,
                "move_gh_right_across_adjacent_vowel",
            ):
                suffix_generator = getattr(self, "suffix_generator", None)
                for source in sources:
                    for candidate in orthographic_generator.move_gh_right_across_adjacent_vowel(
                        self._normalize_word(source)
                    ):
                        if candidate in self.dictionary_set:
                            add(candidate)
                        elif (
                            suffix_generator is not None
                            and self._valid_suffix_surface_candidates(candidate)
                        ):
                            add(candidate)
                        if len(ordered) >= limit:
                            return ordered[:limit]

        return ordered[:limit]

    def _valid_suffix_surface_candidates(self, word: str):
        suffix_generator = getattr(self, "suffix_generator", None)
        if suffix_generator is None or not hasattr(
            suffix_generator,
            "candidates_for_surface",
        ):
            return []

        candidates = suffix_generator.candidates_for_surface(word)
        if not candidates:
            return []

        return [
            candidate
            for candidate in candidates
            if not is_invalid_imperative_suffix_combination(
                tense=getattr(candidate, "tense", ""),
                person=getattr(candidate, "person", ""),
                suffix_kind=getattr(candidate, "suffix_kind", ""),
                suffix_person=getattr(candidate, "suffix_person", ""),
            )
        ]

    def _cache_summary(self) -> dict[str, dict[str, int | None]]:
        names = (
            "_normalize_word_cached",
            "_graphemes_cached",
            "_extract_consonant_anchor",
            "_vowel_slots",
            "_count_vowels",
            "_damerau_levenshtein_distance",
            "_word_distance",
            "_get_candidates_cached",
            "_letter_tokens_raw",
            "meaning_for",
        )
        summary: dict[str, dict[str, int | None]] = {}
        for name in names:
            fn = getattr(type(self), name, None) or getattr(self, name, None)
            if fn is not None and hasattr(fn, "cache_info"):
                info = fn.cache_info()
                summary[name] = {
                    "hits": info.hits,
                    "misses": info.misses,
                    "maxsize": info.maxsize,
                    "currsize": info.currsize,
                }
        return summary

    def clear_disposable_startup_caches(self) -> None:
        before = self._cache_summary()
        type(self)._normalize_word_cached.cache_clear()
        type(self)._graphemes_cached.cache_clear()
        after = self._cache_summary()
        log_spellcheck_event(
            event="SPELLCHECK_CACHE_CLEAR",
            instance_id=self.instance_id,
            before=before,
            after=after,
        )

    def _build_word_metadata(self) -> None:
        for word in self.dictionary:
            tokens = self._letter_tokens_raw(word)
            self.word_lengths[word] = len(tokens)
            self.word_vowel_counts[word] = sum(1 for t in tokens if t in self.VOWELS)
            self.word_anchors[word] = self._extract_consonant_anchor_from_tokens(tokens)

    def _build_anchor_index(self) -> None:
        for word in self.dictionary:
            anchor = self.word_anchors[word]
            self.anchor_map[anchor].append(word)
        self.anchor_letters = tuple(
            sorted({char for anchor in self.anchor_map for char in anchor})
        )

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
        if self._ta_direct_object_gloss(normalized):
            return True
        if self._direct_object_h_base(normalized) is not None:
            return True
        tags = self.word_tags.get(normalized, set())
        if any(tag.startswith(("T-", "Q-", "S-", "AS-", "IS-", "VERB")) for tag in tags):
            return True

        suffix_generator = getattr(self, "suffix_generator", None)
        verb_index = getattr(suffix_generator, "verb_index", None)
        if verb_index is not None and verb_index.word_records(normalized):
            return True
        if (
            suffix_generator is None
            or not hasattr(suffix_generator, "might_have_suffix")
            or not suffix_generator.might_have_suffix(normalized)
        ):
            return False
        return bool(
            suffix_generator.exact_suffix_matches(normalized)
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

    def _verb_uses_initial_i_prefix(self, word: str) -> bool:
        normalized = self._normalize_word(word)
        if not normalized or normalized.startswith("i"):
            return False

        records = self._verb_records_for_surface(normalized)
        if not records:
            return False

        graphemes = self._graphemes(normalized)
        if len(graphemes) < 2:
            return False

        first = graphemes[0]
        second = graphemes[1]
        if len(first) != 1 or len(second) != 1:
            return False

        if first == "r" and second == "ġ":
            return False

        if first == second:
            return first not in self.VOWELS

        return first in {"n", "l", "m", "r"} and second not in self.VOWELS

    def _initial_i_surface_repair(self, word: str) -> str | None:
        normalized = self._normalize_word(word)
        if not normalized:
            return None

        if normalized.startswith("i"):
            base = normalized[1:]
            if not base:
                return None

            if base.startswith("rġ") and self._verb_records_for_surface(base):
                return "e" + base

            if self._is_form7_perf_or_imp_surface(base):
                return base

            base_tags = self.word_tags.get(base, set())
            base_graphemes = self._graphemes(base)
            if (
                len(base_graphemes) >= 2
                and base_graphemes[0] == base_graphemes[1]
                and any(tag.startswith("PASTPAR") for tag in base_tags)
            ):
                return base

            if self._verb_uses_initial_i_prefix(base):
                return None

            if self._verb_records_for_surface(base):
                if not self._is_recognized_surface(normalized):
                    return base
            return None

        if normalized.startswith("rġ") and self._verb_records_for_surface(normalized):
            return "e" + normalized

        if self._verb_uses_initial_i_prefix(normalized):
            return "i" + normalized

        return None

    def _initial_i_participial_adjective_surface(self, word: str) -> str | None:
        """Return sentence-initial i- only for recognised adjectival mC forms."""
        normalized = self._normalize_word(word)
        if (
            len(normalized) < 2
            or normalized.startswith("i")
            or normalized[0] != "m"
            or normalized[1] in self.VOWELS
        ):
            return None

        tags = self.word_tags.get(normalized, set())
        if not any(
            tag.split("-", 1)[0].startswith("ADJ")
            for tag in tags
        ):
            return None
        return f"i{normalized}"

    def _initial_i_j_surface_repair(self, word: str) -> str | None:
        normalized = self._normalize_word(word)
        if len(normalized) < 3 or normalized[0] not in {"i", "j"}:
            return None

        rest = normalized[1:]
        if len(rest) < 2:
            return None
        if rest[0] in self.VOWELS or rest[1] not in self.VOWELS:
            return None

        swapped = ("j" if normalized[0] == "i" else "i") + rest
        if swapped == normalized:
            return None
        return swapped

    def _validated_initial_e_match(self, word: str) -> str | None:
        normalized = self._normalize_word(word)
        if not normalized.startswith("e") or len(normalized) < 3:
            return None

        base = normalized[1:]
        if base.startswith("rġ") and self._verb_records_for_surface(base):
            return normalized
        return None

    def _initial_i_variants(self, word: str) -> list[str]:
        normalized = self._normalize_word(word)
        variants: list[str] = []

        def add(candidate: str) -> None:
            candidate = self._normalize_word(candidate)
            if candidate and candidate != normalized and candidate not in variants:
                variants.append(candidate)

        repaired_switch = self._initial_i_j_surface_repair(normalized)
        if repaired_switch and repaired_switch != normalized:
            add(repaired_switch)

        repaired = self._initial_i_surface_repair(normalized)
        if repaired and repaired != normalized:
            add(repaired)

        return variants

    def _validated_initial_i_match(self, word: str) -> str | None:
        return self._initial_i_surface_repair(word)

    def _is_probable_noun(self, word: str) -> bool:
        normalized = self._normalize_word(word)
        if not normalized:
            return False

        tags = self.word_tags.get(normalized)
        if not tags:
            # An untagged generated candidate is not evidence that the word
            # is nominal. Treat it as a noun only when it is a real lexical
            # entry; otherwise article and clitic branches can consume a
            # distant verb correction (for example f sormok -> f'isromok).
            return normalized in self.dictionary_set

        return not any(
            tag.startswith(("T-", "Q-", "S-", "AS-", "IS-")) for tag in tags
        )

    @lru_cache(maxsize=4096)
    def _is_noun_tagged_word(self, word: str) -> bool:
        normalized = self._normalize_word(word)
        tags = self.word_tags.get(normalized, set())
        return any("NOUN" in tag.split("-", 1)[0] for tag in tags)

    @lru_cache(maxsize=4096)
    def _is_dual_noun(self, word: str) -> bool:
        normalized = self._normalize_word(word)
        return any(
            tag.split("-", 1)[0] == "DUALNOUN"
            for tag in self.word_tags.get(normalized, set())
        )

    @lru_cache(maxsize=4096)
    def _is_pronoun_tagged_word(self, word: str) -> bool:
        normalized = self._normalize_word(word)
        tags = self.word_tags.get(normalized, set())
        return any(tag.startswith("PRON") for tag in tags)

    @lru_cache(maxsize=4096)
    def _is_adverb_tagged_word(self, word: str) -> bool:
        normalized = self._normalize_word(word)
        tags = self.word_tags.get(normalized, set())
        return any(
            tag.startswith(("ADVERB", "ADV-", "SHORTADVERB", "LADVERB")) for tag in tags
        )

    @lru_cache(maxsize=4096)
    def _is_preposition_tagged_word(self, word: str) -> bool:
        normalized = self._normalize_word(word)
        tags = self.word_tags.get(normalized, set())
        return any(
            tag.startswith(("PREP", "SHORTPREP", "DEFPREP", "ISHORTDEFPREP"))
            for tag in tags
        )

    def _supports_l_apostrophe_tail(self, word: str) -> bool:
        normalized = self._normalize_word(word)
        if self._is_adjective_tagged_word(normalized):
            return self._is_li_relative_adjective_allowed(normalized)
        if self._is_verb_tagged_word(normalized) or self._is_adverb_tagged_word(
            normalized
        ):
            return True
        return self._is_preposition_tagged_word(normalized) and self._starts_vowel_gh_or_h(
            normalized
        )

    def _is_li_relative_adjective_allowed(self, word: str) -> bool:
        """Whether an adjective can follow the relative contraction ``l'``."""
        normalized = self._normalize_word(word)
        tags = self.word_tags.get(normalized, set())
        if any(tag.split("-", 1)[0].startswith("EXC") for tag in tags):
            return True

        article_rules = getattr(self, "article_phrase_rules", None)
        if article_rules is None:
            return False
        return bool(
            article_rules._superlative_meaning(meaning_index.meaning_for(normalized))
        )

    def _safe_contracted_fb_tail(self, tail: str) -> str | None:
        normalized_tail = self._normalize_word(tail)
        if (
            not normalized_tail
            or normalized_tail in {"l", "il"}
            or normalized_tail.startswith(("l-", "il-"))
        ):
            return None

        if (
            normalized_tail in self.dictionary_set
            or bool(self.word_tags.get(normalized_tail))
            or bool(meaning_index.meaning_for(normalized_tail))
            or bool(self.meaning_for(normalized_tail))
        ) and not self._is_verb_or_pronoun_tagged(normalized_tail):
            return normalized_tail

        corrected_tail = self._normalize_word(self.correct_word(normalized_tail))
        if not corrected_tail:
            return None
        if corrected_tail == normalized_tail:
            return None
        if self._is_verb_or_pronoun_tagged(corrected_tail):
            return None
        if corrected_tail not in self.dictionary_set:
            return None
        if self._word_distance(normalized_tail, corrected_tail) > self._max_distance(
            normalized_tail
        ):
            return None
        return corrected_tail

    def _repair_contracted_fb_word(self, word: str) -> tuple[str, str] | None:
        normalized = self._normalize_word(word)
        if len(normalized) <= 2 or normalized[0] not in {"f", "b"} or normalized[1] != "'":
            return None

        prefix = normalized[0]
        tail = normalized[2:]
        safe_tail = self._safe_contracted_fb_tail(tail)
        if safe_tail is None:
            return None
        return (
            self._match_capitalisation(word, f"{prefix}'{safe_tail}"),
            safe_tail,
        )

    def _repair_x_apostrophe_word(self, word: str) -> str | None:
        normalized = self._normalize_word(word)
        if not normalized.startswith("x'") or len(normalized) <= 2:
            return None
        tail = normalized[2:]
        corrected_tail = self.correct_word(tail)
        corrected_tail_norm = self._normalize_word(corrected_tail)
        if not corrected_tail_norm or corrected_tail_norm == tail:
            return None
        return self._match_capitalisation(word, f"x'{corrected_tail_norm}")

    def _final_ghat_to_ghat_e_repair(self, word: str) -> str | None:
        normalized = self._normalize_word(word)
        if len(normalized) < 5 or not normalized.endswith(("ghat", "għat")):
            return None

        candidate = normalized[: -4] + "għet"
        if candidate == normalized:
            return None
        if self._valid_generated_surface(candidate):
            return candidate
        if candidate in self.dictionary_set:
            return candidate
        if self._verb_records_for_surface(candidate):
            return candidate
        return None

    def _is_verb_or_pronoun_tagged(self, word: str) -> bool:
        """True when *word* is tagged as a verb or pronoun."""
        normalized = self._normalize_word(word)
        tags = self.word_tags.get(normalized, set())
        return self._is_verb_tagged_word(normalized) or any(
            tag.startswith("PRON") for tag in tags
        )

    def _blocks_initial_stop_confusion(self, original: str, candidate: str) -> bool:
        original_norm = self._normalize_word(original)
        candidate_norm = self._normalize_word(candidate)
        if len(original_norm) < 2 or len(candidate_norm) < 2:
            return False

        if (original_norm[0], candidate_norm[0]) not in {
            ("p", "b"),
            ("b", "p"),
            ("t", "d"),
            ("d", "t"),
        }:
            return False

        original_tokens = self._letter_tokens(original_norm)
        candidate_tokens = self._letter_tokens(candidate_norm)
        if len(original_tokens) < 2 or len(candidate_tokens) < 2:
            return False

        return (
            original_tokens[1] in self.VOWELS
            and candidate_tokens[1] in self.VOWELS
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

    @lru_cache(maxsize=4096)
    def _is_feminine_noun(self, word: str) -> bool:
        normalized = self._normalize_word(word)
        tags = self.word_tags.get(normalized, set())
        if any(tag.split("-", 1)[0] == "SINGNOUNF" for tag in tags):
            return True

        # Fallback for manual words lacking dictionary tags: assume feminine if ending in 'a'
        if self.NOUN_POSSESSIVE_MODE.lower().strip() == "manual":
            if normalized in self._manual_noun_bases:
                return normalized.endswith("a")
        return False

    def _auxiliary_word_list_paths(self, *names: str) -> list[Path]:
        paths: list[Path] = []
        for name in names:
            for candidate in (
                FINAL_DICS_DIR / name,
                BASE_DIR / name,
                BASE_DIR.parent / name,
            ):
                if candidate not in paths:
                    paths.append(candidate)
        return paths

    def _load_auxiliary_word_list(self, paths: Iterable[Path]) -> frozenset[str]:
        words: list[str] = []
        seen: set[str] = set()

        for path in paths:
            if not path.exists():
                continue
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    for raw_line in handle:
                        line = raw_line.split("#", 1)[0].strip()
                        if not line:
                            continue
                        entry = line.split("/", 1)[0].strip()
                        normalized = self._normalize_word(entry)
                        if normalized and normalized not in seen:
                            seen.add(normalized)
                            words.append(normalized)
            except OSError:
                continue

        return frozenset(words)

    def _load_tagged_auxiliary_word_list(
        self,
        paths: Iterable[Path],
        allowed_tags: Iterable[str],
    ) -> frozenset[str]:
        words: list[str] = []
        seen: set[str] = set()
        allowed = {str(tag).strip().upper() for tag in allowed_tags}

        for path in paths:
            if not path.exists():
                continue
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    for raw_line in handle:
                        line = raw_line.split("#", 1)[0].strip()
                        if not line:
                            continue
                        if "/" in line:
                            entry, raw_tag = line.split("/", 1)
                        elif "NAME" in allowed:
                            # Legacy name lists predate NAME/SNAME tags. They
                            # remain authoritative for capitalised names.
                            entry, raw_tag = line, "NAME"
                        else:
                            continue
                        normalized = self._normalize_word(entry.strip())
                        tag = raw_tag.strip().split("-", 1)[0].upper()
                        if (
                            normalized
                            and tag in allowed
                            and normalized not in seen
                        ):
                            seen.add(normalized)
                            words.append(normalized)
            except OSError:
                continue

        return frozenset(words)

    def _no_possession_noun_set(self) -> frozenset[str]:
        if self._no_possession_nouns is None:
            self._no_possession_nouns = self._load_auxiliary_word_list(
                self._auxiliary_word_list_paths(
                    NO_POSSESSION_NOUNS_DIC.name,
                )
            )
        return self._no_possession_nouns

    def _protected_name_set(self) -> frozenset[str]:
        if self._protected_names is None:
            self._protected_names = self._load_auxiliary_word_list(
                self._auxiliary_word_list_paths(
                    PROTECTED_NAMES_DIC.name,
                    "protected_names.txt",
                )
            )
        return self._protected_names

    def _given_name_set(self) -> frozenset[str]:
        if self._given_names is None:
            self._given_names = self._load_tagged_auxiliary_word_list(
                self._auxiliary_word_list_paths(NAMES_DIC.name),
                {"NAME"},
            )
        return self._given_names

    def _surname_set(self) -> frozenset[str]:
        if self._surnames is None:
            self._surnames = self._load_tagged_auxiliary_word_list(
                self._auxiliary_word_list_paths(NAMES_DIC.name),
                {"SNAME"},
            )
        return self._surnames

    def _capitalized_name_kind(self, word: str) -> str | None:
        if not self._is_initial_capitalized(word):
            return None
        normalized = self._normalize_word(word)
        if normalized in self._given_name_set():
            return "NAME"
        if normalized in self._surname_set():
            return "SNAME"
        if normalized in self._calendar_proper_names:
            return "CALENDAR"
        return None

    def _capitalized_name_repair(self, word: str) -> str | None:
        """Return a conservative spelling repair for a capitalized name."""
        if not self._is_initial_capitalized(word):
            return None

        normalized = self._normalize_word(word)
        if not normalized:
            return None

        names = self._given_name_set() | self._surname_set()
        candidates: list[str] = []

        def add(candidate: str) -> None:
            candidate = self._normalize_word(candidate)
            if candidate and candidate not in candidates:
                candidates.append(candidate)

        for variant in self._strict_lookup_variants(normalized):
            add(variant)

        orthographic = getattr(self, "orthographic_generator", None)
        if orthographic is not None and hasattr(
            orthographic, "shortcut_letter_variants"
        ):
            for variant in orthographic.shortcut_letter_variants(
                normalized,
                max_changes=1,
                max_variants=16,
            ):
                add(variant)

        accents = {"a": "à", "e": "è", "i": "ì", "o": "ò", "u": "ù"}
        for variant in tuple(candidates):
            if variant[-1:] in accents:
                add(variant[:-1] + accents[variant[-1]])

        for candidate in candidates:
            if candidate in names:
                return self._capitalize_first_letter(candidate)
        return None

    def _exact_lowercase_proper_name(self, word: str) -> str | None:
        """Restore canonical initial capitals for exact lowercase proper names."""
        normalized = self._normalize_word(word)
        if not normalized or not str(word).islower():
            return None

        place_display = self._exact_place_word(normalized)
        if place_display:
            return place_display
        if normalized in self._calendar_proper_names:
            return self._capitalize_first_letter(normalized)
        if normalized in self._protected_name_set():
            return self._capitalize_first_letter(normalized)
        if normalized in self._given_name_set() or normalized in self._surname_set():
            return self._capitalize_first_letter(normalized)
        return None

    @lru_cache(maxsize=4096)
    def _noun_possessive_base_is_enabled(self, word: str) -> bool:
        normalized = self._normalize_word(word)
        mode = self.NOUN_POSSESSIVE_MODE.lower().strip()

        if normalized in self._no_possession_noun_set():
            return False

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
            # Nouns in -ieħeb use the regular possessive suffixes after the
            # weak penultimate e contracts (sieħeb -> sieħbi).
            if normalized.endswith("ieħeb"):
                add(normalized[:-2] + normalized[-1], False)

        return stems

    @lru_cache(maxsize=2048)
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

    @lru_cache(maxsize=4096)
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

            # A possessive surface can omit the weak e immediately after i
            # (for example siħbi -> sieħbi).  Restore it only as a
            # dictionary-gated candidate below.
            stems_to_consider = [stem]
            for position, letter in enumerate(stem[:-1]):
                if letter == "i" and stem[position + 1] not in self.VOWELS:
                    expanded = stem[: position + 1] + "e" + stem[position + 1 :]
                    if expanded not in stems_to_consider:
                        stems_to_consider.append(expanded)

            for candidate_stem in stems_to_consider:
                if candidate_stem.endswith("j"):
                    possible_bases.append(candidate_stem[:-1] + "'")

                if candidate_stem.endswith("jt"):
                    possible_bases.append(candidate_stem[:-2] + "jja")

                if candidate_stem.endswith("it"):
                    possible_bases.append(candidate_stem[:-2] + "a")

                if candidate_stem.endswith("t"):
                    possible_bases.append(candidate_stem[:-1] + "a")

                possible_bases.append(candidate_stem + "n")
                possible_bases.append(candidate_stem)

                if len(candidate_stem) >= 2:
                    possible_bases.append(
                        candidate_stem[:-1] + "e" + candidate_stem[-1]
                    )
                    possible_bases.append(
                        candidate_stem[:-1] + "i" + candidate_stem[-1]
                    )

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

        # Restore a weak e after i inside a possible possessive surface and
        # then re-run the keyboard-shortcut path.  Every result remains
        # dictionary- and suffix-validated below, so this cannot accept an
        # arbitrary i/e swap.
        for source in tuple(variants) + (normalized,):
            for position, letter in enumerate(source[:-1]):
                if letter != "i" or source[position + 1] in self.VOWELS:
                    continue
                expanded = source[: position + 1] + "e" + source[position + 1 :]
                variants.append(expanded)
                if orthographic_generator is not None and hasattr(
                    orthographic_generator, "shortcut_letter_variants"
                ):
                    variants.extend(
                        orthographic_generator.shortcut_letter_variants(
                            expanded,
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

    def _collapse_invalid_glide_doubling(self, word: str) -> str | None:
        normalized = self._normalize_word(word)
        graphemes = self._graphemes(normalized)
        if len(graphemes) < 3:
            return None

        candidates: list[str] = []
        for index in range(len(graphemes) - 2):
            glide = graphemes[index]
            if (
                glide not in {"j", "w"}
                or graphemes[index + 1] != glide
                or graphemes[index + 2] in self.VOWELS
            ):
                continue
            candidate = self._from_graphemes(
                graphemes[: index + 1] + graphemes[index + 2 :]
            )
            if candidate != normalized and candidate not in candidates:
                candidates.append(candidate)

        orthographic_generator = getattr(self, "orthographic_generator", None)
        for candidate in candidates:
            if (
                candidate in self.dictionary_set
                or self._noun_possessive_base_for_surface(candidate)
                or self._valid_suffix_surface_candidates(candidate)
            ):
                return candidate
            if orthographic_generator is not None and hasattr(
                orthographic_generator,
                "correct_shortcut_letters",
            ):
                shortcut = orthographic_generator.correct_shortcut_letters(candidate)
                if shortcut and (
                    shortcut in self.dictionary_set
                    or self._noun_possessive_base_for_surface(shortcut)
                    or self._valid_suffix_surface_candidates(shortcut)
                ):
                    return shortcut

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

                if "/" in line:
                    word, raw_tag = line.split("/", 1)
                    word = word.strip()
                    raw_tag = raw_tag.strip()
                    if "-" in raw_tag or raw_tag:
                        entries.append((word, raw_tag))
                    else:
                        entries.append((word, None))
                else:
                    first_field = line.split()[0]
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

    @lru_cache(maxsize=8192)
    def _extract_consonant_anchor(self, word: str) -> str:
        normalized = self._normalize_word(word)
        if normalized in self.word_anchors:
            return self.word_anchors[normalized]
        return self._extract_consonant_anchor_from_tokens(
            self._letter_tokens_raw(normalized)
        )

    @lru_cache(maxsize=4096)
    def _vowel_slots(self, word: str) -> list[tuple[int, str]]:
        normalized = self._normalize_word(word)
        tokens = self._letter_tokens_raw(normalized)
        return [(i, t) for i, t in enumerate(tokens) if t in self.VOWELS]

    @lru_cache(maxsize=4096)
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

    @lru_cache(maxsize=8192)
    def _damerau_levenshtein_distance(self, a: tuple[str, ...], b: tuple[str, ...]) -> int:
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

    @lru_cache(maxsize=8192)
    def _word_distance(self, word1: str, word2: str) -> int:
        profiler = current_profiler()
        if profiler is not None:
            profiler.increment("distance_calls")
        return self._damerau_levenshtein_distance(
            tuple(self._letter_tokens(word1)),
            tuple(self._letter_tokens(word2)),
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
            tuple(typo_anchor), tuple(candidate_anchor)
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

    def _guttural_owm_variant(self, word: str) -> str | None:
        """Restore a dictionary verb with the common owm -> għum spelling loss."""
        normalized = self._normalize_word(word)
        if len(normalized) < 4 or not normalized[0].isalpha():
            return None

        endings = (("owmu", "għumu"), ("owm", "għum"))
        for typed_ending, corrected_ending in endings:
            if not normalized.endswith(typed_ending):
                continue
            candidate = normalized[: -len(typed_ending)] + corrected_ending
            if self._is_verb_tagged_word(candidate):
                return candidate
        return None

    def _dictionary_final_vowel_accent(self, word: str) -> str | None:
        """Restore a final Maltese vowel accent only on an exact dictionary hit."""
        normalized = self._normalize_word(word)
        accents = {"a": "à", "e": "è", "i": "ì", "o": "ò", "u": "ù"}
        if not normalized or normalized[-1] not in accents:
            return None
        candidate = normalized[:-1] + accents[normalized[-1]]
        return candidate if candidate in self.dictionary_set else None

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

    def _suffix_anchor_compatible(self, source: str, candidate: str) -> bool:
        """Allow suffix reconstruction to insert silent material, not a new root."""
        source_norm = self._normalize_word(source)
        candidate_norm = self._normalize_word(candidate)
        if not source_norm or not candidate_norm:
            return False

        # Treat typed ``gh`` as the Maltese grapheme before comparing the
        # roots.  The candidate may add għ/h/ħ while reconstructing a surface,
        # but every lexical consonant in the input must still align.
        source_tokens = self._graphemes(source_norm.replace("gh", "għ"))
        candidate_tokens = self._graphemes(candidate_norm)
        source_consonants = [token for token in source_tokens if token not in self.VOWELS]
        candidate_consonants = [token for token in candidate_tokens if token not in self.VOWELS]

        index = 0
        for token in candidate_consonants:
            if index < len(source_consonants) and token == source_consonants[index]:
                index += 1
                continue
            if token in {"għ", "h", "ħ"}:
                continue
            return False
        return index == len(source_consonants)

    def _collapse_vowel_around_token_variants(
        self,
        word: str,
        token: str,
    ) -> list[str]:
        normalized = self._normalize_word(word)
        graphemes = self._graphemes(normalized)
        variants: list[str] = []

        def add(candidate_graphemes: list[str]) -> None:
            candidate = self._from_graphemes(candidate_graphemes)
            candidate = self._normalize_word(candidate)
            if candidate and candidate != normalized and candidate not in variants:
                variants.append(candidate)

        for index in range(1, len(graphemes) - 1):
            if graphemes[index] != token:
                continue
            if graphemes[index - 1] not in self.VOWELS:
                continue
            if graphemes[index + 1] not in self.VOWELS:
                continue

            add(graphemes[: index - 1] + graphemes[index:])
            add(graphemes[: index + 1] + graphemes[index + 2 :])

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

        for candidate in self._final_object_suffix_variants(normalized):
            add(candidate)

        for candidate in self._restore_direct_object_h_vowel(normalized):
            add(candidate)

        # Missing connecting vowel before 3P object suffix -hom or bare -om:
        #   narhom -> narahom, narom -> narahom
        for typed_ending, canonical_suffix in (("hom", "ahom"), ("om", "ahom")):
            if not normalized.endswith(typed_ending):
                continue
            if typed_ending == "om" and normalized.endswith("hom"):
                continue
            pre_idx = -(len(typed_ending) + 1)
            if len(normalized) + pre_idx < 0:
                continue
            pre_char = normalized[pre_idx]
            if pre_char in self.VOWELS:
                continue
            stem = normalized[: -len(typed_ending)]
            for inserted in ("a", "o"):
                cand = stem + inserted + "hom"
                if self._valid_generated_surface(cand):
                    add(cand)
                    break

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
                self._damerau_levenshtein_distance(tuple(word_key), tuple(target_key))
                for word_key in word_keys
                for target_key in target_keys
            )
            exact_key_match = bool(word_keys & target_keys)
            # Lexicalized forms are allowed to normalise their own spelling,
            # but they must not become a second fuzzy dictionary search.  A
            # nearby unrelated word must remain unresolved rather than being
            # rewritten to a lexicalized item.
            near_lexicalized_match = (
                best_distance <= 3
                and any(
                    len(word_key) >= 6
                    and len(target_key) >= 4
                    and word_key[:4] == target_key[:4]
                    for word_key in word_keys
                    for target_key in target_keys
                )
            )
            if not exact_key_match and not near_lexicalized_match:
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
        if not (normalized.endswith("agħhom") or normalized.endswith("agħha")):
            return False

        if normalized in self.dictionary_set:
            return True

        for lookup in self._strict_lookup_variants(normalized):
            if lookup in self.dictionary_set:
                return True

        suffix_generator = getattr(self, "suffix_generator", None)
        if suffix_generator is None:
            return False

        matches = suffix_generator.exact_suffix_matches(normalized)
        if matches:
            return True
        generated = suffix_generator.correct_suffix(normalized)
        return bool(generated and self._normalize_word(generated) == normalized)

    def _final_object_suffix_variants(self, word: str) -> list[str]:
        """Repair doubled h/ħ at the two final object-suffix surfaces."""
        normalized = self._normalize_word(word)
        graphemes = self._graphemes(normalized)
        h_like = {"h", "ħ"}
        variants: list[str] = []

        # a[h/ħ][h/ħ]om always represents the elided guttural object form.
        if (
            len(graphemes) >= 5
            and graphemes[-5] == "a"
            and graphemes[-4] in h_like
            and graphemes[-3] in h_like
            and graphemes[-2:] == ["o", "m"]
        ):
            variants.append(
                self._from_graphemes(
                    graphemes[:-5] + ["a", "għ", "h", "o", "m"]
                )
            )

        # a[h/ħ][h/ħ]a can be -agħha or an extra h before ordinary -aha.
        # Each resulting verb surface must be independently recognised.
        if (
            len(graphemes) >= 4
            and graphemes[-4] == "a"
            and graphemes[-3] in h_like
            and graphemes[-2] in h_like
            and graphemes[-1] == "a"
        ):
            guttural = self._from_graphemes(
                graphemes[:-4] + ["a", "għ", "h", "a"]
            )
            if self._supports_verb_final_gh_repair(guttural):
                variants.append(guttural)

            plain = self._from_graphemes(graphemes[:-4] + ["a", "h", "a"])
            suffix_generator = getattr(self, "suffix_generator", None)
            if (
                plain in self.dictionary_set
                or any(
                    candidate in self.dictionary_set
                    for candidate in self._strict_lookup_variants(plain)
                )
                or (
                    suffix_generator is not None
                    and bool(suffix_generator.exact_suffix_matches(plain))
                )
            ):
                variants.append(plain)

        return list(dict.fromkeys(variants))

    def _restore_direct_object_h_vowel(self, word: str) -> list[str]:
        """Restore the stem vowel before a final direct-object ``h``."""
        normalized = self._normalize_word(word)
        variants: list[str] = []
        if normalized.endswith("ah") and len(normalized) >= 3:
            variants.append(normalized[:-2] + "ieh")
        if normalized.endswith("ih") and not normalized.endswith("ieh") and len(normalized) >= 3:
            variants.append(normalized[:-2] + "ieh")

        valid: list[str] = []
        for candidate in variants:
            # The restored surface itself can be valid even though its final
            # accented counterpart is a different lexical word (ġieh vs
            # ġieħ). Check it before lookup variants add that alternative.
            if self._direct_object_h_base(candidate) is not None:
                valid.append(candidate)
            for lookup in self._strict_lookup_variants(candidate):
                if (
                    self._is_verb_tagged_word(lookup)
                    or self._valid_suffix_surface_candidates(lookup)
                    or self._direct_object_h_base(lookup) is not None
                ):
                    valid.append(lookup)
        return list(dict.fromkeys(valid))

    def _direct_object_h_base(self, word: str) -> str | None:
        """Return a lexical verb base for a repaired final ``-ieh`` surface."""
        normalized = self._normalize_word(word)
        if not normalized.endswith("ieh") or len(normalized) < 4:
            return None

        # ``għollieh`` comes from ``għolla`` + -h, while ``ġieh`` comes
        # from ``ġie`` + -h.  Both are checked against actual verb entries.
        bases = (normalized[:-3] + "a", normalized[:-1])
        for base in bases:
            if base in self.dictionary_set:
                tags = self.word_tags.get(base, set())
                if any(tag.startswith(("T-", "Q-", "S-", "AS-", "IS-", "VERB")) for tag in tags):
                    return base
            suffix_generator = getattr(self, "suffix_generator", None)
            verb_index = getattr(suffix_generator, "verb_index", None)
            if verb_index is not None and verb_index.word_records(base):
                return base
        return None

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
        anchors = {self._extract_consonant_anchor(v) for v in variants if v}
        normalized = self._normalize_word(word)
        if normalized:
            if normalized.startswith("i"):
                j_variant = "j" + normalized[1:]
                anchors.add(self._extract_consonant_anchor(j_variant))
            elif normalized.startswith("u"):
                w_variant = "w" + normalized[1:]
                anchors.add(self._extract_consonant_anchor(w_variant))
        orthographic_anchors = set()
        for a in anchors:
            alt = (
                a.replace("h", "ħ")
                .replace("z", "ż")
                .replace("g", "ġ")
                .replace("c", "ċ")
            )
            orthographic_anchors.add(alt)
            alt_reverse = (
                a.replace("ħ", "h")
                .replace("ż", "z")
                .replace("ġ", "g")
                .replace("ċ", "c")
            )
            orthographic_anchors.add(alt_reverse)
        anchors.update(orthographic_anchors)
        return anchors

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
        letters = self.anchor_letters
        profiler = current_profiler()
        anchors_inspected = 0
        anchor_inspection_limit = 256

        for anchor in anchors:
            if max_anchor_distance == 1:
                splits = [(anchor[:i], anchor[i:]) for i in range(len(anchor) + 1)]
                deletes = [L + R[1:] for L, R in splits if R]
                transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
                replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
                inserts = [L + c + R for L, R in splits for c in letters]
                edits = set([anchor] + deletes + transposes + replaces + inserts)
                
                for edit in edits:
                    anchors_inspected += 1
                    if anchors_inspected > anchor_inspection_limit:
                        break
                    if edit in self.anchor_map:
                        candidates.extend(self.anchor_map[edit])
            else:
                if profiler is not None:
                    profiler.increment("anchor_broad_lookup_skipped")

        if profiler is not None:
            profiler.increment("anchor_entries_inspected", anchors_inspected)
            if anchors_inspected >= anchor_inspection_limit:
                profiler.increment("anchor_inspection_budget_exhausted")

        return candidates

    @lru_cache(maxsize=2048)
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

        return tuple(candidates)

    def _get_candidates(self, word: str) -> list[str]:
        normalized = self._normalize_word(word)
        profiler = current_profiler()
        before = self._get_candidates_cached.cache_info() if profiler is not None else None
        started = time.perf_counter() if profiler is not None else 0.0
        candidates = list(self._get_candidates_cached(normalized))
        if profiler is not None and before is not None:
            after = self._get_candidates_cached.cache_info()
            cache_hit = after.hits > before.hits
            profiler.increment("candidates_generated", len(candidates))
            profiler.log_stage(
                "candidate_generation",
                (time.perf_counter() - started) * 1000,
                token=normalized,
                candidates_generated=len(candidates),
                cache_hit=cache_hit,
            )
        return candidates

    def _symspell_candidates(self, word: str, *, limit: int = SYMSPELL_MAX_RESULTS) -> list[str]:
        index = getattr(self, "symspell_index", None)
        if index is None:
            return []
        normalized = self._normalize_word(word)
        profiler = current_profiler()
        started = time.perf_counter() if profiler is not None else 0.0
        candidates = index.lookup(normalized, limit=limit)
        if profiler is not None:
            profiler.increment("symspell_candidates", len(candidates))
            profiler.log_stage(
                "symspell_lookup",
                (time.perf_counter() - started) * 1000,
                token=normalized,
                candidates=len(candidates),
            )
        return candidates

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

    def _is_safe_capitalized_repair(
        self,
        original_word: str,
        candidate_word: str,
    ) -> bool:
        original_norm = self._normalize_word(original_word)
        candidate_norm = self._normalize_word(candidate_word)

        if not candidate_norm or candidate_norm == original_norm:
            return True

        safe_variants: list[str] = []
        seen: set[str] = set()

        def add(candidate: str) -> None:
            normalized = self._normalize_word(candidate)
            if normalized and normalized not in seen:
                seen.add(normalized)
                safe_variants.append(normalized)

        add(original_norm)

        for variant in self._strict_lookup_variants(original_norm):
            add(variant)
        for variant in self._manual_repair_variants(original_norm):
            add(variant)
        for variant in self._pattern_repair_variants(original_norm):
            add(variant)
        for variant in self._initial_i_variants(original_norm):
            add(variant)
        orthographic_generator = getattr(self, "orthographic_generator", None)
        if orthographic_generator is not None:
            for variant in orthographic_generator.shortcut_letter_variants(
                original_norm
            ):
                add(variant)
        for variant in self._collapse_vowel_around_token_variants(original_norm, "għ"):
            add(variant)
        for variant in self._collapse_vowel_around_token_variants(original_norm, "h"):
            add(variant)

        for source in tuple(safe_variants):
            for variant in self._insert_token_next_to_vowels(source, "għ"):
                add(variant)
            for variant in self._insert_token_next_to_vowels(source, "h"):
                add(variant)
            for variant in self._remove_token(source, "għ"):
                add(variant)
            for variant in self._remove_token(source, "h"):
                add(variant)

        return candidate_norm in seen

    def _time_expression_key(self, word: str) -> str:
        return self._strip_maltese_shortcuts(word).replace("-", "").replace("'", "")

    def _fixed_time_expression_word(self, word: str) -> str | None:
        normalized = self._normalize_word(word)
        fixed = self.TIME_EXPRESSION_WORDS.get(self._time_expression_key(normalized))
        if fixed:
            return fixed
        return None

    def _fixed_time_phrase_match(
        self,
        word_tokens: list[WordToken],
        index: int,
        sentence_initial: bool,
    ) -> tuple[str, list[dict[str, str]], int] | None:
        current_norm = self._normalize_word(word_tokens[index].text)
        next_norm = (
            self._normalize_word(word_tokens[index + 1].text)
            if index + 1 < len(word_tokens)
            else ""
        )
        third_norm = (
            self._normalize_word(word_tokens[index + 2].text)
            if index + 2 < len(word_tokens)
            else ""
        )
        fourth_norm = (
            self._normalize_word(word_tokens[index + 3].text)
            if index + 3 < len(word_tokens)
            else ""
        )

        def display(text: str) -> str:
            return self._apply_surface_case(
                word_tokens[index].text,
                text,
                sentence_initial=sentence_initial,
            )

        fixed_next = self._fixed_time_expression_word(next_norm)
        if current_norm in {"il", "l"} and fixed_next in {
            "bieraħ",
            "ilbieraħ",
            "llum",
            "illum",
        }:
            corrected_time = (
                "ilbieraħ"
                if fixed_next in {"bieraħ", "ilbieraħ"}
                else "illum"
            )
            corrected = display(corrected_time)
            return corrected, [
                {"word": corrected, "meaning": self.meaning_for(corrected_time)}
            ], 2

        if current_norm in {"il", "l"} and next_norm == "lejla":
            corrected = display("il-lejla")
            return corrected, [
                {"word": corrected, "meaning": self.meaning_for("lejla")}
            ], 2

        # Treat the fixed time expression as a compound before the generic
        # article/preposition parser sees "bil" as an article-like form.
        # With ta, the same compound keeps its apostrophe: ta' billejl.
        if current_norm == "ta" and next_norm == "bil" and third_norm == "lejl":
            corrected = display("ta' billejl")
            return corrected, [
                {"word": corrected, "meaning": self.meaning_for("billejl")}
            ], 3

        if current_norm == "bil" and next_norm == "lejl":
            corrected = display("billejl")
            return corrected, [
                {"word": corrected, "meaning": self.meaning_for("billejl")}
            ], 2

        if current_norm == "fil" and next_norm in {"ghodu", "għodu"}:
            corrected = display("filgħodu")
            return corrected, [{"word": corrected, "meaning": self.meaning_for("filgħodu")}], 2

        if current_norm == "fil" and next_norm in {"ghaxija", "għaxija"}:
            corrected = display("filgħaxija")
            return corrected, [{"word": corrected, "meaning": self.meaning_for("filgħaxija")}], 2

        if current_norm == "dal" and next_norm in {"ghodu", "għodu"}:
            corrected = display("dalgħodu")
            return corrected, [{"word": corrected, "meaning": self.meaning_for("dalgħodu")}], 2

        if current_norm == "nofs" and next_norm in {"in-nhar", "innhar", "inhar"}:
            corrected = display("nofsinhar")
            return corrected, [{"word": corrected, "meaning": self.meaning_for("nofsinhar")}], 2

        # Clock expressions retain the cardinal form: il-ħdax u nofs, not
        # the numeral-plus-noun form il-ħdax-il. The same rule works for any
        # tagged cardinal followed by "u nofs".
        if (
            current_norm in {"il", "l"}
            and third_norm == "u"
            and fourth_norm == "nofs"
        ):
            number = self._normalize_word(self.correct_word(next_norm))
            if "CARDNUM" in self._word_tag_markers(number):
                corrected = display(f"il-{number} u nofs")
                return corrected, [
                    {"word": corrected, "meaning": self.meaning_for(number)}
                ], 4

        if current_norm == "wara" and next_norm == "nofsinhar":
            corrected = display("wara nofsinhar")
            return corrected, [
                {"word": corrected, "meaning": self.meaning_for("waranofsinhar")},
                {
                    "word": self._apply_surface_case(
                        word_tokens[index].text,
                        "waranofsinhar",
                        sentence_initial=sentence_initial,
                    ),
                    "meaning": self.meaning_for("waranofsinhar"),
                },
            ], 2

        if (
            current_norm == "wara"
            and next_norm == "nofs"
            and third_norm in {"in-nhar", "innhar", "inhar"}
        ):
            corrected = display("wara nofsinhar")
            return corrected, [
                {"word": corrected, "meaning": self.meaning_for("waranofsinhar")},
                {
                    "word": self._apply_surface_case(
                        word_tokens[index].text,
                        "waranofsinhar",
                        sentence_initial=sentence_initial,
                    ),
                    "meaning": self.meaning_for("waranofsinhar"),
                },
            ], 3

        return None

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

        # Missing għ before the feminine object suffix -agħha typed as -ahha:
        #   nitfahha -> nitfagħha
        if original_norm.endswith("ahha"):
            stem = original_norm[:-4]
            if candidate_norm == stem + "agħha":
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
        typo_anchor = self._extract_consonant_anchor(normalized)

        candidates_to_score = set(self._get_candidates(normalized))
        for variant in self._lexicalized_form_variants(normalized):
            candidates_to_score.update(self._get_candidates(variant))

        # Also expand shortcut letter variants (c→ċ, g→ġ, z→ż, h→ħ) so that
        # e.g. 'iddecidew' searches candidates of 'iddeċidew' too.
        ortho_gen = getattr(self, "orthographic_generator", None)
        if ortho_gen is not None and hasattr(ortho_gen, "shortcut_letter_variants"):
            for sc_variant in ortho_gen.shortcut_letter_variants(normalized):
                sc_norm = self._normalize_word(sc_variant)
                if sc_norm and sc_norm != normalized:
                    candidates_to_score.update(self._get_candidates(sc_norm))

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

        # Pre-filter by anchor distance: reject candidates whose consonant skeleton
        # differs from the typo's by more than max_distance+1.
        # This removes clearly unrelated words (e.g. anchor_dist=4+) without the
        # ordering bias of a top-N sort, which can incorrectly prefer related-but-wrong
        # forms over the actual target.
        anchor_reject_threshold = max_distance + 1

        profiler = current_profiler()
        started = time.perf_counter() if profiler is not None else 0.0
        rows: list[ScoreRow] = []
        for candidate in candidates_to_score:
            if candidate_filter and not candidate_filter(candidate):
                continue
            # Fast length pre-filter: skip expensive scoring if lengths differ too much
            cand_len = self.word_lengths.get(
                candidate, len(self._letter_tokens(candidate))
            )
            if abs(cand_len - typo_len) > max_distance + 2:
                continue
            # Fast anchor distance pre-filter: skip if consonant skeletons are too far
            cand_anchor = self._extract_consonant_anchor(candidate)
            if self._damerau_levenshtein_distance(
                tuple(typo_anchor), tuple(cand_anchor)
            ) > anchor_reject_threshold:
                continue
            if self._is_implausible_vowel_swap(normalized, candidate):
                continue
            row = self._candidate_score(normalized, candidate, stage)
            rows.append(row)
            # Early exit: perfect phonetic match found, no need to keep scoring
            if row.edit_distance == 0 and row.score < score_limit:
                break

        if profiler is not None:
            profiler.increment("candidates_scored", len(rows))
            profiler.log_stage(
                "ranked_candidate",
                (time.perf_counter() - started) * 1000,
                token=normalized,
                ranked_stage=stage,
                candidates_generated=len(candidates_to_score),
                candidates_scored=len(rows),
            )

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
        max_distance = min(1, self._max_distance(original_word))
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

    def _close_apostrophe_ranked_match(self, word: str) -> str | None:
        normalized = self._normalize_word(word)
        if "'" in normalized:
            return None

        row = self._best_ranked_candidate(
            normalized,
            stage="close_apostrophe",
            score_limit=0.24,
            max_distance=1,
        )
        if (
            row is not None
            and "'" in row.candidate
            and row.edit_distance <= 1
        ):
            return self._match_capitalisation(word, row.candidate)
        return None

    def _reduced_il_xi_phrase_match(
        self,
        word_tokens: list[WordToken],
        index: int,
        previous_surface_word: str | None,
        sentence_initial: bool,
    ) -> tuple[str, list[dict[str, str]], int] | None:
        """Keep ``'il xi PRON`` after a preceding vowel as one phrase."""
        if index + 2 >= len(word_tokens):
            return None
        if self._normalize_word(word_tokens[index].text) != "il":
            return None
        if self._normalize_word(word_tokens[index + 1].text) != "xi":
            return None
        if not self._word_ends_with_vowel(previous_surface_word):
            return None

        tail = self.correct_word(word_tokens[index + 2].text)
        tail_norm = self._normalize_word(tail)
        if "PRON" not in self._word_tag_markers(tail_norm):
            return None

        corrected = self._apply_surface_case(
            word_tokens[index].text,
            f"'il xi {tail_norm}",
            sentence_initial=sentence_initial,
        )
        return corrected, [{"word": corrected, "meaning": self.meaning_for(tail_norm)}], 3

    def _strict_terminal_apostrophe_match(self, word: str) -> str | None:
        """Resolve an omitted final apostrophe before consonant confusion.

        The candidate must be an existing verb form, so this does not turn
        ordinary function words such as ``ta`` into an apostrophe form without
        the later contextual parser deciding that question.
        """
        normalized = self._normalize_word(word)
        if not normalized or normalized.endswith("'"):
            return None

        candidates = list(self._strict_lookup_variants(normalized))
        orthographic_generator = getattr(self, "orthographic_generator", None)
        if orthographic_generator is not None and hasattr(
            orthographic_generator, "shortcut_letter_variants"
        ):
            candidates.extend(
                orthographic_generator.shortcut_letter_variants(normalized)
            )

        seen: set[str] = set()
        for candidate in candidates:
            candidate = self._normalize_word(candidate)
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)
            apostrophe_form = f"{candidate}'"
            if (
                apostrophe_form in self.dictionary_set
                and self._is_verb_tagged_word(apostrophe_form)
            ):
                return self._match_capitalisation(word, apostrophe_form)
        return None

    def _shortcut_gh_suggestion_match(self, word: str) -> str | None:
        orthographic_generator = getattr(self, "orthographic_generator", None)
        if orthographic_generator is None:
            return None
        if not hasattr(orthographic_generator, "shortcut_letter_variants"):
            return None
        if not hasattr(orthographic_generator, "dictionary_gh_suggestion_variants"):
            return None

        candidates: list[str] = []
        for shortcut_variant in orthographic_generator.shortcut_letter_variants(
            word,
            max_changes=3,
        ):
            for candidate in orthographic_generator.dictionary_gh_suggestion_variants(
                shortcut_variant
            ):
                normalized_candidate = self._normalize_word(candidate)
                if (
                    normalized_candidate
                    and normalized_candidate not in candidates
                ):
                    candidates.append(normalized_candidate)

        if len(candidates) == 1:
            return self._match_capitalisation(word, candidates[0])
        return None

    def _match_capitalisation(self, original: str, corrected: str) -> str:
        if original.isupper():
            return corrected.upper()
        if original[:1].isupper():
            return self._capitalize_first_letter(corrected)
        # Direct word correction preserves title case. Rich-text correction can
        # still lower ordinary mid-sentence words after context is known.
        return corrected

    def _capitalize_first_letter(self, word: str) -> str:
        if not word:
            return word
        return word[:1].upper() + word[1:]

    def _apply_surface_case(
        self,
        original: str,
        corrected: str,
        *,
        sentence_initial: bool = False,
    ) -> str:
        if original.isupper():
            return corrected.upper()
        if sentence_initial:
            return self._capitalize_first_letter(corrected)
        return self._match_capitalisation(original, corrected)

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
        # Newlines often merely wrap a paragraph in pasted social-media text;
        # they do not by themselves begin a new sentence.
        while cursor >= 0 and text[cursor].isspace():
            cursor -= 1
        if cursor < 0:
            return True
        if text[cursor] == ".":
            previous_is_dot = cursor > 0 and text[cursor - 1] == "."
            next_is_dot = cursor + 1 < len(text) and text[cursor + 1] == "."
            return not (previous_is_dot or next_is_dot)
        return text[cursor] in "?!"

    def _ensure_terminal_period(self, text: str) -> str:
        # Punctuation belongs to the preceding token in Maltese output. Do
        # this once on the assembled text so every phrase path shares the
        # same spacing rule, including input such as "kelma !".
        text = re.sub(r"[ \t]{2,}", " ", text)
        text = re.sub(r"[ \t]+([,.;?!])", r"\1", text)
        text = re.sub(r"\(\s+", "(", text)
        text = re.sub(r"\s+\)", ")", text)
        stripped = text.rstrip()
        if not stripped or stripped[-1] in ".?!":
            return text
        if (
            len(stripped) >= 2
            and stripped[-1] in "'\"\u2019\u201d"
            and stripped[-2] in ".?!"
        ):
            return text
        trailing = text[len(stripped) :]
        return f"{stripped}.{trailing}"

    def _canonical_suggestion_key(self, word: str) -> str:
        normalized = self._normalize_word(word)
        if normalized in {
            "xħin",
            "x'ħin",
            "waranofsinhar",
            "wara nofsinhar",
            "llejla",
            "il-lejla",
        }:
            return normalized
        return "".join(char for char in normalized if char.isalnum() or char == "'")

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

    def _exact_place_word(self, word: str) -> str | None:
        normalized = self._normalize_word(word)
        if not normalized or not self.place_words:
            return None

        if normalized in self.place_word_set:
            return self.place_word_display.get(normalized, word)

        return None

    def _is_recognized_surface(self, word: str) -> bool:
        normalized = self._normalize_word(word)
        if not normalized:
            return True
        if self._capitalized_name_kind(word):
            return True
        initial_vowel_surfaces = self._initial_vowel_surface_options(word)
        if any(initial_vowel_surfaces):
            return True
        if normalized in self.dictionary_set:
            return True
        if self._correct_noun_possessive_suffix(normalized) == normalized:
            return True
        if normalized in self._no_possession_noun_set():
            return True
        if self._exact_place_word(word):
            return True
        if normalized in self.country_english_to_maltese:
            return True
        if normalized in self.country_maltese_to_english:
            return True
        if self._is_recognized_hyphenated_compound(normalized):
            return True
        if self._valid_suffix_surface_candidates(normalized):
            return True
        return any(
            (
                self._is_verb_tagged_word(normalized),
                self._is_noun_tagged_word(normalized),
                self._is_pronoun_tagged_word(normalized),
                self._is_adjective_tagged_word(normalized),
                self._is_adverb_tagged_word(normalized),
                self._is_preposition_tagged_word(normalized),
            )
        )

    def _is_recognized_hyphenated_compound(self, word: str) -> bool:
        """Recognise verified fused function-word compounds such as bil-quddiem."""
        normalized = self._normalize_word(word)
        if normalized.count("-") != 1:
            return False

        prefix, tail = normalized.split("-", 1)
        if not prefix or not tail or not self._valid_generated_surface(tail):
            return False

        # These fused prepositions can precede nominal, adjectival, adverbial,
        # and prepositional lexical tails.  The tail is still verified, so this
        # does not turn arbitrary hyphenated text into a recognised word.
        if prefix in {"bil", "fil", "mal", "mill", "tal", "għall", "bħall"}:
            return any(
                (
                    self._is_noun_tagged_word(tail),
                    self._is_adjective_tagged_word(tail),
                    self._is_adverb_tagged_word(tail),
                    self._is_preposition_tagged_word(tail),
                    self._is_pronoun_tagged_word(tail),
                )
            )

        return False

    def _mark_unrecognized_tokens(self, tokens: list[dict]) -> None:
        bulk_mode = getattr(self._local, "bulk_mode", False)
        for token in tokens:
            if token.get("type") != "word":
                continue
            if token.pop("force_unrecognized", False):
                token["unrecognized"] = True
                continue
            if token.get("is_quote", False):
                token["unrecognized"] = bool(token.pop("force_unrecognized", False))
                continue
            corrected = str(token.get("corrected", "")).strip()
            original = str(token.get("original", "")).strip()
            if original and self._normalize_word(original) != self._normalize_word(corrected):
                token["unrecognized"] = False
                continue
            if bulk_mode:
                normalized = self._normalize_word(corrected)
                token["unrecognized"] = bool(corrected) and not (
                    normalized in self.dictionary_set
                    or bool(self.word_tags.get(normalized))
                    or self._exact_place_word(corrected)
                    or self._is_recognized_surface(normalized)
                )
            else:
                token["unrecognized"] = bool(corrected) and not self._is_recognized_surface(
                    corrected
                )

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
                        tuple(anchor),
                        tuple(known_anchor),
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
            original_phrase = text[
                matches[index].start() : matches[index + 1].end()
            ]
            if current_normalized == "l" and self._is_initial_capitalized(next_word):
                corrected_tail = self._correct_place_word(next_word)
                if corrected_tail:
                    prefix = (
                        self._capitalize_first_letter(current)
                        if self._is_sentence_initial_position(text, matches[index].start())
                        else current
                    )
                    return original_phrase, f"{prefix}-{corrected_tail}", 2
            if current_capitalized or self._is_initial_capitalized(next_word):
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
        if self._capitalized_name_kind(word):
            return word
        if normalized in self._protected_name_set():
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
        if best and self._is_safe_capitalized_repair(word, best):
            return self._match_capitalisation(word, best)

        return word

    def _conservative_capitalized_word(self, word: str) -> str:
        original_norm = self._normalize_word(word)
        if self._capitalized_name_kind(word):
            return word
        repaired_name = self._capitalized_name_repair(word)
        if repaired_name:
            return repaired_name
        if original_norm in self._protected_name_set():
            return word
        corrected_place_word = self._exact_place_word(word)
        if corrected_place_word:
            return corrected_place_word

        strict_capitalized = self._try_exact_variants(
            word,
            self._strict_lookup_variants(original_norm),
        )
        final_capitalized = None
        if (
            strict_capitalized is not None
            and self._strip_maltese_shortcuts(original_norm)
            == self._strip_maltese_shortcuts(strict_capitalized)
        ):
            final_capitalized = strict_capitalized
        if final_capitalized is None:
            diacritic_candidate = self.correct_word(original_norm)
            if (
                self._normalize_word(diacritic_candidate) in self.dictionary_set
                and self._is_safe_capitalized_repair(
                    original_norm,
                    diacritic_candidate,
                )
                and self._word_distance(
                    original_norm,
                    self._normalize_word(diacritic_candidate),
                )
                <= self._max_distance(original_norm)
            ):
                final_capitalized = self._match_capitalisation(
                    word,
                    diacritic_candidate,
                )
        if final_capitalized is None:
            final_capitalized = word
        return final_capitalized

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

    def _exact_function_word_pass_through(self, word: str) -> bool:
        normalized = self._normalize_word(word)
        return normalized in {
            "bil",
            "fil",
            "mal",
            "mil",
            "mill",
            "mid",
            "sal",
            "tal",
            "għal",
            "għall",
            "lil",
            "lill",
        }

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

    def _word_starts_with_doubled_consonant(self, word: str) -> bool:
        graphemes = self._graphemes(self._normalize_word(word))
        letters = [
            token for token in graphemes if token and any(ch.isalpha() for ch in token)
        ]
        if len(letters) < 2:
            return False
        return (
            letters[0] == letters[1]
            and len(letters[0]) == 1
            and letters[0] not in self.VOWELS
        )

    def _prefer_plain_after_vowel_surface(self, word: str) -> bool:
        return self._word_starts_with_doubled_consonant(word)

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

    def _manual_initial_vowel_variants(self, word: str) -> tuple[str, ...]:
        normalized = self._normalize_word(word)
        return tuple(
            self._normalize_word(candidate)
            for candidate in self.INITIAL_VOWEL_NOUN_EXCEPTIONS.get(normalized, ())
        )

    def _is_manual_initial_vowel_exception(self, word: str) -> bool:
        normalized = self._normalize_word(word)
        if normalized in self.INITIAL_VOWEL_NOUN_EXCEPTIONS:
            return True
        return any(
            normalized == candidate
            for variants in self.INITIAL_VOWEL_NOUN_EXCEPTIONS.values()
            for candidate in variants
        )

    def _is_imperfect_surface_candidate(self, word: str) -> bool:
        normalized = self._normalize_word(word)
        if not normalized:
            return False
        records = self._verb_records_for_surface(normalized)
        return bool(records) and all(record.tense == "MPERF" for record in records)

    def _blocks_initial_vowel_insertion(self, word: str) -> bool:
        normalized = self._normalize_word(word)
        return normalized.startswith("pr") or normalized == "ngħamel"

    def _starts_with_consonant(self, word: str) -> bool:
        graphemes = self._graphemes(self._normalize_word(word))
        for token in graphemes:
            if token and any(ch.isalpha() for ch in token):
                return token not in self.VOWELS
        return False

    def _is_initial_vowel_base_candidate(self, word: str) -> bool:
        normalized = self._normalize_word(word)
        if not normalized:
            return False
        return (
            self._verb_uses_initial_i_prefix(normalized)
            or self._is_manual_initial_vowel_exception(normalized)
        )

    def _is_form7_perf_or_imp_surface(self, word: str) -> bool:
        normalized = self._normalize_word(word)
        if not normalized:
            return False
        return any(
            record.form_class.startswith("F7") and record.tense in {"PERF", "IMP"}
            for record in self._verb_records_for_surface(normalized)
        )

    def _initial_vowel_surface_options(
        self,
        corrected_word: str,
    ) -> tuple[list[str], list[str]]:
        normalized = self._normalize_word(corrected_word)
        suffix_generator = getattr(self, "suffix_generator", None)
        if normalized.startswith("i") and (
            normalized in self.dictionary_set
            or (
                suffix_generator is not None
                and bool(suffix_generator.exact_suffix_matches(normalized))
            )
        ):
            # A verified lexical i- form, including non-Semitic iCC verbs
            # with generated suffixes, is never a removable epenthetic vowel.
            return [], []
        if (
            not normalized
            or " " in corrected_word
            or "-" in corrected_word
            or "'" in corrected_word
        ):
            return [], []

        prefer_vowel: list[str] = []
        prefer_plain: list[str] = []

        def add_unique(items: list[str], candidate: str) -> None:
            candidate = self._normalize_word(candidate)
            if candidate and candidate not in items:
                items.append(candidate)

        for candidate in self._manual_initial_vowel_variants(normalized):
            if normalized.startswith(("i", "u")):
                add_unique(prefer_plain, candidate)
            else:
                add_unique(prefer_vowel, candidate)

        if normalized.startswith("i") and len(normalized) > 1:
            plain_cluster = normalized[1:]
            if (
                not plain_cluster.startswith(("j", "w"))
                and
                self._word_starts_with_two_consonants(plain_cluster)
                and not self._blocks_initial_vowel_insertion(plain_cluster)
                and (
                    self._is_initial_vowel_base_candidate(plain_cluster)
                    or self._is_form7_perf_or_imp_surface(plain_cluster)
                )
            ):
                add_unique(prefer_plain, plain_cluster)

            plain_j = f"j{normalized[1:]}"
            if (
                self._starts_with_consonant(normalized[1:])
                and (
                    self._is_initial_vowel_base_candidate(plain_j)
                    or self._is_imperfect_surface_candidate(plain_j)
                    or self._is_verb_tagged_word(plain_j)
                )
            ):
                add_unique(prefer_plain, plain_j)

            # Some imperfect entries retain ji- in the verb dictionary while
            # their iCC surface is used after a consonant or punctuation.
            # Recognize that surface directly so it remains valid until
            # context decides whether to expose ji- instead.
            ji_surface = f"ji{normalized[1:]}"
            if (
                self._is_imperfect_surface_candidate(ji_surface)
                or self._is_verb_tagged_word(ji_surface)
            ):
                add_unique(prefer_vowel, normalized)
                add_unique(prefer_plain, ji_surface)

        if normalized.startswith("u") and len(normalized) > 1:
            plain_w = f"w{normalized[1:]}"
            if (
                self._starts_with_consonant(normalized[1:])
                and self._is_initial_vowel_base_candidate(plain_w)
            ):
                add_unique(prefer_plain, plain_w)

        if (
            not normalized.startswith(("j", "w"))
            and
            self._word_starts_with_two_consonants(normalized)
            and not self._blocks_initial_vowel_insertion(normalized)
            and (
                self._is_initial_vowel_base_candidate(normalized)
                or self._is_form7_perf_or_imp_surface(normalized)
            )
        ):
            add_unique(prefer_vowel, f"i{normalized}")

        if (
            normalized.startswith("j")
            and self._starts_with_consonant(normalized[1:])
            # Verbs whose second radical is għ have their own surface
            # behaviour.  They never enter the generic ji-/i- alternation.
            and not normalized.startswith(("jgħ", "jgh"))
            and self._is_imperfect_surface_candidate(normalized)
        ):
            add_unique(prefer_vowel, f"i{normalized[1:]}")

        if (
            normalized.startswith("ji")
            and self._word_starts_with_two_consonants(normalized[2:])
            and self._is_imperfect_surface_candidate(normalized)
        ):
            add_unique(prefer_vowel, f"i{normalized[2:]}")

        if (
            normalized.startswith("w")
            and self._starts_with_consonant(normalized[1:])
            and self._is_initial_vowel_base_candidate(normalized)
        ):
            add_unique(prefer_vowel, f"u{normalized[1:]}")

        return prefer_vowel, prefer_plain

    def _apply_initial_vowel_surface(
        self,
        original_word: str,
        corrected_word: str,
        *,
        prefer_vowel_surface: bool,
    ) -> str:
        original_norm = self._normalize_word(original_word)
        if original_norm:
            # Preserve a correctly written dictionary ji-/w- imperfect form.
            # Context may transform an input i-/u- surface, but must not erase
            # the initial consonant of an already valid jixtieq-style input.
            if (
                original_norm.startswith(("ji", "wi"))
                and self._normalize_word(corrected_word) == original_norm
                and self._is_recognized_surface(original_norm)
            ):
                return original_word
            if (
                original_norm.startswith("ji")
                and self._normalize_word(corrected_word).startswith("ji")
                and self._is_recognized_surface(corrected_word)
            ):
                return corrected_word
            original_vowel, original_plain = self._initial_vowel_surface_options(
                original_word
            )
            if original_norm.startswith(("i", "u")):
                if prefer_vowel_surface and (original_vowel or original_plain):
                    return original_word
                original_ordered = original_vowel if prefer_vowel_surface else original_plain
                if original_ordered:
                    return self._match_capitalisation(original_word, original_ordered[0])

        prefer_vowel, prefer_plain = self._initial_vowel_surface_options(corrected_word)
        if (
            prefer_vowel_surface
            and not prefer_vowel
            and prefer_plain
            and self._prefer_plain_after_vowel_surface(prefer_plain[0])
        ):
            return self._match_capitalisation(corrected_word, prefer_plain[0])

        ordered = prefer_vowel if prefer_vowel_surface else prefer_plain
        if not ordered:
            return corrected_word
        return self._match_capitalisation(corrected_word, ordered[0])

    def _breaks_empathetic_i_context(self, raw_text: str) -> bool:
        return any(mark in raw_text for mark in ",;:")

    def _starts_initial_vowel_context(self, raw_text: str) -> bool:
        for char in raw_text:
            if char.isspace():
                continue
            if unicodedata.category(char).startswith("P"):
                return True
        return False

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

    def _split_article_unknown_tail(
        self,
        article: str,
        tail: str,
        *,
        previous: str | None,
    ) -> str | None:
        article_rules = getattr(self, "article_phrase_rules", None)
        if article_rules is None:
            return None

        article_norm = self._normalize_word(article).rstrip("-")
        tail_norm = self._normalize_word(tail)
        if not article_norm or not tail_norm:
            return None

        corrected_tail = self._normalize_word(self.correct_word(tail))
        if (
            corrected_tail in self.dictionary_set
            or self._is_recognized_surface(corrected_tail)
            or self._is_noun_tagged_word(corrected_tail)
            or self._is_adjective_tagged_word(corrected_tail)
            or self._is_verb_tagged_word(corrected_tail)
        ):
            return None

        definite_articles = {
            "il", "l", "ic", "iċ", "id", "in", "ir", "is", "it", "ix", "iz", "iż",
        }
        article_prepositions = {
            "tal", "mal", "bil", "fil", "fis", "lill", "mill", "għall",
            "ghall", "bħall", "bhall", "sal", "mic", "miċ",
        }

        if article_norm in definite_articles:
            return article_rules.corrected_article_phrase(article_norm, tail_norm, previous)
        if article_norm in article_prepositions:
            assimilated = None
            if hasattr(article_rules, "_assimilated_prefix_surface"):
                assimilated = article_rules._assimilated_prefix_surface(
                    article_norm,
                    tail_norm,
                )
            if assimilated is not None:
                surface_prefix, _canonical = assimilated
                return f"{surface_prefix}-{tail_norm}"
            preposition_stems = {
                "tal": "ta",
                "mal": "ma",
                "bil": "bi",
                "fil": "fi",
                "mill": "mi",
                "għall": "għa",
                "ghall": "għa",
                "bħall": "bħa",
                "bhall": "bħa",
                "lill": "li",
                "sal": "sa",
                "mic": "mi",
                "miċ": "mi",
            }
            canonical_prefixes = {
                "ghall": "għall",
                "bhall": "bħall",
                "mic": "miċ",
            }
            sun_letters = {"ċ", "d", "n", "r", "s", "t", "x", "z", "ż"}
            first = self._graphemes(tail_norm)[0] if tail_norm else ""
            if first in sun_letters and article_norm in preposition_stems:
                return f"{preposition_stems[article_norm]}{first}-{tail_norm}"
            return f"{canonical_prefixes.get(article_norm, article_norm)}-{tail_norm}"
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
        corrected_place = self._correct_place_word(noun)
        if corrected_place is not None:
            corrected_noun = corrected_place
            corrected_noun_norm = self._normalize_word(corrected_place)
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

    def _word_tag_markers(self, word: str) -> set[str]:
        normalized = self._normalize_word(word)
        markers: set[str] = set()
        for tag in self.word_tags.get(normalized, ()):
            marker = tag.split("-", 1)[0].upper()
            if marker:
                markers.add(marker)
        return markers

    def _noun_number_markers(self, word: str) -> set[str]:
        markers = self._word_tag_markers(word)
        out: set[str] = set()
        for marker in markers:
            if marker.startswith("SINGNOUN"):
                out.add("SINGNOUN")
            elif marker in {"PLUNOUN", "PAUCNOUN", "COLLNOUN"}:
                out.add(marker)
        return out

    def _is_plural_like_noun(self, word: str) -> bool:
        return bool(self._noun_number_markers(word) & {"PLUNOUN", "PAUCNOUN", "COLLNOUN"})

    def _english_meaning_key(self, meaning: str) -> str:
        cleaned = re.sub(r"\([^)]*\)", " ", str(meaning or "").casefold())
        cleaned = re.sub(r"[^\w\s]", " ", cleaned)
        parts: list[str] = []
        for token in cleaned.split():
            if token in {"the", "a", "an", "or", "of", "and"}:
                continue
            if len(token) > 4 and token.endswith("ies"):
                token = token[:-3] + "y"
            elif len(token) > 4 and token.endswith("es") and not token.endswith(("ses", "xes", "zes")):
                token = token[:-2]
            elif len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
                token = token[:-1]
            parts.append(token)
        return " ".join(parts)

    def _noun_meaning_keys(self, word: str) -> set[str]:
        keys: set[str] = set()
        for meaning in meaning_index.meanings_for(word):
            key = self._english_meaning_key(meaning)
            if key:
                keys.add(key)
        fallback = self._english_meaning_key(self.meaning_for(word))
        if fallback:
            keys.add(fallback)
        return keys

    @lru_cache(maxsize=512)
    def _english_fixed_noun_suggestions(self, english: str) -> tuple[str, ...]:
        """Return exact Maltese noun gloss matches for an English surface."""
        target = self._english_meaning_key(english)
        if not target:
            return ()

        matches: list[str] = []
        for word, tags in self.word_tags.items():
            if not any(
                tag.split("-", 1)[0].upper().startswith(
                    ("SINGNOUN", "PLUNOUN", "PAUCNOUN", "COLLNOUN")
                )
                for tag in tags
            ):
                continue
            meanings = {
                self._english_meaning_key(meaning)
                for meaning in meaning_index.meanings_for(word)
            }
            if target in meanings:
                matches.append(word)
        return tuple(matches[:3])

    def _ensure_noun_number_index(self) -> None:
        if self._noun_number_index is not None and self._noun_number_prefix_index is not None:
            return

        index: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
        prefix_index: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
        for word, tags in self.word_tags.items():
            marker: str | None = None
            for tag in tags:
                prefix = tag.split("-", 1)[0].upper()
                if prefix.startswith("SINGNOUN"):
                    marker = "SINGNOUN"
                    break
                if prefix in {"PLUNOUN", "PAUCNOUN", "COLLNOUN"}:
                    marker = prefix
                    break
            if marker is None:
                continue
            anchor = self.word_anchors.get(word) or self._extract_consonant_anchor(word)
            if not anchor:
                continue
            index[anchor][marker].append(word)
            prefix_index[word[:3]][marker].append(word)

        self._noun_number_index = {
            anchor: {
                marker: tuple(sorted(words))
                for marker, words in marker_map.items()
            }
            for anchor, marker_map in index.items()
        }
        self._noun_number_prefix_index = {
            prefix: {
                marker: tuple(sorted(words))
                for marker, words in marker_map.items()
            }
            for prefix, marker_map in prefix_index.items()
        }

    def _best_numbered_noun_variant(
        self,
        word: str,
        target_markers: tuple[str, ...],
    ) -> str | None:
        normalized = self._normalize_word(word)
        if not normalized:
            return None

        current_markers = self._noun_number_markers(normalized)
        if target_markers and target_markers[0] in current_markers:
            return normalized

        self._ensure_noun_number_index()
        if self._noun_number_index is None or self._noun_number_prefix_index is None:
            return None

        anchor = self.word_anchors.get(normalized) or self._extract_consonant_anchor(normalized)
        candidate_pool: list[tuple[int, int, str]] = []
        if anchor in self._noun_number_index:
            for priority, marker in enumerate(target_markers):
                candidate_pool.extend(
                    (priority, 0, candidate)
                    for candidate in self._noun_number_index[anchor].get(marker, ())
                )
        if normalized[:3] in self._noun_number_prefix_index:
            existing_candidates = {candidate for _, _, candidate in candidate_pool}
            for priority, marker in enumerate(target_markers):
                candidate_pool.extend(
                    (priority, 1, candidate)
                    for candidate in self._noun_number_prefix_index[normalized[:3]].get(marker, ())
                    if candidate not in existing_candidates
                )

        if not candidate_pool:
            return None

        source_meanings = self._noun_meaning_keys(normalized)
        source_prefix = normalized[:3]

        def score(item: tuple[int, int, str]) -> tuple[int, int, int, int, int, int, str]:
            marker_priority, source_priority, candidate = item
            candidate_meanings = self._noun_meaning_keys(candidate)
            shares_meaning = bool(source_meanings and candidate_meanings & source_meanings)
            same_prefix = candidate[:3] == source_prefix
            return (
                marker_priority,
                source_priority,
                0 if shares_meaning else 1,
                0 if same_prefix else 1,
                self._word_distance(normalized, candidate),
                abs(len(candidate) - len(normalized)),
                candidate,
            )

        _, _, best = min(candidate_pool, key=score)
        return best

    def _attnum_needs_initial_vowel_surface(self, noun: str) -> bool:
        normalized = self._normalize_word(noun)
        if not normalized:
            return False
        if normalized.startswith("i") and len(normalized) > 1:
            tail = normalized[1:]
            return (
                not tail.startswith(("j", "w"))
                and self._word_starts_with_two_consonants(tail)
                and not self._blocks_initial_vowel_insertion(tail)
                and self._is_noun_tagged_word(tail)
            )
        return (
            self._word_starts_with_two_consonants(normalized)
            and not self._blocks_initial_vowel_insertion(normalized)
            and self._is_noun_tagged_word(normalized)
        )

    def _attnum_surface_noun(self, noun: str) -> str:
        normalized = self._normalize_word(noun)
        if not normalized:
            return normalized
        if normalized.startswith("i") and len(normalized) > 1:
            tail = normalized[1:]
            if (
                not tail.startswith(("j", "w"))
                and self._word_starts_with_two_consonants(tail)
                and not self._blocks_initial_vowel_insertion(tail)
                and self._is_noun_tagged_word(tail)
            ):
                return normalized
        if (
            self._word_starts_with_two_consonants(normalized)
            and not self._blocks_initial_vowel_insertion(normalized)
            and self._is_noun_tagged_word(normalized)
        ):
            return f"i{normalized}"
        return normalized

    def _cardinal_attnum_surface(self, numeral: str, *, require_long: bool) -> str:
        normalized = self._normalize_word(numeral)
        if require_long:
            return self.SHORT_ATTNUM_TO_LONG.get(normalized, normalized)
        return normalized

    def _cardinal_to_short_attnum_surface(self, numeral: str) -> str:
        normalized = self._normalize_word(numeral)
        return self.CARDINAL_TO_SHORT_ATTNUM.get(normalized, normalized)

    def _number_surface_for_noun(
        self,
        numeral: str,
        *,
        prefer_long: bool,
    ) -> str:
        short_surface = self._cardinal_to_short_attnum_surface(numeral)
        if prefer_long:
            return self._cardinal_attnum_surface(short_surface, require_long=True)
        return short_surface

    def _split_article_token(self, word: str) -> tuple[str, str] | None:
        normalized = self._normalize_word(word)
        if "-" not in normalized:
            return None
        prefix, tail = normalized.split("-", 1)
        if not self._article_like_token(prefix):
            return None
        return prefix, tail

    def _number_phrase_payload(
        self,
        numeral_word: str,
        noun_word: str,
        *,
        sentence_initial: bool,
        article_word: str | None = None,
    ) -> tuple[str, list[dict], bool] | None:
        numeral_norm = self._normalize_word(numeral_word)
        article_rules = getattr(self, "article_phrase_rules", None)
        if article_rules is None:
            return None
        article_prefix: str | None = None
        article_numeral = article_word

        if article_word is None:
            split_article = self._split_article_token(numeral_word)
            if split_article is not None:
                article_prefix, numeral_norm = split_article
                article_numeral = numeral_word
        else:
            article_prefix = self._normalize_word(article_word)

        noun_article_prefix: str | None = None
        noun_surface_word = noun_word
        split_noun_article = self._split_article_token(noun_word)
        if split_noun_article is not None:
            noun_article_prefix, noun_surface_word = split_noun_article

        # Numeral agreement only applies to nominal targets.  Several number
        # words are also ordinary lexical words, so allowing a following verb
        # here caused rewrites such as ``wieħed kien`` -> ``wieħed iknien``.
        if self._is_verb_tagged_word(noun_surface_word):
            return None

        corrected_numeral = self._normalize_word(self.correct_word(numeral_word))
        if corrected_numeral and article_rules.is_num(corrected_numeral):
            numeral_norm = corrected_numeral

        markers = self._word_tag_markers(numeral_norm)
        has_short_attnum = "SHORTATTNUM" in markers
        has_long_attnum = "LONGATTNUM" in markers
        has_ordnum = "ORDNUM" in markers
        has_cardnum = "CARDNUM" in markers
        if not any((has_short_attnum, has_long_attnum, has_ordnum, has_cardnum)):
            return None

        corrected_noun = self._normalize_word(self.correct_word(noun_surface_word))
        if not corrected_noun or not self._is_noun_tagged_word(corrected_noun):
            return None

        def compose_number(base: str) -> str:
            if article_prefix is None:
                return base
            if article_prefix in {"l", "il"}:
                if self._normalize_word(base).startswith(
                    ("għ", "gh", "a", "e", "i", "o", "u", "à", "è", "ì", "ò", "ù")
                ):
                    return f"l-{base}"
                return f"il-{base}"
            inline = self._correct_inline_article_word(
                f"{article_prefix}-{base}",
                previous=None,
            )
            return inline or f"{article_prefix}-{base}"

        def compose_phrase(number_surface: str, noun_surface: str) -> str:
            phrase = f"{number_surface} {noun_surface}"
            original_head = article_numeral or numeral_word
            phrase = self._match_capitalisation(
                f"{original_head} {noun_surface_word}",
                phrase,
            )
            if sentence_initial:
                phrase = self._capitalize_first_letter(phrase)
            return phrase

        singular_candidate = self._best_numbered_noun_variant(
            corrected_noun,
            ("SINGNOUN",),
        ) or corrected_noun
        plural_candidate = self._best_numbered_noun_variant(
            corrected_noun,
            ("PAUCNOUN", "PLUNOUN", "COLLNOUN"),
        ) or corrected_noun

        require_long = self._attnum_needs_initial_vowel_surface(plural_candidate)
        plural_surface_noun = (
            self._attnum_surface_noun(plural_candidate)
            if require_long
            else plural_candidate
        )
        singular_surface_number = self._number_surface_for_noun(
            numeral_norm,
            prefer_long=True,
        )
        plural_surface_number = self._number_surface_for_noun(
            numeral_norm,
            prefer_long=require_long,
        )

        singular_phrase = compose_phrase(
            compose_number(singular_surface_number),
            singular_candidate,
        )
        plural_phrase = compose_phrase(
            compose_number(plural_surface_number),
            plural_surface_noun,
        )

        current_is_singular = "SINGNOUN" in self._noun_number_markers(corrected_noun)
        current_is_plural = self._is_plural_like_noun(corrected_noun)
        prefers_singular = numeral_norm in self.SINGULAR_NUMBER_WORDS

        if has_long_attnum:
            return singular_phrase, [
                {
                    "word": singular_phrase,
                    "meaning": self.meaning_for(singular_candidate),
                }
            ], True

        if has_short_attnum:
            if article_prefix is not None and has_ordnum:
                ordinal_surface = self._cardinal_to_short_attnum_surface(
                    numeral_norm
                )
                corrected_noun = (
                    singular_candidate if current_is_singular else plural_candidate
                )
                alternate_noun = (
                    plural_candidate if current_is_singular else singular_candidate
                )
                corrected_phrase = compose_phrase(
                    compose_number(ordinal_surface),
                    corrected_noun,
                )
                alternate_phrase = compose_phrase(
                    compose_number(ordinal_surface),
                    alternate_noun,
                )
                ordinal_meaning = self.meaning_for(ordinal_surface).split("/", 1)[0].strip()
                corrected_meaning = self.meaning_for(corrected_noun)
                alternate_meaning = self.meaning_for(alternate_noun)

                def compose_ordinal_meaning(ordinal: str, noun_meaning: str) -> str:
                    if ordinal and noun_meaning:
                        return f"the {ordinal} {noun_meaning}"
                    if ordinal:
                        return f"the {ordinal}"
                    return f"the {noun_meaning}" if noun_meaning else "the"

                choices = [
                    {
                        "word": corrected_phrase,
                        "meaning": compose_ordinal_meaning(
                            ordinal_meaning,
                            corrected_meaning,
                        ),
                    }
                ]
                if alternate_phrase != corrected_phrase:
                    choices.append(
                        {
                            "word": alternate_phrase,
                            "meaning": compose_ordinal_meaning(
                                ordinal_meaning,
                                alternate_meaning,
                            ),
                        }
                    )
                return corrected_phrase, choices, True

            return plural_phrase, [
                {
                    "word": plural_phrase,
                    "meaning": self.meaning_for(plural_candidate),
                }
            ], True

        if has_cardnum and not has_short_attnum and not has_long_attnum:
            corrected_phrase = singular_phrase if prefers_singular else plural_phrase
            chosen_candidate = (
                singular_candidate if prefers_singular else plural_candidate
            )
            return corrected_phrase, [
                {
                    "word": corrected_phrase,
                    "meaning": self.meaning_for(chosen_candidate),
                }
            ], True

        if has_ordnum:
            return singular_phrase, [
                {
                    "word": singular_phrase,
                    "meaning": self.meaning_for(singular_candidate),
                }
            ], True

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

    def _compact_preposition_tail_variants(self, tail: str) -> list[str]:
        normalized = self._normalize_word(tail)
        variants: list[str] = []

        def add(candidate: str) -> None:
            candidate = self._normalize_word(candidate)
            if (
                candidate
                and candidate not in variants
                and self._valid_generated_surface(candidate)
            ):
                variants.append(candidate)

        if normalized.startswith("i") and len(normalized) > 1:
            stripped = normalized[1:]
            j_variant = "j" + stripped
            if self._is_imperfect_surface_candidate(j_variant):
                add(j_variant)
                add(stripped)
            else:
                add(stripped)
                add(j_variant)
        elif normalized.startswith("u") and len(normalized) > 1:
            add(normalized[1:])
            add("w" + normalized[1:])
        add(normalized)
        return variants

    def _expand_compact_long_preposition_phrase(self, word: str) -> str | None:
        normalized = self._normalize_word(word)
        # These are established contracted function-word forms, not compact
        # misspellings that should be expanded into ``fi/bi + xi/li``.
        if normalized in {"f'xi", "b'xi", "f'li", "b'li"}:
            return None
        prefix_map = {
            "x'": "xi",
            "b'": "bi",
            "f'": "fi",
            "l'": "li",
        }

        for compact_prefix, long_prefix in prefix_map.items():
            if not normalized.startswith(compact_prefix):
                continue
            tail = normalized[len(compact_prefix) :]
            if not tail:
                return None
            if compact_prefix == "l'":
                repaired_tail = self._article_tail_repair(tail)
                if repaired_tail and self._supports_l_apostrophe_tail(repaired_tail):
                    # A valid relative form (for example l'ikbar) must not be
                    # reinterpreted as the long preposition li + kbar.
                    return None
                if repaired_tail and self._is_adjective_tagged_word(repaired_tail):
                    # Ordinary adjectives take the definite article, not li.
                    return f"l-{repaired_tail}"
                if not tail.startswith(("i", "u")):
                    return None
            variants = self._compact_preposition_tail_variants(tail)
            if variants:
                return f"{long_prefix} {variants[0]}"
            return None

        if normalized.startswith("x") and len(normalized) > 2 and normalized[1] in {"i", "u"}:
            variants = self._compact_preposition_tail_variants(normalized[1:])
            if variants:
                if variants[0][:1] in self.VOWELS:
                    return f"x'{variants[0]}"
                return f"xi {variants[0]}"

        if normalized.startswith("xi") and len(normalized) > 2:
            tail = normalized[2:]
            variants = self._compact_preposition_tail_variants(tail)
            if variants:
                if variants[0][:1] in self.VOWELS:
                    return f"x'{variants[0]}"
                return f"xi {variants[0]}"

        return None

    def _attached_l_apostrophe_repair(self, word: str) -> str | None:
        normalized = self._normalize_word(word)
        if (
            not normalized.startswith("l")
            or normalized.startswith(("l'", "l-"))
            or len(normalized) <= 2
        ):
            return None

        tail = normalized[1:]
        if tail[0] not in self.VOWELS:
            return None

        tail_variants = [tail]
        if tail.endswith("om"):
            tail_variants.append(tail[:-2] + "hom")

        for tail_variant in tail_variants:
            for onset in ("għ", "h"):
                candidate = f"l'{onset}{tail_variant}"
                if self._valid_apostrophe_prefix_word(candidate) is not None:
                    return candidate
        return None

    def _simple_noun_possessive_surface_base(self, word: str) -> str | None:
        normalized = self._normalize_word(word)
        suffixes = ("kom", "hom", "ek", "ok", "ha", "na", "i", "u", "h", "k")
        for suffix in suffixes:
            if not normalized.endswith(suffix) or len(normalized) <= len(suffix):
                continue
            base = normalized[: -len(suffix)]
            if (
                base in self.dictionary_set
                and self._is_probable_noun(base)
                and base not in self._no_possession_noun_set()
                and (
                    self._noun_possessive_base_is_enabled(base)
                    or self._is_noun_tagged_word(base)
                )
            ):
                return base
        return None

    def _valid_initial_vowel_surface_word(self, word: str) -> bool:
        normalized = self._normalize_word(word)
        if not normalized.startswith(("i", "u")) or len(normalized) < 3:
            return False
        if normalized.startswith("u"):
            return False

        candidates: list[str] = []
        if normalized.startswith("i"):
            rest = normalized[1:]
            candidates.append("j" + rest)
            if len(rest) >= 2 and rest[0] == rest[1]:
                candidates.append(rest)
                candidates.append("j" + rest[1:])
        else:
            rest = normalized[1:]
            candidates.append("w" + rest)

        for candidate in candidates:
            if (
                self._valid_generated_surface(candidate)
                or self._is_imperfect_surface_candidate(candidate)
            ):
                return True
        return False

    def _initial_u_to_w_repair(self, word: str) -> str | None:
        normalized = self._normalize_word(word)
        if (
            not normalized.startswith("u")
            or len(normalized) < 3
            or self._is_manual_initial_vowel_exception(normalized)
        ):
            return None
        candidate = "w" + normalized[1:]
        if self._valid_generated_surface(candidate):
            return candidate
        return None

    def _adjacent_n_m_confusion_repair(self, word: str) -> str | None:
        normalized = self._normalize_word(word)
        replacements: list[str] = []

        if "nm" in normalized:
            replacements.append(normalized.replace("nm", "mm", 1))
        if "mn" in normalized:
            replacements.append(normalized.replace("mn", "nn", 1))

        for candidate in replacements:
            if (
                self._valid_generated_surface(candidate)
                or self._valid_initial_vowel_surface_word(candidate)
            ):
                return candidate
        return None

    def _suffix_surface_vowel_swap_repair(self, word: str) -> str | None:
        normalized = self._normalize_word(word)
        suffix_generator = getattr(self, "suffix_generator", None)
        if suffix_generator is None:
            return None

        swaps = {"u": "o", "o": "u", "i": "e", "e": "i"}
        for index, char in enumerate(normalized):
            replacement = swaps.get(char)
            if replacement is None:
                continue
            candidate = normalized[:index] + replacement + normalized[index + 1 :]
            if suffix_generator.exact_suffix_matches(candidate):
                return candidate
        return None

    def _silent_hu_noun_possessive_repair(self, word: str) -> str | None:
        normalized = self._normalize_word(word)
        if normalized.endswith("hu") and len(normalized) > 3:
            stem = normalized[:-2] + "u"
            candidates = [stem]
            if hasattr(self, "orthographic_generator"):
                candidates.extend(self.orthographic_generator.shortcut_letter_variants(stem))
            for cand in candidates:
                if self._is_recognized_surface(cand) or self._simple_noun_possessive_surface_base(cand):
                    return cand
        return None

    def _jja_ending_surface_repair(self, word: str) -> str | None:
        normalized = self._normalize_word(word)
        if normalized.endswith("jja") and len(normalized) > 3:
            single_j = normalized[:-3] + "ja"
            if hasattr(self, "doubled_letter_generator"):
                doubled = self.doubled_letter_generator.correct_missing_double(single_j)
                if doubled and self._is_recognized_surface(doubled):
                    return doubled
            if self._is_recognized_surface(single_j):
                return single_j
        return None

    def _h_bar_variant_repair(self, word: str) -> str | None:
        normalized = self._normalize_word(word)
        if "h" in normalized and "ħ" not in normalized:
            if hasattr(self, "orthographic_generator"):
                for variant in self.orthographic_generator.shortcut_letter_variants(normalized):
                    if "ħ" in variant and self._is_recognized_surface(variant):
                        return variant
            candidate = normalized.replace("h", "ħ")
            if self._is_recognized_surface(candidate):
                return candidate
        return None

    def _initial_i_imperfect_spelling_repair(self, word: str) -> str | None:
        normalized = self._normalize_word(word)
        if not normalized.startswith("i") or len(normalized) < 4:
            return None

        candidate = "j" + normalized[1:]
        variants = [candidate]
        doubled = None
        if hasattr(self, "doubled_letter_generator"):
            doubled = self.doubled_letter_generator.correct_missing_double(candidate)
        if doubled:
            variants.insert(0, self._normalize_word(doubled))

        for variant in variants:
            if self._is_imperfect_surface_candidate(variant):
                return variant
        return None

    def _initial_i_form7_surface_repair(self, word: str) -> str | None:
        normalized = self._normalize_word(word)
        if not normalized.startswith("i") or len(normalized) < 4:
            return None

        candidate = normalized[1:]
        if not self._word_starts_with_two_consonants(candidate):
            return None

        records = self._verb_records_for_surface(candidate)
        if not records:
            return None

        if any(
            record.form_class.startswith("F7") and record.tense in {"PERF", "IMP"}
            for record in records
        ):
            return candidate
        return None

    def _multi_insert_suffix_verb_repair(self, word: str) -> str | None:
        normalized = self._normalize_word(word)
        if "h" not in normalized or len(normalized) > 16:
            return None

        seeds = [normalized]
        if normalized.startswith("in") and len(normalized) > 3:
            seeds.append(normalized[1:])

        checked: set[str] = set()
        candidates: list[str] = []

        def add(candidate: str) -> None:
            candidate = self._normalize_word(candidate)
            if candidate and candidate not in checked:
                checked.add(candidate)
                candidates.append(candidate)

        for seed in seeds:
            h_positions = [index for index, char in enumerate(seed) if char == "h"]
            for h_index in h_positions:
                hbar_seed = seed[:h_index] + "ħ" + seed[h_index + 1 :]
                add(hbar_seed)
                graphemes = list(self._graphemes(hbar_seed))
                for index, token in enumerate(graphemes):
                    if (
                        token in self.VOWELS
                        or not token.isalpha()
                        or index + 1 >= len(graphemes)
                        or graphemes[index + 1] == token
                    ):
                        continue
                    doubled = self._from_graphemes(
                        graphemes[: index + 1] + [token] + graphemes[index + 1 :]
                    )
                    add(doubled)
                    if doubled.endswith("a"):
                        add(doubled[:-1] + "ha")
                    if doubled.endswith("om"):
                        add(doubled[:-2] + "hom")
                    if doubled.endswith("u"):
                        add(doubled[:-1] + "hu")

        for candidate in candidates:
            if self._valid_generated_surface(candidate):
                return candidate
        return None

    def _short_guttural_repair_shape(self, graphemes: list[str]) -> bool:
        skeleton = "".join(
            "V" if token in self.VOWELS else "C"
            for token in graphemes
            if token.isalpha() or token == "għ"
        )
        if (
            3 <= len(skeleton) <= 7
            and skeleton.count("V") == 1
            and skeleton[0] == "C"
            and skeleton[-1] == "C"
        ):
            return True
        if graphemes and graphemes[0] in {"n", "t", "j"}:
            prefixed_skeleton = "".join(
                "V" if token in self.VOWELS else "C"
                for token in graphemes[1:]
                if token.isalpha() or token == "għ"
            )
            return (
                3 <= len(prefixed_skeleton) <= 7
                and prefixed_skeleton.count("V") == 1
                and prefixed_skeleton[0] == "C"
                and prefixed_skeleton[-1] == "C"
            )
        return False

    def _missing_medial_guttural_variants(self, word: str) -> list[str]:
        normalized = self._normalize_word(word)
        graphemes = list(self._graphemes(normalized))
        if len(graphemes) < 3 or len(graphemes) > 8:
            return []
        if not self._short_guttural_repair_shape(graphemes):
            return []

        variants: list[str] = []

        def add(candidate_tokens: list[str]) -> None:
            candidate = self._from_graphemes(candidate_tokens)
            if (
                candidate
                and candidate != normalized
                and candidate not in variants
                and self._valid_generated_surface(candidate)
            ):
                variants.append(candidate)

        for index in range(1, len(graphemes)):
            previous = graphemes[index - 1]
            current = graphemes[index]
            if not previous.isalpha() or not current.isalpha():
                continue
            for guttural in ("għ", "ħ", "h"):
                add(graphemes[:index] + [guttural] + graphemes[index:])
                if previous in self.VOWELS:
                    add(graphemes[:index] + [guttural, previous] + graphemes[index:])

        return variants

    def _medial_guttural_vowel_restoration_variants(self, word: str) -> list[str]:
        normalized = self._normalize_word(word)
        graphemes = list(self._graphemes(normalized))
        if len(graphemes) < 4 or len(graphemes) > 10:
            return []
        if not self._short_guttural_repair_shape(graphemes):
            return []

        variants: list[str] = []

        def add(candidate_tokens: list[str]) -> None:
            candidate = self._from_graphemes(candidate_tokens)
            if (
                candidate
                and candidate != normalized
                and candidate not in variants
                and self._valid_generated_surface(candidate)
            ):
                variants.append(candidate)

        for index, token in enumerate(graphemes):
            if token not in {"h", "ħ", "għ"} or index == 0 or index + 1 >= len(graphemes):
                continue
            previous = graphemes[index - 1]
            following = graphemes[index + 1]
            if previous not in self.VOWELS and following in self.VOWELS:
                add(graphemes[:index] + [following] + graphemes[index:])
            if previous in self.VOWELS and following not in self.VOWELS:
                add(graphemes[: index + 1] + [previous] + graphemes[index + 1 :])

        return variants

    def _missing_h_before_r_verb_variants(self, word: str) -> list[str]:
        normalized = self._normalize_word(word)
        graphemes = list(self._graphemes(normalized))
        if len(graphemes) < 4 or len(graphemes) > 10:
            return []

        variants: list[str] = []

        def add(candidate_tokens: list[str]) -> None:
            candidate = self._from_graphemes(candidate_tokens)
            if (
                candidate
                and candidate != normalized
                and candidate not in variants
                and (
                    self._is_verb_tagged_word(candidate)
                    or self._valid_generated_surface(candidate)
                )
            ):
                variants.append(candidate)

        for index in range(1, len(graphemes)):
            if graphemes[index] != "r":
                continue
            if graphemes[index - 1] in self.VOWELS:
                add(graphemes[:index] + ["h"] + graphemes[index:])

        return variants

    def _build_missing_h_verb_repairs(self) -> dict[str, str]:
        suffix_generator = getattr(self, "suffix_generator", None)
        verb_index = getattr(suffix_generator, "verb_index", None)

        if verb_index is None:
            return {}

        repairs: dict[str, str] = {}

        for record in verb_index.iter_records():
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
        cached = self._cached_correct_word(word)
        if cached is not None:
            return cached
        normalized = self._normalize_word(word)
        fixed_time = self._fixed_time_expression_word(normalized)
        if fixed_time is not None:
            return self._store_correct_word_result(
                word,
                self._match_capitalisation(word, fixed_time),
                is_deterministic=True,
            )
        suffix_generator = getattr(self, "suffix_generator", None)
        if (
            suffix_generator is not None
            and bool(suffix_generator.exact_suffix_matches(normalized))
        ):
            return self._store_correct_word_result(
                word,
                word,
                is_deterministic=True,
                candidates=(word,),
            )
        analysis = self._phase_x_collect_candidates(word)
        phase_y = self._phase_y_basic_resolution(word, analysis)
        if phase_y is not None:
            return self._store_correct_word_result(
                word,
                phase_y,
                is_deterministic=True,
                candidates=analysis.x_candidates or (phase_y,),
            )
        # X/Y are the only automatic correction stages.  W may still expose
        # looser alternatives as suggestions, but it must never silently
        # replace unresolved text with a distant dictionary neighbour.
        return self._store_correct_word_result(
            word,
            word,
            candidates=analysis.x_candidates or (word,),
        )

    def _correct_word_uncached(self, word: str) -> str:
        if not word:
            return word

        normalized = self._normalize_word(word)
        if not normalized or len(normalized) > MAX_WORD_LENGTH:
            return word
        if normalized == "mid":
            return word
        if normalized in {"il", "l"}:
            return self._store_correct_word_result(
                word,
                self._match_capitalisation(word, normalized),
                is_deterministic=True,
            )
        social_comment_repair = self.SOCIAL_COMMENT_REPAIRS.get(normalized)
        if social_comment_repair:
            return self._store_correct_word_result(
                word,
                self._match_capitalisation(word, social_comment_repair),
                is_deterministic=True,
            )

        if normalized in self.dictionary_set:
            return self._store_correct_word_result(
                word,
                word,
                is_deterministic=True,
            )

        if self._exact_function_word_pass_through(normalized):
            return self._store_correct_word_result(
                word,
                word,
                is_deterministic=True,
            )

        if normalized in {"l-hawn", "l-hemm"}:
            return self._match_capitalisation(word, normalized[2:])

        if normalized.startswith("l-") and len(normalized) > 2:
            article_tail = normalized[2:]
            corrected_tail = self.correct_word(article_tail)
            if self._normalize_word(corrected_tail) != article_tail:
                return self._match_capitalisation(word, f"l-{self._normalize_word(corrected_tail)}")

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

        compact_long_preposition = self._expand_compact_long_preposition_phrase(normalized)
        if compact_long_preposition is not None:
            corrected = self._match_capitalisation(word, compact_long_preposition)
            return self._store_correct_word_result(
                word,
                corrected,
                is_deterministic=True,
            )

        attached_l_apostrophe = self._attached_l_apostrophe_repair(normalized)
        if attached_l_apostrophe is not None:
            corrected = self._match_capitalisation(word, attached_l_apostrophe)
            return self._store_correct_word_result(
                word,
                corrected,
                is_deterministic=True,
            )

        contracted_fb = self._repair_contracted_fb_word(word)
        if contracted_fb is not None:
            corrected_word, _ = contracted_fb
            return self._store_correct_word_result(
                word,
                corrected_word,
                is_deterministic=True,
            )

        medial_guttural_vowel = []
        if not self._is_recognized_surface(normalized):
            medial_guttural_vowel = self._medial_guttural_vowel_restoration_variants(
                normalized
            )
        if medial_guttural_vowel:
            corrected = self._match_capitalisation(word, medial_guttural_vowel[0])
            return self._store_correct_word_result(
                word,
                corrected,
                is_deterministic=True,
            )

        repaired_x_word = self._repair_x_apostrophe_word(word)
        if repaired_x_word is not None:
            return self._store_correct_word_result(
                word,
                repaired_x_word,
                is_deterministic=True,
            )

        apostrophe_prefix_word = self._valid_apostrophe_prefix_word(normalized)
        if apostrophe_prefix_word is not None:
            return self._match_capitalisation(word, apostrophe_prefix_word)

        fixed_time_expression = self._fixed_time_expression_word(normalized)
        if fixed_time_expression and self._normalize_word(fixed_time_expression) != normalized:
            corrected = self._match_capitalisation(word, fixed_time_expression)
            return self._store_correct_word_result(
                word,
                corrected,
                is_deterministic=True,
            )

        if normalized in self.dictionary_set:
            return self._store_correct_word_result(
                word,
                word,
                is_deterministic=True,
            )

        if normalized in self._no_possession_noun_set():
            return self._store_correct_word_result(
                word,
                word,
                is_deterministic=True,
            )

        if self._is_manual_initial_vowel_exception(normalized):
            return self._store_correct_word_result(
                word,
                word,
                is_deterministic=True,
            )

        if self._valid_initial_vowel_surface_word(normalized):
            return self._store_correct_word_result(
                word,
                word,
                is_deterministic=True,
            )

        initial_u_w_repair = self._initial_u_to_w_repair(normalized)
        if initial_u_w_repair:
            corrected = self._match_capitalisation(word, initial_u_w_repair)
            return self._store_correct_word_result(
                word,
                corrected,
                is_deterministic=True,
            )

        final_ghat_repair = self._final_ghat_to_ghat_e_repair(normalized)
        if final_ghat_repair:
            corrected = self._match_capitalisation(word, final_ghat_repair)
            return self._store_correct_word_result(
                word,
                corrected,
                is_deterministic=True,
            )

        if any(mark in normalized for mark in ("'", "-")):
            return self._store_correct_word_result(
                word,
                word,
                is_deterministic=True,
            )

        # ------------------------------------------------------------------
        # STRICT PRIORITY PIPELINE
        # ------------------------------------------------------------------

        initial_i_form7 = self._initial_i_form7_surface_repair(normalized)
        if initial_i_form7:
            corrected = self._match_capitalisation(word, initial_i_form7)
            return self._store_correct_word_result(
                word,
                corrected,
                is_deterministic=True,
            )

        initial_i_imperfect = self._initial_i_imperfect_spelling_repair(normalized)
        if initial_i_imperfect:
            corrected = self._match_capitalisation(word, initial_i_imperfect)
            return self._store_correct_word_result(
                word,
                corrected,
                is_deterministic=True,
            )

        orthographic_generator = getattr(
            self,
            "orthographic_generator",
            None,
        )

        noun_suffix_match = self._correct_noun_possessive_suffix(normalized)
        if noun_suffix_match:
            corrected = self._match_capitalisation(word, noun_suffix_match)
            return self._store_correct_word_result(
                word,
                corrected,
                is_deterministic=True,
            )

        if orthographic_generator is not None and hasattr(
            orthographic_generator,
            "correct_shortcut_letters",
        ):
            shortcut_match = orthographic_generator.correct_shortcut_letters(word)

            if shortcut_match:
                corrected = self._match_capitalisation(
                    word,
                    shortcut_match,
                )
                return self._store_correct_word_result(
                    word,
                    corrected,
                    is_deterministic=True,
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
                corrected = self._match_capitalisation(word, gh_shortcut_match)
                return self._store_correct_word_result(
                    word,
                    corrected,
                    is_deterministic=True,
                )

        for combined_variant in self._dictionary_i_ie_shortcut_variants(normalized):
            corrected = self._match_capitalisation(word, combined_variant)
            return self._store_correct_word_result(
                word,
                corrected,
                is_deterministic=True,
            )

        strict_terminal_apostrophe = self._strict_terminal_apostrophe_match(word)
        if strict_terminal_apostrophe:
            return self._store_correct_word_result(
                word,
                strict_terminal_apostrophe,
                is_deterministic=True,
            )

        close_apostrophe = self._close_apostrophe_ranked_match(word)
        if close_apostrophe:
            return self._store_correct_word_result(
                word,
                close_apostrophe,
                is_deterministic=True,
            )

        simple_noun_possessive = self._simple_noun_possessive_surface_base(normalized)
        if simple_noun_possessive:
            return self._store_correct_word_result(
                word,
                word,
                is_deterministic=True,
            )

        lexicalized_forms = self._lexicalized_form_variants(normalized)
        if lexicalized_forms:
            corrected = self._match_capitalisation(word, lexicalized_forms[0])
            return self._store_correct_word_result(
                word,
                corrected,
                is_deterministic=True,
            )

        pattern_repairs = self._pattern_repair_variants(normalized)
        if pattern_repairs and (
            normalized not in self.dictionary_set or normalized in {"tajru"}
        ):
            corrected = self._match_capitalisation(word, pattern_repairs[0])
            return self._store_correct_word_result(
                word,
                corrected,
                is_deterministic=True,
            )

        manual_repairs = self._manual_repair_variants(normalized)
        if manual_repairs and normalized not in self.dictionary_set:
            corrected = self._match_capitalisation(word, manual_repairs[0])
            return self._store_correct_word_result(
                word,
                corrected,
                is_deterministic=True,
            )

        adjacent_nm_repair = self._adjacent_n_m_confusion_repair(normalized)
        if adjacent_nm_repair:
            corrected = self._match_capitalisation(word, adjacent_nm_repair)
            return self._store_correct_word_result(
                word,
                corrected,
                is_deterministic=True,
            )

        suffix_vowel_swap = self._suffix_surface_vowel_swap_repair(normalized)
        if suffix_vowel_swap:
            corrected = self._match_capitalisation(word, suffix_vowel_swap)
            return self._store_correct_word_result(
                word,
                corrected,
                is_deterministic=True,
            )

        silent_hu_repair = self._silent_hu_noun_possessive_repair(normalized)
        if silent_hu_repair:
            corrected = self._match_capitalisation(word, silent_hu_repair)
            return self._store_correct_word_result(
                word,
                corrected,
                is_deterministic=True,
            )

        jja_repair = self._jja_ending_surface_repair(normalized)
        if jja_repair:
            corrected = self._match_capitalisation(word, jja_repair)
            return self._store_correct_word_result(
                word,
                corrected,
                is_deterministic=True,
            )

        hbar_repair = self._h_bar_variant_repair(normalized)
        if hbar_repair:
            corrected = self._match_capitalisation(word, hbar_repair)
            return self._store_correct_word_result(
                word,
                corrected,
                is_deterministic=True,
            )

        multi_insert_suffix = self._multi_insert_suffix_verb_repair(normalized)
        if multi_insert_suffix:
            corrected = self._match_capitalisation(word, multi_insert_suffix)
            return self._store_correct_word_result(
                word,
                corrected,
                is_deterministic=True,
            )

        if normalized in {"il", "l"}:
            corrected = self._match_capitalisation(word, normalized)
            return self._store_correct_word_result(
                word,
                corrected,
                is_deterministic=True,
            )

        compact_xi_article = self._expand_compact_xi_article(normalized)
        if compact_xi_article is not None:
            corrected = self._match_capitalisation(word, compact_xi_article)
            return self._store_correct_word_result(
                word,
                corrected,
                is_deterministic=True,
            )

        fused_preposition_rules = getattr(self, "fused_preposition_rules", None)
        if fused_preposition_rules is not None:
            fused_match = fused_preposition_rules.match(normalized)
            if fused_match is not None:
                corrected = self._match_capitalisation(word, fused_match.corrected)
                return self._store_correct_word_result(
                    word,
                    corrected,
                    is_deterministic=True,
                )

        article_rules = getattr(self, "article_phrase_rules", None)
        if article_rules is not None:
            compact_article = None
            if not (
                "-" not in normalized
                and normalized.startswith(("bix", "fix"))
                or (
                    normalized.startswith("inm")
                )
            ):
                compact_article = article_rules.match_compact_definite_article(
                    normalized,
                    previous=None,
                )
                if compact_article is None:
                    compact_article = article_rules.match_compact_preposition_article(
                        normalized,
                    )
            if compact_article is not None:
                corrected = self._match_capitalisation(
                    word,
                    compact_article.corrected,
                )
                return self._store_correct_word_result(
                    word,
                    corrected,
                    is_deterministic=True,
                )
            inline_article = self._correct_inline_article_word(
                normalized,
                previous=None,
            )
            if inline_article is not None:
                corrected = self._match_capitalisation(word, inline_article)
                return self._store_correct_word_result(
                    word,
                    corrected,
                    is_deterministic=True,
                )

        noun_suffix_match = self._correct_noun_possessive_suffix(normalized)
        if noun_suffix_match:
            corrected = self._match_capitalisation(word, noun_suffix_match)
            return self._store_correct_word_result(
                word,
                corrected,
                is_deterministic=True,
            )

        collapsed_glide = self._collapse_invalid_glide_doubling(normalized)
        if collapsed_glide:
            corrected = self._match_capitalisation(word, collapsed_glide)
            return self._store_correct_word_result(
                word,
                corrected,
                is_deterministic=True,
            )

        if article_rules is not None:
            collapsed = article_rules.collapse_three_same_consonants(normalized)
            if collapsed != normalized:
                corrected = self._match_capitalisation(word, collapsed)
                return self._store_correct_word_result(
                    word,
                    corrected,
                    is_deterministic=True,
                )

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
                corrected = self._match_capitalisation(word, i_ie_match)
                return self._store_correct_word_result(
                    word,
                    corrected,
                    is_deterministic=True,
                )

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
                corrected = self._match_capitalisation(
                    word,
                    gh_priority_match,
                )
                return self._store_correct_word_result(
                    word,
                    corrected,
                    is_deterministic=True,
                )

        # Keyboard shortcuts for Maltese letters:
        # h -> ħ, c -> ċ, z -> ż, g -> ġ
        if orthographic_generator is not None and hasattr(
            orthographic_generator,
            "correct_shortcut_letters",
        ):
            shortcut_match = orthographic_generator.correct_shortcut_letters(word)

            if shortcut_match:
                corrected = self._match_capitalisation(
                    word,
                    shortcut_match,
                )
                return self._store_correct_word_result(
                    word,
                    corrected,
                    is_deterministic=True,
                )

            if hasattr(orthographic_generator, "shortcut_letter_variants"):
                for shortcut_variant in orthographic_generator.shortcut_letter_variants(
                    normalized
                ):
                    gh_after_shortcut = orthographic_generator.correct_gh_priority(
                        shortcut_variant
                    )

                    if gh_after_shortcut:
                        corrected = self._match_capitalisation(
                            word,
                            gh_after_shortcut,
                        )
                        return self._store_correct_word_result(
                            word,
                            corrected,
                            is_deterministic=True,
                        )

            shortcut_gh_suggestion = self._shortcut_gh_suggestion_match(word)
            if shortcut_gh_suggestion:
                return self._store_correct_word_result(
                    word,
                    shortcut_gh_suggestion,
                    is_deterministic=True,
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
                corrected = self._match_capitalisation(
                    word,
                    dt_match,
                )
                return self._store_correct_word_result(
                    word,
                    corrected,
                    is_deterministic=True,
                )

        # b/p confusion for an invalid word.
        # This can cooperate with strict gh -> għ expansion.
        if orthographic_generator is not None and hasattr(
            orthographic_generator,
            "correct_b_p_confusion",
        ):
            bp_match = orthographic_generator.correct_b_p_confusion(word)

            if bp_match:
                corrected = self._match_capitalisation(
                    word,
                    bp_match,
                )
                return self._store_correct_word_result(
                    word,
                    corrected,
                    is_deterministic=True,
                )

        dt_double_match = self._correct_d_t_then_double(word)
        if dt_double_match:
            return self._store_correct_word_result(
                word,
                dt_double_match,
                is_deterministic=True,
            )

        if hasattr(self, "doubled_letter_generator"):
            j_priority_match = self.doubled_letter_generator.correct_j_priority(word)
            if j_priority_match:
                return self._store_correct_word_result(
                    word,
                    j_priority_match,
                    is_deterministic=True,
                )

        if hasattr(
            self, "suffix_generator"
        ) and self.suffix_generator.exact_suffix_matches(word):
            return self._store_correct_word_result(
                word,
                word,
                is_deterministic=True,
            )

        if orthographic_generator is not None and hasattr(
            orthographic_generator,
            "correct_extra_double",
        ):
            extra_double_match = orthographic_generator.correct_extra_double(word)

            if extra_double_match:
                corrected = self._match_capitalisation(
                    word,
                    extra_double_match,
                )
                return self._store_correct_word_result(
                    word,
                    corrected,
                    is_deterministic=True,
                )

        missing_h_verb_match = self._missing_h_verb_repair(normalized)
        if missing_h_verb_match and normalized not in {"m'hemmx"}:
            corrected = self._match_capitalisation(word, missing_h_verb_match)
            return self._store_correct_word_result(
                word,
                corrected,
                is_deterministic=True,
            )

        if orthographic_generator is not None and hasattr(
            orthographic_generator,
            "correct_missing_h_after_d",
        ):
            missing_h_after_d_match = orthographic_generator.correct_missing_h_after_d(
                word
            )

            if missing_h_after_d_match:
                corrected = self._match_capitalisation(
                    word,
                    missing_h_after_d_match,
                )
                return self._store_correct_word_result(
                    word,
                    corrected,
                    is_deterministic=True,
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
                corrected = self._match_capitalisation(
                    word,
                    dt_match,
                )
                return self._store_correct_word_result(
                    word,
                    corrected,
                    is_deterministic=True,
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
                corrected = self._match_capitalisation(
                    word,
                    i_ie_match,
                )
                return self._store_correct_word_result(
                    word,
                    corrected,
                    is_deterministic=True,
                )

        if orthographic_generator is not None and hasattr(
            orthographic_generator,
            "correct_final_aw_to_ghu",
        ):
            aw_ghu_match = orthographic_generator.correct_final_aw_to_ghu(word)

            if aw_ghu_match:
                corrected = self._match_capitalisation(
                    word,
                    aw_ghu_match,
                )
                return self._store_correct_word_result(
                    word,
                    corrected,
                    is_deterministic=True,
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
                corrected = self._match_capitalisation(
                    word,
                    final_gh_h_hbar_match,
                )
                return self._store_correct_word_result(
                    word,
                    corrected,
                    is_deterministic=True,
                )

        # Stage 0.5: Missing doubled-letter repair.
        # Example:
        #   basejt -> bassejt
        if hasattr(self, "doubled_letter_generator"):
            doubled_letter = self.doubled_letter_generator.correct_missing_double(word)
            if doubled_letter:
                return self._store_correct_word_result(
                    word,
                    doubled_letter,
                    is_deterministic=True,
                )

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
            return self._store_correct_word_result(
                word,
                exact_ortho,
                is_deterministic=True,
            )

        initial_i_match = self._validated_initial_i_match(normalized)
        if initial_i_match:
            return self._store_correct_word_result(
                word,
                self._match_capitalisation(word, initial_i_match),
                is_deterministic=True,
            )

        initial_e_match = self._validated_initial_e_match(normalized)
        if initial_e_match:
            return self._store_correct_word_result(
                word,
                self._match_capitalisation(word, initial_e_match),
                is_deterministic=True,
            )

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
                corrected = self._match_capitalisation(word, liex_lix_match)
                return self._store_correct_word_result(
                    word,
                    corrected,
                    is_deterministic=True,
                )

            generated_suffix = self.suffix_generator.correct_suffix(word)

            if generated_suffix:
                generated_norm = self._normalize_word(generated_suffix)
                if not self._suffix_anchor_compatible(normalized, generated_norm):
                    generated_suffix = None
                    suffix_parse_guard = True
                if generated_suffix and generated_norm not in self.dictionary_set:
                    generated_row = self._candidate_score(
                        normalized,
                        generated_norm,
                        stage="generated_suffix_compare",
                    )

                    # A generated suffix surface is useful only when the
                    # consonant skeleton still supports it. Without that
                    # evidence, a broad suffix parse can invent a distant
                    # imperative from an unrelated word (sormok -> isromok).
                    # Legitimate multi-letter repairs retain a positive
                    # anchor score, even where they insert għ or a suffix.
                    if generated_row.consonant_score <= 0.0:
                        generated_suffix = None
                        suffix_parse_guard = True
                    elif self._violates_ghi_sequence_rule(
                        normalized,
                        generated_norm,
                    ):
                        generated_suffix = None
                        suffix_parse_guard = True

                    # Strong suffix parses are already narrow and expensive
                    # to produce. Only consult broad dictionary scoring when
                    # the generated form is weak enough to be suspicious.
                    if generated_suffix and generated_row.score > 0.36:
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
                if generated_suffix:
                    corrected = self._match_capitalisation(word, generated_suffix)
                    return self._store_correct_word_result(
                        word,
                        corrected,
                        is_deterministic=True,
                    )

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
                return self._store_correct_word_result(
                    word,
                    exact_suffix,
                    is_deterministic=True,
                )

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

        collapsed_vowel_variants = (
            self._collapse_vowel_around_token_variants(normalized, "għ")
            + self._collapse_vowel_around_token_variants(normalized, "h")
        )
        exact_collapsed_vowel = self._try_exact_variants(
            word,
            collapsed_vowel_variants,
        )
        if exact_collapsed_vowel:
            return self._store_correct_word_result(
                word,
                exact_collapsed_vowel,
                is_deterministic=True,
            )

        # Stage 7: Remove h and check again
        remove_h = self._remove_token(normalized, "h")
        exact_rm_h = self._try_exact_variants(word, remove_h)
        if exact_rm_h:
            return self._store_correct_word_result(
                word,
                exact_rm_h,
                is_deterministic=True,
            )

        # Stage 2.5: Remove q and check again
        remove_q = self._remove_token(normalized, "q")
        exact_rm_q = self._try_exact_variants(word, remove_q)
        if exact_rm_q:
            return self._store_correct_word_result(
                word,
                exact_rm_q,
                is_deterministic=True,
            )

        # Stage 3: Insert għ next to vowels (Exact & Ranked)
        insert_gh = self._insert_token_next_to_vowels(normalized, "għ")
        exact_gh = self._try_exact_variants(word, insert_gh)
        if exact_gh:
            return self._store_correct_word_result(
                word,
                exact_gh,
                is_deterministic=True,
            )

        # Stage 4: Insert h next to vowels (Exact & Ranked)
        insert_h = self._insert_token_next_to_vowels(normalized, "h")
        exact_h = self._try_exact_variants(word, insert_h)
        if exact_h:
            return self._store_correct_word_result(
                word,
                exact_h,
                is_deterministic=True,
            )

        if suffix_parse_guard:
            return word

        # Whole-dictionary ranking is suggestion-only.  It used to be able
        # to replace an unresolved word with a distant lexical neighbour
        # (for example personali -> pproponali or pronti -> prosit).  All
        # automatic corrections above this point are dictionary-validated
        # transformations; unresolved text remains visible for the user.
        return word

        ranked_rm_h = self._try_ranked_from_variants(
            word, remove_h, stage="remove_h", score_limit=0.42
        )
        if ranked_rm_h:
            return self._store_correct_word_result(
                word,
                ranked_rm_h,
                is_deterministic=False,
            )

        ranked_gh = self._try_ranked_from_variants(
            word, insert_gh, stage="insert_gh_near_vowel", score_limit=0.42
        )
        if ranked_gh:
            return self._store_correct_word_result(
                word,
                ranked_gh,
                is_deterministic=False,
            )

        ranked_h = self._try_ranked_from_variants(
            word, insert_h, stage="insert_h_near_vowel", score_limit=0.42
        )
        if ranked_h:
            return self._store_correct_word_result(
                word,
                ranked_h,
                is_deterministic=False,
            )

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
            result_candidate = broad_best.candidate
            # If the candidate lacks the same initial letter as the input, try to
            # recover the correct prefixed form via shortcut letter substitution.
            # E.g. 'iddecidew' finds 'ddeċidew' → check 'iddeċidew'.
            if (
                result_candidate[:1] != normalized[:1]
                and normalized[:1]
            ):
                prefixed = normalized[:1] + result_candidate
                prefixed_norm = self._normalize_word(prefixed)
                if prefixed_norm in self.dictionary_set:
                    result_candidate = prefixed_norm
                else:
                    # Check shortcut substitution of the input against i-prefixed candidate
                    ortho_gen_b = getattr(self, "orthographic_generator", None)
                    if ortho_gen_b is not None and hasattr(ortho_gen_b, "shortcut_letter_variants"):
                        for sc_v in ortho_gen_b.shortcut_letter_variants(normalized):
                            sc_n = self._normalize_word(sc_v)
                            if sc_n and sc_n in self.dictionary_set:
                                result_candidate = sc_n
                                break
            if candidate_seq == target_vowel_sequence or broad_best.score <= 0.38:
                return self._match_capitalisation(word, result_candidate)

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

        override_candidates = list(self.EXACT_SUGGESTION_OVERRIDES.get(normalized, ()))

        suggestion_cache = self._request_suggestion_cache()
        cache_key = (word, limit, edit_distance_tolerance)
        cached_suggestions = suggestion_cache.get(cache_key)
        if cached_suggestions is not None:
            return list(cached_suggestions)

        contracted_fb = self._repair_contracted_fb_word(word)
        if contracted_fb is not None:
            corrected_word, _ = contracted_fb
            corrected_normalized = self._normalize_word(corrected_word)
            if corrected_normalized == normalized:
                suggestion_cache[cache_key] = tuple()
                return []
            result = [corrected_word]
            suggestion_cache[cache_key] = tuple(result)
            return result

        if override_candidates:
            ordered: list[str] = []
            if normalized in self.dictionary_set:
                ordered.append(self._match_capitalisation(word, normalized))
            for candidate in override_candidates:
                displayed = self._match_capitalisation(word, candidate)
                if self._normalize_word(displayed) not in {
                    self._normalize_word(existing) for existing in ordered
                }:
                    ordered.append(displayed)
                if len(ordered) >= limit:
                    break
            if ordered:
                suggestion_cache[cache_key] = tuple(ordered[:limit])
                return ordered[:limit]

        analysis = self._get_token_analysis(word)
        if analysis is None:
            analysis = self._phase_x_collect_candidates(word)
        if analysis is not None and analysis.is_deterministic:
            deterministic = self._phase_w_seed_suggestions(
                word,
                analysis,
                limit=limit,
            )
            if not deterministic and analysis.corrected:
                deterministic = [analysis.corrected]
            if deterministic:
                result = deterministic[:limit]
                suggestion_cache[cache_key] = tuple(result)
                return result

        if self._is_initial_capitalized(word):
            if normalized in self.place_word_set:
                result = [self.place_word_display.get(normalized, word)]
                suggestion_cache[cache_key] = tuple(result)
                return result
            if normalized in self.dictionary_set:
                result = [self._match_capitalisation(word, normalized)]
                suggestion_cache[cache_key] = tuple(result)
                return result
            corrected_place = self._correct_place_word(word)
            if corrected_place:
                result = [corrected_place]
                suggestion_cache[cache_key] = tuple(result)
                return result
            strict = self._try_exact_variants(
                word,
                self._strict_lookup_variants(normalized),
            )
            result = [strict] if strict else []
            suggestion_cache[cache_key] = tuple(result)
            return result

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
                candidate.endswith("ħek")
                and not any(marker in normalized for marker in ("h", "ħ", "għ", "gh"))
            ):
                return
            suffix_candidates = self._valid_suffix_surface_candidates(candidate)
            if (
                candidate not in trusted_generated
                and
                suffix_candidates == []
                and hasattr(self, "suffix_generator")
                and self.suffix_generator.candidates_for_surface(candidate)
            ):
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
        for lexicalized in lexicalized_forms:
            add_generated(lexicalized)
            # Alternatives here are an explicit, bounded lexical rule (for
            # example bilqiegħda / bil-qiegħda), not a fuzzy candidate. Keep
            # them even when the general whole-word plausibility guard rejects
            # the spacing or hyphen difference.
            lexicalized_norm = self._normalize_word(lexicalized)
            if lexicalized_norm and lexicalized_norm not in suggestions:
                suggestions.append(lexicalized_norm)
            if len(suggestions) >= limit:
                return suggestions[:limit]

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
            doubled_letter = getattr(self, "doubled_letter_generator", None)
            if doubled_letter is not None:
                for candidate in doubled_letter.missing_double_variants(normalized):
                    if candidate in self.dictionary_set:
                        add_generated(candidate)
                        if len(suggestions) >= limit:
                            return suggestions[:limit]
            early_orthographic = getattr(self, "orthographic_generator", None)
            if early_orthographic is not None and hasattr(
                early_orthographic,
                "dictionary_insert_h_after_d_variants",
            ):
                for candidate in early_orthographic.dictionary_insert_h_after_d_variants(
                    normalized
                ):
                    add_generated(candidate)
                    if len(suggestions) >= limit:
                        return suggestions[:limit]
            for candidate in self._missing_h_before_r_verb_variants(normalized):
                trusted_generated.add(self._normalize_word(candidate))
                add_generated(candidate)
                if len(suggestions) >= limit:
                    return suggestions[:limit]
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
            restorative_candidates = (
                self._medial_guttural_vowel_restoration_variants(normalized)
                + self._missing_medial_guttural_variants(normalized)
            )
            trusted_generated.update(
                self._normalize_word(candidate)
                for candidate in restorative_candidates
            )
            for candidate in restorative_candidates:
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
                restorative_candidates = (
                    self._medial_guttural_vowel_restoration_variants(normalized)
                    + self._missing_medial_guttural_variants(normalized)
                )
                trusted_generated.update(
                    self._normalize_word(candidate)
                    for candidate in restorative_candidates
                )
                for candidate in restorative_candidates:
                    add_generated(candidate)
                    if len(suggestions) >= limit:
                        return suggestions[:limit]
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
            restorative_candidates = (
                self._medial_guttural_vowel_restoration_variants(normalized)
                + self._missing_medial_guttural_variants(normalized)
            )
            trusted_generated.update(
                self._normalize_word(candidate)
                for candidate in restorative_candidates
            )
            for candidate in restorative_candidates:
                add_generated(candidate)
                if len(suggestions) >= limit:
                    return suggestions[:limit]
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
                if self._blocks_initial_stop_confusion(normalized, candidate):
                    continue
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
                if self._blocks_initial_stop_confusion(normalized, candidate):
                    continue
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
            self._initial_i_variants(normalized),
            self._suffix_repair_variants(normalized),
            self._collapse_vowel_around_token_variants(normalized, "għ"),
            self._collapse_vowel_around_token_variants(normalized, "h"),
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

        symspell_candidates = self._symspell_candidates(normalized)
        if ENABLE_SYMSPELL_CANDIDATES and symspell_candidates and not SYMSPELL_SHADOW_MODE:
            candidates_to_score = set(symspell_candidates)
        else:
            candidates_to_score = set(self._get_candidates(normalized))
            if symspell_candidates:
                candidates_to_score.update(symspell_candidates)
        for variant in lexicalized_forms:
            variant_symspell = self._symspell_candidates(variant)
            if ENABLE_SYMSPELL_CANDIDATES and variant_symspell and not SYMSPELL_SHADOW_MODE:
                candidates_to_score.update(variant_symspell)
            else:
                candidates_to_score.update(self._get_candidates(variant))
                candidates_to_score.update(variant_symspell)

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

        return self._store_suggest_result(
            word,
            limit,
            edit_distance_tolerance,
            suggestions,
        )

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

    @lru_cache(maxsize=2048)
    def meaning_for(self, word: str) -> str:
        if getattr(self._local, "suppress_meanings", False):
            return ""
        normalized = self._normalize_word(word)
        ta_direct_object_meaning = self._ta_direct_object_gloss(normalized)
        if ta_direct_object_meaning:
            return ta_direct_object_meaning
        direct_object_base = self._direct_object_h_base(normalized)
        if direct_object_base:
            meanings = meaning_index.meanings_for(direct_object_base)
            if meanings:
                meaning = meanings[0].replace(" sb", " him")
                if direct_object_base == "ġie":
                    meaning = meaning.replace("he came", "he/it came") + " for him"
                return meaning
        pronoun_meanings = {
            "hu": "he",
            "huwa": "he",
            "hi": "she",
            "hija": "she",
            "huma": "they",
            "jien": "I",
            "jiena": "I",
            "aħna": "we",
            "inti": "you",
            "int": "you",
            "intom": "you all",
        }
        if normalized in pronoun_meanings:
            return pronoun_meanings[normalized]
        if normalized.startswith("m'") and normalized[2:] in pronoun_meanings:
            return f"not {pronoun_meanings[normalized[2:]]}"
        if normalized == "ħu":
            return "brother, male sibling"
        if normalized == "ħi":
            return "friend"
        if normalized in {"llajs", "llaħwa", "llami"}:
            return "expression of awe"
        if normalized in {"u"}:
            return "and"
        if normalized in {"innerdjat", "innerdjata", "innerdjati"}:
            return "irritated, annoyed"
        if normalized in {"kif", "kief"}:
            meanings = meaning_index.meanings_for(normalized)
            if meanings:
                seen: list[str] = []
                for meaning in meanings:
                    if meaning and meaning not in seen:
                        seen.append(meaning)
                if seen:
                    return " / ".join(seen)
        contracted_fb = self._repair_contracted_fb_word(word)
        if contracted_fb is not None:
            _, safe_tail = contracted_fb
            return self.meaning_for(safe_tail)

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
            repaired_tail = self._article_tail_repair(relative_tail) or self.correct_word(
                relative_tail
            )
            lookup_tail = (
                self._normalize_word(repaired_tail)
                if repaired_tail and self._normalize_word(repaired_tail) != relative_tail
                else relative_tail
            )
            if self._is_verb_tagged_word(lookup_tail):
                relative_meaning = self.meaning_for(lookup_tail)
            else:
                relative_meaning = meaning_index.meaning_for(lookup_tail)
            if not relative_meaning:
                relative_meaning = self.meaning_for(lookup_tail)
            if relative_meaning:
                if self._is_adjective_tagged_word(lookup_tail):
                    return f"which is {relative_meaning}"
                if self._is_verb_tagged_word(lookup_tail):
                    if relative_meaning.casefold().startswith("to "):
                        relative_meaning = relative_meaning[3:]
                    for pronoun_prefix in ("he ", "she ", "it "):
                        if relative_meaning.casefold().startswith(pronoun_prefix):
                            relative_meaning = relative_meaning[len(pronoun_prefix):]
                            break
                    return f"which {relative_meaning}"
                if self._is_adverb_tagged_word(lookup_tail) or self._is_preposition_tagged_word(
                    lookup_tail
                ):
                    return f"which {relative_meaning}"

        noun_base = self._noun_possessive_base_for_surface(word)
        if noun_base:
            noun_meaning = meaning_index.meaning_for(noun_base)
            if noun_meaning:
                return noun_meaning

        tag_meanings: list[str] = []
        for tag in self.word_tags.get(normalized, ()):
            if not tag.startswith(("T-", "Q-", "S-", "AS-", "IS-")):
                continue
            meaning = extract_meaning_from_payload(tag)
            if meaning and meaning not in tag_meanings:
                tag_meanings.append(meaning)
        if tag_meanings:
            return " / ".join(tag_meanings)

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

        for candidate in self._valid_suffix_surface_candidates(word):
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
        profiler = current_profiler()
        if profiler is None:
            return self._ambiguity_choices_uncached(
                original_word,
                corrected_word,
                limit=limit,
                edit_distance_tolerance=edit_distance_tolerance,
            )

        key = (original_word, corrected_word, limit, edit_distance_tolerance)
        hit, value = profiler.cache_get("ambiguity_choices", key)
        if hit:
            return value

        with profiler.span(
            "ambiguity_choices",
            token=original_word,
            corrected=corrected_word,
            cache_hit=False,
            ambiguity_invoked=True,
        ):
            value = self._ambiguity_choices_uncached(
                original_word,
                corrected_word,
                limit=limit,
                edit_distance_tolerance=edit_distance_tolerance,
            )
        return profiler.cache_set("ambiguity_choices", key, value)

    def _near_homograph_suggestion_variants(self, base: str) -> list[str]:
        base_norm = self._normalize_word(base)
        if not base_norm or len(base_norm) < 2:
            return []
        cands: list[str] = []
        g = list(self._graphemes(base_norm))

        # 1. Apostrophe toggle (closest near-homograph)
        if base_norm.endswith("'"):
            cands.append(base_norm[:-1])
        else:
            cands.append(base_norm + "'")

        # 2. De-gemination (remove a doubled consonant)
        for i in range(len(g)):
            ch = g[i]
            if ch in {"m", "t", "r", "p", "k", "z", "ż", "s", "ċ", "n", "l", "b", "d", "f", "ġ"}:
                if i + 1 < len(g) and g[i + 1] == ch:
                    singled = self._from_graphemes(g[:i] + g[i + 1:])
                    cands.append(singled)
                    cands.append(singled + "'")

        # 3. Re-gemination (add a doubled consonant — less likely)
        for i in range(len(g)):
            ch = g[i]
            if ch in {"m", "t", "r", "p", "k", "z", "ż", "s", "ċ", "n", "l", "b", "d", "f", "ġ"}:
                if not (i + 1 < len(g) and g[i + 1] == ch):
                    doubled = self._from_graphemes(g[:i] + [ch] + g[i:])
                    cands.append(doubled)
                    cands.append(doubled + "'")

        matches: list[str] = []
        for cand in cands:
            norm = self._normalize_word(cand)
            if norm and norm != base_norm and self._is_recognized_surface(norm):
                if norm not in matches:
                    matches.append(norm)
        return matches

    def _ambiguity_choices_uncached(
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

        if original_norm in self.QIEGHED_SPELLING_VARIANTS:
            return self._finalize_ambiguity_choices(
                original_word,
                corrected_norm,
                ordered,
                limit=limit,
            )

        if self._direct_object_h_base(corrected_norm) is not None:
            return self._finalize_ambiguity_choices(
                original_word,
                corrected_norm,
                ordered,
                limit=limit,
            )

        # A validated possessive surface is already a complete lexical form.
        # Keeping fuzzy neighbours here produced false alternatives such as
        # ``kazzu`` for the possessive noun ``kasu``.
        if (
            len(self._graphemes(corrected_norm)) >= 4
            and self._correct_noun_possessive_suffix(corrected_norm) == corrected_norm
        ):
            return self._finalize_ambiguity_choices(
                original_word,
                corrected_norm,
                ordered,
                limit=limit,
            )

        # The compact direct-object forms of ``ta`` are already exact verb
        # surfaces. Do not contaminate their popovers with fuzzy lookalikes.
        if self._ta_direct_object_gloss(corrected_norm):
            return self._finalize_ambiguity_choices(
                original_word,
                corrected_norm,
                ordered,
                limit=limit,
            )

        for candidate in self._near_homograph_suggestion_variants(original_norm):
            add(candidate)
            if len(ordered) >= limit:
                break

        if original_norm in {"xhin", "xħin", "x'ħin"}:
            add("xħin")
            add("x'ħin")
            return self._finalize_ambiguity_choices(
                original_word,
                corrected_norm,
                ordered,
                limit=limit,
            )

        if corrected_norm == original_norm and original_norm in self.EXACT_SUGGESTION_OVERRIDES:
            for candidate in self.EXACT_SUGGESTION_OVERRIDES[original_norm]:
                add(candidate)
                if len(ordered) >= limit:
                    break
            return self._finalize_ambiguity_choices(
                original_word,
                corrected_norm,
                ordered,
                limit=limit,
            )

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
                if self._blocks_initial_stop_confusion(original_norm, alternative):
                    continue
                add(alternative)

                if len(ordered) >= limit:
                    break

        for suggestion in suggestions:
            suggestion_norm = self._normalize_word(suggestion)

            if suggestion_norm == corrected_norm:
                continue
            if self._blocks_initial_stop_confusion(original_norm, suggestion_norm):
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

        return self._finalize_ambiguity_choices(
            original_word,
            corrected_norm,
            ordered,
            limit=limit,
        )

    def _finalize_ambiguity_choices(
        self,
        original_word: str,
        corrected_norm: str,
        ordered: list[str],
        *,
        limit: int,
    ) -> list[dict]:
        choices: list[dict] = []
        seen_norms: set[str] = set()
        seen_canonical: set[str] = set()
        corrected_canonical = self._canonical_suggestion_key(corrected_norm)

        for word in ordered:
            displayed_word = self._match_capitalisation(original_word, word)
            displayed_norm = self._normalize_word(displayed_word)
            displayed_canonical = self._canonical_suggestion_key(displayed_word)

            if not displayed_norm or displayed_norm in seen_norms:
                continue
            if not self._is_recognized_surface(displayed_norm):
                continue
            if choices and displayed_canonical == corrected_canonical:
                continue
            if displayed_canonical in seen_canonical:
                continue

            choices.append(
                {
                    "word": displayed_word,
                    "meaning": self.meaning_for(word),
                }
            )
            seen_norms.add(displayed_norm)
            seen_canonical.add(displayed_canonical)
            if len(choices) >= limit:
                break

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
        if hasattr(self, "_near_homograph_suggestion_variants") and self._near_homograph_suggestion_variants(original_norm):
            return max(default_limit, 3)
        return default_limit

    def _long_text_bulk_mode(self, text: str, *, word_count: int) -> bool:
        return len(text) >= LONG_TEXT_CHAR_THRESHOLD or word_count >= LONG_TEXT_WORD_THRESHOLD

    def _bulk_ambiguity_choices(
        self,
        original_word: str,
        corrected_word: str,
        *,
        limit: int = 3,
    ) -> list[dict]:
        original_norm = self._normalize_word(original_word)
        corrected_norm = self._normalize_word(corrected_word)
        if original_norm == corrected_norm:
            overrides = self.EXACT_SUGGESTION_OVERRIDES.get(original_norm, ())
            if not overrides:
                return []
            return self._finalize_ambiguity_choices(
                original_word,
                corrected_norm,
                [corrected_norm, *overrides],
                limit=limit,
            )

        ordered = [corrected_norm]
        analysis = self._get_token_analysis(original_word)
        if analysis is not None:
            for candidate in analysis.x_candidates:
                candidate = self._normalize_word(candidate)
                if candidate and candidate not in ordered:
                    ordered.append(candidate)
        return self._finalize_ambiguity_choices(
            original_word,
            corrected_norm,
            ordered,
            limit=limit,
        )

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

        text = repair_mojibake_text(text)

        self._reset_request_token_cache()

        tokens: list[dict] = []
        corrected_parts: list[str] = []
        quote_matches = [
            quote
            for quote in self.ENGLISH_QUOTES_PATTERN.finditer(text)
            if not (quote.start() > 0 and text[quote.start() - 1].isalpha())
        ]

        word_matches = []
        for m in self.WORD_PATTERN.finditer(text):
            overlaps_quote = any(
                m.start() < q.end() and m.end() > q.start() for q in quote_matches
            )
            if not overlaps_quote:
                word_matches.append(m)

        bulk_mode = self._long_text_bulk_mode(text, word_count=len(word_matches))
        self._local.bulk_mode = bulk_mode
        if not getattr(self._local, "suppress_meanings", False):
            self.meaning_for.cache_clear()
        # The main corrected text never needs lexical meanings.  They are
        # populated only for actual alternatives immediately before return.
        self._local.suppress_meanings = True
        effective_tolerance = 1 if bulk_mode else edit_distance_tolerance

        matches = []
        for q in quote_matches:
            matches.append(
                UnifiedMatch(
                    q.start(),
                    q.end(),
                    q.group(0),
                    True,
                    q.group("inner"),
                    self._accepted_exact_english(q.group("inner")),
                )
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
        after_punctuation_initial_vowel = True

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
                if self._breaks_empathetic_i_context(raw_text):
                    previous_surface_word = None
                if self._starts_initial_vowel_context(raw_text):
                    after_punctuation_initial_vowel = True

            if getattr(match, "is_quote", False):
                inner_text = match.inner_text
                canonical_english, quote_is_recognized = self._correct_quoted_english_text(
                    inner_text
                )
                tokens.append(
                    {
                        "type": "word",
                        "original": match.group(0),
                        "corrected": canonical_english,
                        "meaning": "",
                        "ambiguous": False,
                        "crucial": False,
                        "choices": [],
                        "name_like": False,
                        "is_quote": True,
                        "force_unrecognized": not quote_is_recognized,
                    }
                )
                corrected_parts.append(canonical_english)
                previous_surface_word = None
                last_end = match.end()
                index += 1
                continue

            original_word = match.group(0)
            english_phrase = self._exact_unquoted_english_phrase_at(
                text,
                matches,
                index,
            )
            if english_phrase is not None:
                original_phrase, consumed = english_phrase
                tokens.append(
                    self._english_token(
                        original=original_phrase,
                        corrected=original_phrase,
                        inner_text=original_phrase,
                    )
                )
                corrected_parts.append(original_phrase)
                previous_surface_word = original_phrase
                last_end = matches[index + consumed - 1].end()
                index += consumed
                continue

            original_norm = self._normalize_word(original_word)
            sentence_initial = self._is_sentence_initial_position(text, match.start())
            if (
                not sentence_initial
                and original_word[:1].isupper()
                and "\n" in text[last_end : match.start()]
            ):
                # Preserve a deliberately capitalised new paragraph without
                # treating every wrapped lowercase line as a new sentence.
                sentence_initial = True
            prefer_initial_vowel_surface = (
                sentence_initial
                or after_punctuation_initial_vowel
                or (
                    previous_surface_word is not None
                    and not self._word_ends_with_vowel(previous_surface_word)
                )
            )
            after_punctuation_initial_vowel = False

            if (
                original_norm in {"l", "il"}
                and index + 1 < len(matches)
                and self._normalize_word(matches[index + 1].group(0))
                in {"hawn", "hemm", "hinn"}
            ):
                tail_norm = self._normalize_word(matches[index + 1].group(0))
                corrected_phrase = f"'l {tail_norm}"
                tokens.append(
                    {
                        "type": "phrase",
                        "original": text[match.start() : matches[index + 1].end()],
                        "corrected": corrected_phrase,
                        "meaning": "",
                        "ambiguous": False,
                        "crucial": False,
                        "choices": [],
                        "name_like": False,
                    }
                )
                corrected_parts.append(corrected_phrase)
                previous_surface_word = tail_norm
                last_end = matches[index + 1].end()
                index += 2
                continue

            if original_norm.startswith("x'"):
                corrected_x = self.correct_word(original_word)
                if self._normalize_word(corrected_x) != original_norm:
                    surface_x = self._apply_surface_case(
                        original_word,
                        corrected_x,
                        sentence_initial=sentence_initial,
                    )
                    tokens.append(
                        {
                            "type": "word",
                            "original": original_word,
                            "corrected": surface_x,
                            "meaning": "",
                            "ambiguous": False,
                            "crucial": False,
                            "choices": [],
                            "name_like": False,
                        }
                    )
                    corrected_parts.append(surface_x)
                    previous_surface_word = self._normalize_word(surface_x)
                    last_end = match.end()
                    index += 1
                    continue

            # ``għal xi`` is a live two-word phrase, never the assimilated
            # article ``għax-``. Resolve the tail first and preserve the
            # phrase before generic article parsing sees the leading token.
            if (
                original_norm in {"għal", "ghal"}
                and index + 1 < len(matches)
                and self._normalize_word(matches[index + 1].group(0)) == "xi"
            ):
                corrected_phrase = "għal xi"
                if sentence_initial:
                    corrected_phrase = self._capitalize_first_letter(corrected_phrase)
                tokens.append(
                    {
                        "type": "phrase",
                        "original": text[match.start() : matches[index + 1].end()],
                        "corrected": corrected_phrase,
                        "meaning": "",
                        "ambiguous": False,
                        "crucial": False,
                        "choices": [],
                        "name_like": False,
                    }
                )
                corrected_parts.append(corrected_phrase)
                previous_surface_word = "xi"
                last_end = matches[index + 1].end()
                index += 2
                continue

            if article_rules is not None and index + 1 < len(matches):
                between_tokens = text[matches[index].end() : matches[index + 1].start()]
                if "-" in between_tokens and not between_tokens.replace("-", "").strip():
                    previous_word = None if sentence_initial else (
                        word_tokens[index - 1].text if index > 0 else None
                    )
                    article_match = article_rules.match_hyphenated_article_after(
                        f"{original_word}-{word_tokens[index + 1].text}",
                        previous=previous_word,
                    )
                    if article_match is not None:
                        corrected_phrase = (
                            self._capitalize_first_letter(article_match.corrected)
                            if sentence_initial
                            else article_match.corrected
                        )
                        is_ambiguous, is_crucial = token_choice_state(
                            article_match.choices,
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
                                "choices": article_match.choices,
                            }
                        )
                        corrected_parts.append(corrected_phrase)
                        previous_surface_word = self._normalize_word(
                            corrected_phrase.split()[-1]
                        )
                        last_end = matches[index + 1].end()
                        index += 2
                        continue

            # The negative particle remains bare before consonant-initial
            # words. It contracts only before a vowel/guttural or an
            # explicitly parsed article, never through generic h insertion.
            if original_norm == "ma" and index + 1 < len(matches):
                next_norm = self._normalize_word(matches[index + 1].group(0))
                article_heads = {
                    "l", "il", "ic", "iċ", "id", "in", "ir", "is",
                    "it", "ix", "iz", "iż", "d", "n", "r", "s", "t", "x", "z", "ż",
                }
                if (
                    next_norm not in article_heads
                    and not next_norm.startswith(("għ", "gh", "h"))
                    and not (next_norm and next_norm[0] in self.VOWELS)
                ):
                    corrected_ma = self._apply_surface_case(
                        original_word,
                        "ma",
                        sentence_initial=sentence_initial,
                    )
                    tokens.append(
                        {
                            "type": "word",
                            "original": original_word,
                            "corrected": corrected_ma,
                            "meaning": "",
                            "ambiguous": False,
                            "crucial": False,
                            "choices": [],
                            "name_like": False,
                        }
                    )
                    corrected_parts.append(corrected_ma)
                    previous_surface_word = "ma"
                    last_end = match.end()
                    index += 1
                    continue

            # Bare f/b before a noun or possessive surface is the apostrophe
            # preposition. Do not route it through verb suffix recovery.
            if original_norm in {"f", "b"} and index + 1 < len(matches):
                next_word = matches[index + 1].group(0)
                next_norm = self._normalize_word(next_word)
                corrected_next = self.correct_word(next_word)
                corrected_next_norm = self._normalize_word(corrected_next)
                possessive_endings = ("i", "ek", "ok", "u", "ha", "na", "kom", "hom")
                noun_like_tail = (
                    self._is_noun_tagged_word(corrected_next_norm)
                    or self._correct_noun_possessive_suffix(corrected_next_norm)
                    == corrected_next_norm
                    or corrected_next_norm.endswith(possessive_endings)
                )
                if noun_like_tail and not self._is_verb_tagged_word(
                    corrected_next_norm
                ):
                    corrected_phrase = self._apply_surface_case(
                        original_word,
                        f"{original_norm}'{corrected_next_norm}",
                        sentence_initial=sentence_initial,
                    )
                    tokens.append(
                        {
                            "type": "phrase",
                            "original": text[match.start() : matches[index + 1].end()],
                            "corrected": corrected_phrase,
                            "ambiguous": False,
                            "crucial": False,
                            "choices": [
                                {
                                    "word": corrected_phrase,
                                    "meaning": self.meaning_for(corrected_next_norm),
                                }
                            ],
                            "name_like": False,
                        }
                    )
                    corrected_parts.append(corrected_phrase)
                    previous_surface_word = corrected_next_norm
                    last_end = matches[index + 1].end()
                    index += 2
                    continue
                if not self._is_verb_tagged_word(corrected_next_norm):
                    fallback_tail = corrected_next_norm or next_norm
                    corrected_phrase = self._apply_surface_case(
                        original_word,
                        f"{original_norm}'{fallback_tail}",
                        sentence_initial=sentence_initial,
                    )
                    tokens.append(
                        {
                            "type": "phrase",
                            "original": text[match.start() : matches[index + 1].end()],
                            "corrected": corrected_phrase,
                            "ambiguous": False,
                            "crucial": False,
                            "choices": [],
                            "name_like": False,
                        }
                    )
                    corrected_parts.append(corrected_phrase)
                    previous_surface_word = fallback_tail
                    last_end = matches[index + 1].end()
                    index += 2
                    continue

            if index + 1 < len(matches) and not getattr(matches[index + 1], "is_quote", False):
                early_article_fallback = self._split_article_unknown_tail(
                    original_word,
                    matches[index + 1].group(0),
                    previous=word_tokens[index - 1].text if index > 0 and not sentence_initial else None,
                )
                if early_article_fallback is not None and original_norm not in {
                    "il", "l", "ic", "iċ", "id", "in", "ir", "is", "it", "ix", "iz", "iż",
                }:
                    corrected_phrase = (
                        self._capitalize_first_letter(early_article_fallback)
                        if sentence_initial
                        else early_article_fallback
                    )
                    tokens.append(
                        {
                            "type": "phrase",
                            "original": text[match.start() : matches[index + 1].end()],
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

            # ``li`` is a relative pronoun before a verb, not a shortened
            # article. This early guard retains a corrected verb surface.
            if original_norm == "li" and index + 1 < len(matches):
                protected_next = self.correct_word(matches[index + 1].group(0))
                if self._is_verb_tagged_word(protected_next):
                    corrected_phrase = f"li {protected_next}"
                    if sentence_initial:
                        corrected_phrase = self._capitalize_first_letter(
                            corrected_phrase
                        )
                    tokens.append(
                        {
                            "type": "phrase",
                            "original": text[match.start() : matches[index + 1].end()],
                            "corrected": corrected_phrase,
                            "meaning": "",
                            "ambiguous": False,
                            "crucial": False,
                            "choices": [],
                            "name_like": False,
                        }
                    )
                    corrected_parts.append(corrected_phrase)
                    previous_surface_word = self._normalize_word(protected_next)
                    last_end = matches[index + 1].end()
                    index += 2
                    continue

            # Exact English words may take a Maltese article. Route these
            # through the standard article matcher so its dictionary-backed
            # Maltese equivalents remain available as blue alternatives.
            if (
                article_rules is not None
                and index + 1 < len(matches)
                and original_norm in {
                    "il", "l", "ic", "iċ", "id", "in", "ir", "is",
                    "it", "ix", "iz", "iż",
                }
                and self._accepted_exact_english(matches[index + 1].group(0))
            ):
                english_article = article_rules.match_split_article(word_tokens, index)
                if english_article is not None:
                    corrected_phrase = (
                        self._capitalize_first_letter(english_article.corrected)
                        if sentence_initial
                        else english_article.corrected
                    )
                    phrase_choices = english_article.choices
                    if sentence_initial:
                        phrase_choices = [
                            {
                                **choice,
                                "word": self._capitalize_first_letter(
                                    choice.get("word", "")
                                ),
                            }
                            for choice in phrase_choices
                        ]
                    is_ambiguous, is_crucial = token_choice_state(
                        phrase_choices,
                        force_crucial=True,
                    )
                    tokens.append(
                        {
                            "type": "phrase",
                            "original": text[match.start() : matches[index + 1].end()],
                            "corrected": corrected_phrase,
                            "ambiguous": is_ambiguous,
                            "crucial": is_crucial,
                            "choices": phrase_choices,
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

            # A spaced ``bħall`` cannot be an already-fused article: that
            # form requires an article tail (for example ``bħall-ktieb`` or
            # ``bħas-soltu``).  Treat it as the comparison word ``bħal`` and
            # let the following word use the regular correction pipeline.
            if (
                original_norm in {"bħall", "bhall"}
                and index + 1 < len(matches)
                and not getattr(matches[index + 1], "is_quote", False)
                and text[match.end() : matches[index + 1].start()].isspace()
            ):
                following_norm = self._normalize_word(matches[index + 1].group(0))
                if following_norm not in {
                    "il", "l", "ic", "iċ", "id", "in", "ir", "is",
                    "it", "ix", "iz", "iż",
                }:
                    corrected_next = self.correct_word(matches[index + 1].group(0))
                    corrected_phrase = f"bħal {corrected_next}"
                    phrase_choices = []
                    article_rules = getattr(self, "article_phrase_rules", None)
                    if article_rules is not None:
                        corrected_next_norm = self._normalize_word(corrected_next)
                        article_form = article_rules.preposition_article_form(
                            "bħall",
                            corrected_next_norm,
                        )
                        meaning = self.meaning_for(corrected_next_norm)
                        for choice_word in (corrected_phrase, article_form):
                            if choice_word and all(
                                self._normalize_word(choice.get("word", ""))
                                != self._normalize_word(choice_word)
                                for choice in phrase_choices
                            ):
                                phrase_choices.append(
                                    {"word": choice_word, "meaning": meaning}
                                )
                    if sentence_initial:
                        corrected_phrase = self._capitalize_first_letter(corrected_phrase)
                        phrase_choices = [
                            {
                                **choice,
                                "word": self._capitalize_first_letter(
                                    choice.get("word", "")
                                ),
                            }
                            for choice in phrase_choices
                        ]
                    is_ambiguous, is_crucial = token_choice_state(
                        phrase_choices,
                        force_crucial=True,
                    )
                    tokens.append(
                        {
                            "type": "phrase",
                            "original": text[
                                match.start() : matches[index + 1].end()
                            ],
                            "corrected": corrected_phrase,
                            "meaning": "",
                            "ambiguous": is_ambiguous,
                            "crucial": is_crucial,
                            "choices": phrase_choices,
                            "name_like": False,
                        }
                    )
                    corrected_parts.append(corrected_phrase)
                    previous_surface_word = self._normalize_word(corrected_next)
                    last_end = matches[index + 1].end()
                    index += 2
                    continue

            if (
                original_norm in {"bhas", "bħas"}
                and index + 1 < len(matches)
                and not getattr(matches[index + 1], "is_quote", False)
                and self._normalize_word(matches[index + 1].group(0))
                in {"soltu", "solitu"}
            ):
                corrected_phrase = "bħas-soltu"
                if sentence_initial:
                    corrected_phrase = self._capitalize_first_letter(corrected_phrase)
                tokens.append(
                    {
                        "type": "phrase",
                        "original": text[match.start() : matches[index + 1].end()],
                        "corrected": corrected_phrase,
                        "meaning": "",
                        "ambiguous": False,
                        "crucial": False,
                        "choices": [],
                        "name_like": False,
                    }
                )
                corrected_parts.append(corrected_phrase)
                previous_surface_word = "soltu"
                last_end = matches[index + 1].end()
                index += 2
                continue

            split_compact_prefixes = {
                "mis": "mis",
                "miss": "mis",
                "mas": "mas",
                "ghat": "għat",
                "għat": "għat",
                "ghad": "għad",
                "għad": "għad",
                "lis": "lis",
            }
            if (
                original_norm in split_compact_prefixes
                and index + 1 < len(matches)
                and not getattr(matches[index + 1], "is_quote", False)
                and text[match.end() : matches[index + 1].start()].isspace()
            ):
                next_word = matches[index + 1].group(0)
                next_norm = self._normalize_word(next_word)
                if self._accepted_article_english(next_word):
                    compact_tail = self._english_key(next_word)
                else:
                    corrected_next = self.correct_word(next_word)
                    compact_tail = self._normalize_word(corrected_next)
                if compact_tail and (
                    self._accepted_article_english(compact_tail)
                    or self._is_recognized_surface(compact_tail)
                    or self._is_noun_tagged_word(compact_tail)
                    or self._is_adjective_tagged_word(compact_tail)
                ):
                    corrected_phrase = (
                        f"{split_compact_prefixes[original_norm]}-{compact_tail}"
                    )
                    if sentence_initial:
                        corrected_phrase = self._capitalize_first_letter(
                            corrected_phrase
                        )
                    tokens.append(
                        {
                            "type": "phrase",
                            "original": text[match.start() : matches[index + 1].end()],
                            "corrected": corrected_phrase,
                            "meaning": "",
                            "ambiguous": False,
                            "crucial": False,
                            "choices": [],
                            "name_like": False,
                        }
                    )
                    corrected_parts.append(corrected_phrase)
                    previous_surface_word = compact_tail
                    last_end = matches[index + 1].end()
                    index += 2
                    continue

            if self._accepted_exact_english(original_word):
                tokens.append(
                    self._english_token(
                        original=original_word,
                        corrected=original_word,
                        inner_text=original_word,
                    )
                )
                corrected_parts.append(original_word)
                previous_surface_word = original_word
                last_end = match.end()
                index += 1
                continue

            # A verified j-initial verb correction must not be swallowed by a
            # later phrase heuristic.  This is deliberately limited to an
            # actual spelling change, so ordinary verb context still flows
            # through the normal grammar and i-/ji- stages.
            if original_norm.startswith("j"):
                verified_verb = self.correct_word(original_word)
                verified_verb_norm = self._normalize_word(verified_verb)
                if (
                    verified_verb_norm != original_norm
                    and self._is_verb_tagged_word(verified_verb_norm)
                ):
                    surface_verb = self._apply_surface_case(
                        original_word,
                        verified_verb_norm,
                        sentence_initial=sentence_initial,
                    )
                    choices = (
                        self._bulk_ambiguity_choices(
                            original_word,
                            surface_verb,
                            limit=3,
                        )
                        if bulk_mode
                        else self.ambiguity_choices(
                            original_word,
                            surface_verb,
                            limit=3,
                            edit_distance_tolerance=effective_tolerance,
                        )
                    )
                    tokens.append(
                        {
                            "type": "word",
                            "original": original_word,
                            "corrected": surface_verb,
                            "ambiguous": len(choices) > 1,
                            "crucial": False,
                            "choices": choices,
                            "name_like": False,
                        }
                    )
                    corrected_parts.append(surface_verb)
                    previous_surface_word = verified_verb_norm
                    last_end = match.end()
                    index += 1
                    continue

            near_english_double = None
            exact_proper_name = self._exact_lowercase_proper_name(original_word)
            if (
                exact_proper_name
                and exact_proper_name != original_word
                and original_norm not in self.dictionary_set
                and not self._is_verb_tagged_word(original_norm)
            ):
                tokens.append(
                    {
                        "type": "word",
                        "original": original_word,
                        "corrected": exact_proper_name,
                        "meaning": self.meaning_for(exact_proper_name),
                        "ambiguous": False,
                        "choices": [],
                        "name_like": True,
                    }
                )
                corrected_parts.append(exact_proper_name)
                previous_surface_word = self._normalize_word(exact_proper_name)
                last_end = match.end()
                index += 1
                continue
            if (
                original_norm not in self.dictionary_set
                and hasattr(self, "doubled_letter_generator")
            ):
                near_english_double = (
                    self.doubled_letter_generator.correct_missing_double(original_norm)
                )

            if self._near_accepted_english_word(original_word) and not near_english_double:
                tokens.append(
                    {
                        "type": "word",
                        "original": original_word,
                        "corrected": original_word,
                        "meaning": "",
                        "ambiguous": False,
                        "choices": [],
                        "name_like": False,
                        # A close English typo (for example aotomatic) must
                        # never be silently accepted or forced into Maltese.
                        "force_unrecognized": True,
                    }
                )
                corrected_parts.append(original_word)
                previous_surface_word = original_word
                last_end = match.end()
                index += 1
                continue

            # A dictionary-verified missing double is an orthographic repair,
            # not a fuzzy alternative. Resolve it before phrase look-aheads can
            # consume the token through an unrelated path.
            direct_double = None
            article_prefixes = {
                "il", "l", "tal", "mal", "bil", "fil", "fis", "fir",
                "mil", "mid", "mill", "lil", "lill", "sal",
                "ghal", "għal", "ghall", "għall", "ghat", "għat",
            }
            if (
                original_norm not in self.dictionary_set
                and not self._article_like_token(original_word)
                and original_norm not in article_prefixes
                and original_norm not in {"ghar", "għar"}
                and hasattr(self, "doubled_letter_generator")
            ):
                direct_double = near_english_double
            compact_long_phrase = self._expand_compact_long_preposition_phrase(
                original_word
            )
            if compact_long_phrase is not None:
                corrected_phrase = self._apply_surface_case(
                    original_word,
                    compact_long_phrase,
                    sentence_initial=sentence_initial,
                )
                phrase_choices = [
                    {
                        "word": corrected_phrase,
                        "meaning": self.meaning_for(compact_long_phrase.split()[-1]),
                    }
                ]
                tokens.append(
                    {
                        "type": "word",
                        "original": original_word,
                        "corrected": corrected_phrase,
                        "ambiguous": False,
                        "choices": phrase_choices,
                        "name_like": False,
                    }
                )
                corrected_parts.append(corrected_phrase)
                previous_surface_word = self._normalize_word(
                    corrected_phrase.split()[-1]
                )
                last_end = match.end()
                index += 1
                continue

            reduced_il_xi_phrase = self._reduced_il_xi_phrase_match(
                word_tokens,
                index,
                previous_surface_word,
                sentence_initial,
            )
            if reduced_il_xi_phrase is not None:
                corrected_phrase, phrase_choices, consumed = reduced_il_xi_phrase
                tokens.append(
                    {
                        "type": "phrase",
                        "original": text[
                            matches[index].start() : matches[index + consumed - 1].end()
                        ],
                        "corrected": corrected_phrase,
                        "ambiguous": False,
                        "crucial": False,
                        "choices": phrase_choices,
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

            fixed_time_phrase = self._fixed_time_phrase_match(
                word_tokens,
                index,
                sentence_initial,
            )
            if fixed_time_phrase is not None:
                corrected_phrase, phrase_choices, consumed = fixed_time_phrase
                is_ambiguous, is_crucial = token_choice_state(
                    phrase_choices,
                    force_crucial=False,
                )
                tokens.append(
                    {
                        "type": "phrase",
                        "original": text[
                            matches[index].start() : matches[index + consumed - 1].end()
                        ],
                        "corrected": corrected_phrase,
                        "ambiguous": is_ambiguous,
                        "crucial": is_crucial,
                        "choices": phrase_choices,
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

            number_phrase = None
            if index + 1 < len(matches) and not getattr(matches[index + 1], "is_quote", False):
                number_phrase = self._number_phrase_payload(
                    original_word,
                    word_tokens[index + 1].text,
                    sentence_initial=sentence_initial,
                )
                if number_phrase is not None:
                    corrected_phrase, phrase_choices, is_crucial = number_phrase
                    is_ambiguous, _ = token_choice_state(
                        phrase_choices,
                        force_crucial=is_crucial,
                    )
                    tokens.append(
                        {
                            "type": "phrase",
                            "original": text[
                                matches[index].start() : matches[index + 1].end()
                            ],
                            "corrected": corrected_phrase,
                            "ambiguous": is_ambiguous,
                            "crucial": bool(is_crucial),
                            "choices": phrase_choices,
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

            if (
                self._article_like_token(original_word)
                and index + 2 < len(matches)
                and not getattr(matches[index + 1], "is_quote", False)
                and not getattr(matches[index + 2], "is_quote", False)
            ):
                number_phrase = self._number_phrase_payload(
                    word_tokens[index + 1].text,
                    word_tokens[index + 2].text,
                    sentence_initial=sentence_initial,
                    article_word=original_word,
                )
                if number_phrase is not None:
                    corrected_phrase, phrase_choices, is_crucial = number_phrase
                    is_ambiguous, _ = token_choice_state(
                        phrase_choices,
                        force_crucial=is_crucial,
                    )
                    tokens.append(
                        {
                            "type": "phrase",
                            "original": text[
                                matches[index].start() : matches[index + 2].end()
                            ],
                            "corrected": corrected_phrase,
                            "ambiguous": is_ambiguous,
                            "crucial": bool(is_crucial),
                            "choices": phrase_choices,
                            "name_like": False,
                        }
                    )
                    corrected_parts.append(corrected_phrase)
                    previous_surface_word = self._normalize_word(
                        corrected_phrase.split()[-1]
                    )
                    last_end = matches[index + 2].end()
                    index += 3
                    continue

            if index + 1 < len(matches) and not getattr(
                matches[index + 1], "is_quote", False
            ):
                next_word_for_phrase = word_tokens[index + 1].text
                next_norm_for_phrase = self._normalize_word(next_word_for_phrase)
                separator_between = text[match.end() : matches[index + 1].start()]
                if original_norm in {"il", "l"} and separator_between.isspace():
                    corrected_next = self.correct_word(next_word_for_phrase)
                    if self._normalize_word(corrected_next) == "ħaddieħor":
                        primary_phrase = self._apply_surface_case(
                            original_word,
                            "lil ħaddieħor",
                            sentence_initial=sentence_initial,
                        )
                        literal_article = self._apply_surface_case(
                            original_word,
                            "'il ħaddieħor",
                            sentence_initial=sentence_initial,
                        )
                        phrase_choices = [
                            {
                                "word": primary_phrase,
                                "meaning": self.meaning_for("ħaddieħor"),
                            },
                            {
                                "word": literal_article,
                                "meaning": self.meaning_for("ħaddieħor"),
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
                                "corrected": primary_phrase,
                                "ambiguous": is_ambiguous,
                                "crucial": is_crucial,
                                "choices": phrase_choices,
                                "name_like": False,
                            }
                        )
                        corrected_parts.append(primary_phrase)
                        previous_surface_word = "ħaddieħor"
                        last_end = matches[index + 1].end()
                        index += 2
                        continue
                if original_norm == "li" and separator_between.isspace():
                    corrected_next = self.correct_word(next_word_for_phrase)
                    corrected_next_norm = self._normalize_word(corrected_next)
                    if (
                        self._is_adjective_tagged_word(corrected_next_norm)
                        and not self._is_li_relative_adjective_allowed(
                            corrected_next_norm
                        )
                    ):
                        corrected_phrase = self._apply_surface_case(
                            original_word,
                            f"l-{corrected_next_norm}",
                            sentence_initial=sentence_initial,
                        )
                        tokens.append(
                            {
                                "type": "phrase",
                                "original": text[
                                    matches[index].start() : matches[index + 1].end()
                                ],
                                "corrected": corrected_phrase,
                                "meaning": self.meaning_for(
                                    corrected_next_norm
                                ),
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
                if (
                    original_norm in {"ghar", "għar"}
                    and (
                        next_norm_for_phrase.startswith("mill-")
                        or next_norm_for_phrase.startswith("mill")
                        or next_norm_for_phrase in {"mil", "minn"}
                    )
                ):
                    corrected_word = self._apply_surface_case(
                        original_word,
                        "agħar",
                        sentence_initial=sentence_initial,
                    )
                    tokens.append(
                        {
                            "type": "word",
                            "original": original_word,
                            "corrected": corrected_word,
                            "meaning": self.meaning_for("agħar"),
                            "ambiguous": True,
                            "choices": [
                                {
                                    "word": corrected_word,
                                    "meaning": self.meaning_for("agħar"),
                                },
                                {
                                    "word": self._apply_surface_case(
                                        original_word,
                                        "għar",
                                        sentence_initial=sentence_initial,
                                    ),
                                    "meaning": self.meaning_for("għar"),
                                },
                            ],
                            "name_like": False,
                        }
                    )
                    corrected_parts.append(corrected_word)
                    previous_surface_word = self._normalize_word(corrected_word)
                    last_end = match.end()
                    index += 1
                    continue
                if (
                    original_norm in {"ghal", "għal", "ghall", "għall"}
                    and next_norm_for_phrase in {"ghar", "għar"}
                    and previous_surface_word == "dejjem"
                ):
                    corrected_phrase = self._apply_surface_case(
                        original_word,
                        "għall-agħar",
                        sentence_initial=sentence_initial,
                    )
                    tokens.append(
                        {
                            "type": "phrase",
                            "original": text[
                                matches[index].start() : matches[index + 1].end()
                            ],
                            "corrected": corrected_phrase,
                            "ambiguous": True,
                            "choices": [
                                {
                                    "word": corrected_phrase,
                                    "meaning": self.meaning_for("agħar"),
                                },
                                {
                                    "word": self._apply_surface_case(
                                        original_word,
                                        "għall-għar",
                                        sentence_initial=sentence_initial,
                                    ),
                                    "meaning": self.meaning_for("għar"),
                                },
                            ],
                            "name_like": False,
                        }
                    )
                    corrected_parts.append(corrected_phrase)
                    previous_surface_word = self._normalize_word(corrected_phrase)
                    last_end = matches[index + 1].end()
                    index += 2
                    continue
                if not separator_between:
                    combined_word = original_word + next_word_for_phrase
                    combined_corrected = self.correct_word(combined_word)
                    combined_norm = self._normalize_word(combined_corrected)
                    if (
                        combined_norm in self.dictionary_set
                        or combined_norm != self._normalize_word(combined_word)
                    ):
                        combined_corrected = self._apply_surface_case(
                            combined_word,
                            combined_corrected,
                            sentence_initial=sentence_initial,
                        )
                        tokens.append(
                            {
                                "type": "word",
                                "original": combined_word,
                                "corrected": combined_corrected,
                                "meaning": self.meaning_for(combined_corrected),
                                "ambiguous": False,
                                "choices": [],
                                "name_like": False,
                            }
                        )
                        corrected_parts.append(combined_corrected)
                        previous_surface_word = self._normalize_word(combined_corrected)
                        last_end = matches[index + 1].end()
                        index += 2
                        continue
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

                # Outside a terminal punctuation context, the standalone
                # form ``ta`` is the preposition and contracts to ``ta'``.
                # The explicit ``ta il ...`` article route is handled above.
                if (
                    original_norm == "ta"
                    and separator_between.isspace()
                    and next_norm_for_phrase not in {"il", "l"}
                ):
                    corrected_next = self.correct_word(next_word_for_phrase)
                    corrected_next = self._apply_surface_case(
                        next_word_for_phrase,
                        corrected_next,
                        sentence_initial=False,
                    )
                    ta_surface = self._apply_surface_case(
                        original_word,
                        "ta",
                        sentence_initial=sentence_initial,
                    )
                    if self._starts_vowel_gh_or_h(corrected_next) and self._normalize_word(
                        corrected_next
                    )[0] in self.VOWELS:
                        corrected_phrase = f"{ta_surface[:-1]}'{corrected_next}"
                    else:
                        corrected_phrase = f"{ta_surface}' {corrected_next}"
                    alternate_phrase = f"{ta_surface} {corrected_next}"
                    phrase_choices = [
                        {
                            "word": corrected_phrase,
                            "meaning": self.meaning_for(corrected_next),
                        },
                        {
                            "word": alternate_phrase,
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
                            "name_like": False,
                        }
                    )
                    corrected_parts.append(corrected_phrase)
                    previous_surface_word = self._normalize_word(corrected_next)
                    last_end = matches[index + 1].end()
                    index += 2
                    continue

                spaced_prepositions = {
                    "fl", "bl", "sal", "tal", "fil", "bil", "mil", "mill",
                    "mid", "mis", "miss", "lil", "lill",
                    "ghall", "għall", "ghal", "għal",
                    "fir", "ghat", "għat", "ghadd", "għadd", "ghacc", "għaċċ",
                    "ghatt", "għatt", "mic", "miċ",
                    # Article assimilation forms: ir-, in-, is-, it-, id-, iċ-, iż-
                    "il", "ir", "in", "is", "it", "id", "iċ", "iż",
                }
                article_rules = getattr(self, "article_phrase_rules", None)
                fixed_time_phrase = self._fixed_time_phrase_match(
                    word_tokens,
                    index,
                    sentence_initial,
                )
                if fixed_time_phrase is not None:
                    corrected_phrase, phrase_choices, consumed = fixed_time_phrase
                    is_ambiguous, is_crucial = token_choice_state(
                        phrase_choices,
                        force_crucial=False,
                    )
                    tokens.append(
                        {
                            "type": "phrase",
                            "original": text[
                                matches[index].start() : matches[index + consumed - 1].end()
                            ],
                            "corrected": corrected_phrase,
                            "ambiguous": is_ambiguous,
                            "crucial": is_crucial,
                            "choices": phrase_choices,
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
                if original_norm in {"ghall", "għall"} and next_norm_for_phrase == "xi":
                    corrected_word = self._apply_surface_case(
                        original_word,
                        "għal",
                        sentence_initial=sentence_initial,
                    )
                    tokens.append(
                        {
                            "type": "word",
                            "original": original_word,
                            "corrected": corrected_word,
                            "meaning": self.meaning_for("għal"),
                            "ambiguous": False,
                            "choices": [],
                            "name_like": False,
                        }
                    )
                    corrected_parts.append(corrected_word)
                    previous_surface_word = self._normalize_word(corrected_word)
                    last_end = match.end()
                    index += 1
                    continue
                if original_norm in {"ghan", "għan"} and separator_between.isspace():
                    corrected_next = self.correct_word(next_word_for_phrase)
                    if (
                        self._is_noun_tagged_word(corrected_next)
                        or self._is_adjective_tagged_word(corrected_next)
                        or self._capitalized_name_kind(corrected_next)
                        or self._is_probable_noun(corrected_next)
                    ):
                        corrected_phrase = self._apply_surface_case(
                            original_word,
                            f"għan-{self._normalize_word(corrected_next)}",
                            sentence_initial=sentence_initial,
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
                        previous_surface_word = self._normalize_word(
                            corrected_phrase.split()[-1]
                        )
                        last_end = matches[index + 1].end()
                        index += 2
                        continue
                if (
                    original_norm in {"il", "l"}
                    and separator_between.isspace()
                    and self._is_initial_capitalized(next_word_for_phrase)
                ):
                    corrected_place = self._correct_place_word(next_word_for_phrase)
                    if corrected_place is not None:
                        prefix_surface = (
                            self._capitalize_first_letter(original_norm)
                            if sentence_initial
                            else original_norm
                        )
                        corrected_phrase = self._match_capitalisation(
                            original_word,
                            f"{prefix_surface}-{self._capitalize_first_letter(self._normalize_word(corrected_place))}",
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
                                "name_like": True,
                            }
                        )
                        corrected_parts.append(corrected_phrase)
                        previous_surface_word = self._normalize_word(
                            corrected_phrase.split()[-1]
                        )
                        last_end = matches[index + 1].end()
                        index += 2
                        continue
                phrase_choices = []
                if (
                    (
                        original_norm in spaced_prepositions
                        or (
                            article_rules is not None
                            and article_rules._assimilated_prefix_canonical(original_norm)
                        )
                    )
                    and original_norm not in {"il", "l"}
                    and article_rules
                    and separator_between.isspace()
                    # A following article token belongs to the three-token
                    # contraction matcher below (għal l ktieb -> għall-ktieb),
                    # not to this two-token preposition path.
                    and not (
                        next_norm_for_phrase
                        in {
                            "il", "l", "ic", "iċ", "id", "in", "ir",
                            "is", "it", "ix", "iz", "iż",
                        }
                        and index + 2 < len(matches)
                    )
                ):
                    tail_is_place_word = False
                    tail_is_capitalized = bool(next_word_for_phrase[:1].isupper())
                    capitalized_name_tail = self._capitalize_first_letter(
                        next_word_for_phrase
                    )
                    lookup_word_for_phrase = next_word_for_phrase
                    lookup_norm_for_phrase = next_norm_for_phrase
                    if (
                        article_rules._assimilated_prefix_canonical(original_norm)
                        and "-" in lookup_norm_for_phrase
                    ):
                        typed_tail_prefix, possible_tail = lookup_norm_for_phrase.split("-", 1)
                        if (
                            possible_tail
                            and typed_tail_prefix
                            and typed_tail_prefix[-1:] == original_norm[-1:]
                        ):
                            lookup_word_for_phrase = possible_tail
                            lookup_norm_for_phrase = possible_tail
                    if original_norm in {"il", "l"} and tail_is_capitalized:
                        corrected_tail = capitalized_name_tail
                        tail_is_place_word = True
                    elif self._capitalized_name_kind(capitalized_name_tail):
                        corrected_tail = capitalized_name_tail
                    else:
                        corrected_place = (
                            self._correct_place_word(lookup_word_for_phrase)
                            if tail_is_capitalized
                            else None
                        )
                        if self._accepted_exact_english(lookup_word_for_phrase):
                            corrected_tail = lookup_norm_for_phrase
                        elif corrected_place is not None:
                            corrected_tail = corrected_place
                            tail_is_place_word = True
                        else:
                            corrected_tail = article_rules._strict_dictionary_tail(
                                lookup_norm_for_phrase
                            )
                            if corrected_tail is None:
                                corrected_next = self.correct_word(lookup_word_for_phrase)
                                if tail_is_capitalized:
                                    corrected_next = self._capitalize_first_letter(
                                        self._normalize_word(corrected_next)
                                    )
                                corrected_tail = article_rules._strict_dictionary_tail(
                                    corrected_next
                                )
                            if corrected_tail is None and tail_is_capitalized:
                                corrected_place = self._correct_place_word(
                                    lookup_word_for_phrase
                                )
                                corrected_tail = (
                                    corrected_place
                                    if corrected_place
                                    else None
                                )
                                tail_is_place_word = corrected_place is not None
                            if (
                                corrected_tail is None
                                and original_norm in {"lil", "lill"}
                                and self._noun_possessive_base_for_surface(
                                    lookup_norm_for_phrase
                                )
                            ):
                                corrected_tail = lookup_norm_for_phrase
                            if (
                                corrected_tail is None
                                and tail_is_capitalized
                                and original_norm in {"il", "l"}
                            ):
                                corrected_tail = self._capitalize_first_letter(
                                    next_word_for_phrase
                                )
                                tail_is_place_word = True
                    if tail_is_capitalized and corrected_tail:
                        corrected_tail = self._capitalize_first_letter(
                            self._normalize_word(corrected_tail)
                        )
                    elif corrected_tail:
                        corrected_tail = self._normalize_word(corrected_tail)
                    if original_norm in {"lil", "lill"} and self._capitalized_name_kind(
                        capitalized_name_tail
                    ):
                        corrected_phrase = f"{original_norm} {corrected_tail}"
                    else:
                        if original_norm in {"lil", "lill"}:
                            corrected_phrase = (
                                article_rules.preposition_article_form(
                                    original_norm,
                                    corrected_tail,
                                )
                                if corrected_tail
                                else None
                            )
                        elif tail_is_place_word and original_norm in {"il", "l"}:
                            prefix_surface = (
                                self._capitalize_first_letter(original_norm)
                                if sentence_initial
                                else original_norm
                            )
                            corrected_phrase = f"{prefix_surface}-{corrected_tail}"
                        elif original_norm == "mil" and corrected_tail == "bidu":
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
                        if corrected_phrase and corrected_tail:
                            phrase_choices = article_rules.preposition_article_choices(
                                original_norm,
                                self._normalize_word(corrected_tail),
                                previous_surface_word,
                            )
                else:
                    corrected_phrase = None
                if corrected_phrase:
                    if tail_is_capitalized and "-" in corrected_phrase:
                        corrected_phrase = self._match_hyphenated_tail_capitalisation(
                            next_word_for_phrase,
                            corrected_phrase,
                        )
                    if not tail_is_place_word:
                        corrected_phrase = self._match_capitalisation(
                            text[matches[index].start() : matches[index + 1].end()],
                            corrected_phrase,
                        )
                    corrected_phrase = self._match_hyphenated_tail_capitalisation(
                        next_word_for_phrase,
                        corrected_phrase,
                    )
                    if sentence_initial:
                        corrected_phrase = self._capitalize_first_letter(
                            corrected_phrase
                        )
                        phrase_choices = [
                            {
                                **choice,
                                "word": self._capitalize_first_letter(
                                    choice.get("word", "")
                                ),
                            }
                            for choice in phrase_choices
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
                            "name_like": False,
                        }
                    )
                    corrected_parts.append(corrected_phrase)
                    previous_surface_word = self._normalize_word(corrected_phrase)
                    last_end = matches[index + 1].end()
                    index += 2
                    continue

            fixed_compound_word = self._fixed_time_expression_word(original_word)
            if (
                fixed_compound_word
                and self._normalize_word(fixed_compound_word) != original_norm
            ):
                corrected_word = self._apply_surface_case(
                    original_word,
                    fixed_compound_word,
                    sentence_initial=sentence_initial,
                )
                tokens.append(
                    {
                        "type": "word",
                        "original": original_word,
                        "corrected": corrected_word,
                        "ambiguous": False,
                        "choices": [],
                        "name_like": False,
                    }
                )
                corrected_parts.append(corrected_word)
                previous_surface_word = self._normalize_word(corrected_word)
                last_end = match.end()
                index += 1
                continue

            exact_name_kind = self._capitalized_name_kind(original_word)
            if exact_name_kind is not None:
                tokens.append(
                    {
                        "type": "word",
                        "original": original_word,
                        "corrected": original_word,
                        "ambiguous": False,
                        "choices": [],
                        "name_like": True,
                    }
                )
                corrected_parts.append(original_word)
                previous_surface_word = self._normalize_word(original_word)
                last_end = match.end()
                index += 1
                continue

            early_restorative = []
            if not self._is_recognized_surface(original_norm):
                early_restorative = self._medial_guttural_vowel_restoration_variants(
                    original_word
                )
            if early_restorative:
                corrected_word = self._apply_surface_case(
                    original_word,
                    early_restorative[0],
                    sentence_initial=sentence_initial,
                )
                tokens.append(
                    {
                        "type": "word",
                        "original": original_word,
                        "corrected": corrected_word,
                        "meaning": self.meaning_for(early_restorative[0]),
                        "ambiguous": False,
                        "choices": [],
                        "name_like": False,
                    }
                )
                corrected_parts.append(corrected_word)
                previous_surface_word = self._normalize_word(corrected_word)
                last_end = match.end()
                index += 1
                continue

            token_repairs = []
            social_comment_repair = None
            if original_norm not in self.dictionary_set:
                doubled_letter = None
                if hasattr(self, "doubled_letter_generator"):
                    doubled_letter = (
                        self.doubled_letter_generator.correct_missing_double(
                            original_norm
                        )
                    )
                token_repairs = [doubled_letter] if doubled_letter else []
            if token_repairs:
                corrected_word = (
                    social_comment_repair
                    if social_comment_repair
                    else self.correct_word(original_word)
                )
                choices = (
                    self.ambiguity_choices(
                        original_word,
                        corrected_word,
                        limit=3,
                        edit_distance_tolerance=effective_tolerance,
                    )
                    if not bulk_mode
                    else []
                )
                if sentence_initial:
                    corrected_word = self._apply_surface_case(
                        original_word,
                        corrected_word,
                        sentence_initial=True,
                    )
                    choices = [
                        {
                            **choice,
                            "word": self._apply_surface_case(
                                original_word,
                                choice.get("word", ""),
                                sentence_initial=True,
                            ),
                        }
                        for choice in choices
                    ]
                is_name_like = (
                    self._is_initial_capitalized(original_word) and not sentence_initial
                )
                is_ambiguous = len(choices) >= 2 and self._normalize_word(
                    choices[0]["word"]
                ) != self._normalize_word(choices[1]["word"])
                tokens.append(
                    {
                        "type": "word",
                        "original": original_word,
                        "corrected": corrected_word,
                        "meaning": self.meaning_for(corrected_word),
                        "ambiguous": is_ambiguous,
                        "choices": choices if is_ambiguous else [],
                        "name_like": is_name_like,
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

                    if current_norm in {"bl", "fl"}:
                        article_prefix = (
                            self._match_capitalisation(original_word, current_norm)
                            + "-"
                        )
                        is_crucial_art = True
                        article_choices = [{"word": article_prefix, "meaning": ""}]

                    if article_prefix is None and current_norm in ALL_FORMS:
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

            current_norm = self._normalize_word(original_word)
            capitalized_place_phrase = self._match_capitalized_place_phrase(
                text,
                word_tokens,
                matches,
                index,
            )
            if (
                capitalized_place_phrase is None
                and current_norm == "l"
                and index + 1 < len(matches)
                and self._is_initial_capitalized(word_tokens[index + 1].text)
            ):
                next_word = word_tokens[index + 1].text
                corrected_place = self._correct_place_word(next_word)
                if corrected_place is not None:
                    corrected_phrase = (
                        f"{self._capitalize_first_letter(original_word) if sentence_initial else original_word}-"
                        f"{self._capitalize_first_letter(self._normalize_word(corrected_place))}"
                    )
                    tokens.append(
                        {
                            "type": "phrase",
                            "original": text[
                                matches[index].start() : matches[index + 1].end()
                            ],
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
                    last_end = matches[index + 1].end()
                    index += 2
                    continue
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

                # ``dan/din li`` are demonstrative + relative-pronoun
                # sequences, never fused article forms. Likewise, ``li``
                # must stay intact before a verb and ``għal xi`` is not the
                # x-apostrophe construction.
                protected_phrase = None
                if current_norm in {"dan", "din"} and next_norm == "li":
                    protected_phrase = f"{current_norm} li"
                elif current_norm in {"għal", "ghal"} and next_norm == "xi":
                    protected_phrase = "għal xi"
                elif current_norm == "li":
                    protected_next = self.correct_word(next_word)
                    if self._is_verb_tagged_word(protected_next):
                        protected_phrase = f"li {protected_next}"

                if protected_phrase is not None:
                    corrected_phrase = self._apply_surface_case(
                        original_word,
                        protected_phrase,
                        sentence_initial=sentence_initial,
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
                        }
                    )
                    corrected_parts.append(corrected_phrase)
                    previous_surface_word = self._normalize_word(
                        corrected_phrase.split()[-1]
                    )
                    last_end = matches[index + 1].end()
                    index += 2
                    continue

                fixed_time_phrase = self._fixed_time_phrase_match(
                    word_tokens,
                    index,
                    sentence_initial,
                )
                if fixed_time_phrase is not None:
                    corrected_phrase, phrase_choices, consumed = fixed_time_phrase
                    is_ambiguous, is_crucial = token_choice_state(
                        phrase_choices,
                        force_crucial=False,
                    )
                    tokens.append(
                        {
                            "type": "phrase",
                            "original": text[
                                matches[index].start() : matches[index + consumed - 1].end()
                            ],
                            "corrected": corrected_phrase,
                            "ambiguous": is_ambiguous,
                            "crucial": is_crucial,
                            "choices": phrase_choices,
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
                        literal_noun = (
                            f"i{article_noun_norm}"
                            if article_rules is not None
                            and article_rules._requires_article_epenthetic_i(
                                article_noun_norm
                            )
                            else article_noun_norm
                        )
                        if sentence_initial:
                            corrected_phrase = self._capitalize_first_letter(
                                corrected_phrase
                            )
                        phrase_choices = [
                            {
                                "word": corrected_phrase,
                                "meaning": self.meaning_for(article_noun_norm),
                            },
                            {
                                "word": f"ta 'l {literal_noun}",
                                "meaning": self.meaning_for(article_noun_norm),
                            },
                            {
                                "word": f"ta' l-{literal_noun}",
                                "meaning": self.meaning_for(article_noun_norm),
                            },
                        ]
                        tokens.append(
                            {
                                "type": "phrase",
                                "original": text[
                                    matches[index].start() : matches[index + 2].end()
                                ],
                                "corrected": corrected_phrase,
                                "ambiguous": True,
                                "crucial": True,
                                "choices": phrase_choices,
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
                        display_prefix = self._apply_surface_case(
                            original_word,
                            current_norm,
                            sentence_initial=sentence_initial,
                        )
                        corrected_phrase = f"{display_prefix}'{corrected_next}"
                        negated_meaning = self.meaning_for(corrected_next)
                        if negated_meaning and not negated_meaning.startswith("not "):
                            negated_meaning = f"not {negated_meaning}"
                        phrase_choices = [
                            {
                                "word": corrected_phrase,
                                "meaning": negated_meaning,
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

                if current_norm in {"ma", "ma'"} and next_norm == "tul":
                    corrected_phrase = self._capitalize_first_letter("matul") if sentence_initial else "matul"
                    phrase_choices = [
                        {"word": self._capitalize_first_letter("matul") if sentence_initial else "matul", "meaning": "throughout"},
                        {"word": self._capitalize_first_letter("ma' tul") if sentence_initial else "ma' tul", "meaning": "with length"},
                    ]
                    tokens.append(
                        {
                            "type": "phrase",
                            "original": text[matches[index].start() : matches[index + 1].end()],
                            "corrected": corrected_phrase,
                            "ambiguous": True,
                            "crucial": True,
                            "choices": phrase_choices,
                        }
                    )
                    corrected_parts.append(corrected_phrase)
                    previous_surface_word = "matul"
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
                    corrected_next = (
                        self._conservative_capitalized_word(next_word)
                        if self._is_initial_capitalized(next_word)
                        else self.correct_word(next_word)
                    )
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
                    corrected_next_norm = self._normalize_word(corrected_next)
                    bare_ma_negatives = {"tantx"}

                    target_ma = (
                        "ma"
                        if is_verb or corrected_next_norm in bare_ma_negatives
                        else "ma'"
                    )

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

                        if current_norm != target_ma and is_verb:
                            other_ma = "ma'"
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

                # ``ma'`` is already the only valid form before a non-verb.
                # Keep the pair together so the generic apostrophe toggle
                # cannot subsequently offer bare ``ma`` in noun contexts.
                if current_norm == "ma'":
                    corrected_next = (
                        self._conservative_capitalized_word(next_word)
                        if self._is_initial_capitalized(next_word)
                        else self.correct_word(next_word)
                    )
                    if not self._is_verb_tagged_word(corrected_next):
                        display_ma = self._match_capitalisation(original_word, "ma'")
                        corrected_phrase = f"{display_ma} {corrected_next}"
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
                            }
                        )
                        corrected_parts.append(corrected_phrase)
                        previous_surface_word = self._normalize_word(corrected_next)
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
                    SPECIAL_MINN_WORDS = {"xiex", "meta", "hekk"}
                    HEMM_HAWN_WORDS = {"hemm", "hawn", "hemmhekk", "hawnhekk"}

                    is_name = self._is_initial_capitalized(corrected_next)
                    next_norm_corrected = self._normalize_word(corrected_next)

                    if next_norm == "aw":
                        # ``aw`` has a deterministic correction to ``hawn``;
                        # preserve the user's ``min`` and expose ``minn`` as
                        # the close alternative instead of changing both.
                        target_min = current_norm
                    else:
                        target_min = current_norm

                    if current_norm != target_min or next_norm_corrected != next_norm:
                        display_min = self._apply_surface_case(
                            original_word,
                            target_min,
                            sentence_initial=sentence_initial,
                        )
                        corrected_phrase = f"{display_min} {corrected_next}"
                        phrase_choices = [
                            {
                                "word": display_min,
                                "meaning": self.meaning_for(display_min),
                            },
                        ]

                        if current_norm != target_min or next_norm == "aw" or next_norm_corrected != next_norm:
                            other_min = "min" if target_min == "minn" else "minn"
                            display_other = self._apply_surface_case(
                                original_word,
                                other_min,
                                sentence_initial=sentence_initial,
                            )
                            phrase_choices.append(
                                {
                                    "word": display_other,
                                    "meaning": self.meaning_for(display_other),
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

                if current_norm == "kull" and next_norm == "jum":
                    corrected_phrase = self._capitalize_first_letter("kuljum") if sentence_initial else "kuljum"
                    phrase_choices = [
                        {"word": self._capitalize_first_letter("kuljum") if sentence_initial else "kuljum", "meaning": "every day / daily"},
                        {"word": self._capitalize_first_letter("kull jum") if sentence_initial else "kull jum", "meaning": "every day"},
                    ]
                    tokens.append(
                        {
                            "type": "phrase",
                            "original": text[matches[index].start() : matches[index + 1].end()],
                            "corrected": corrected_phrase,
                            "ambiguous": True,
                            "crucial": True,
                            "choices": phrase_choices,
                        }
                    )
                    corrected_parts.append(corrected_phrase)
                    previous_surface_word = "kuljum"
                    last_end = matches[index + 1].end()
                    index += 2
                    continue

                if current_norm in {"ghar", "għar"}:
                    corrected_next = self.correct_word(next_word)
                    corrected_next_norm = self._normalize_word(corrected_next)
                    if next_norm.startswith("mill") or next_norm in {"mil", "minn"}:
                        corrected_phrase = self._apply_surface_case(
                            original_word,
                            "agħar",
                            sentence_initial=sentence_initial,
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
                        last_end = match.end()
                        index += 1
                        continue
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
            ) and not (
                index + 1 < len(matches)
                and self._normalize_word(word_tokens[index].text) in {"għal", "ghal"}
                and self._normalize_word(word_tokens[index + 1].text) == "xi"
            ):
                if index + 1 < len(matches):
                    between_tokens = text[matches[index].end() : matches[index + 1].start()]
                    if "-" in between_tokens and not between_tokens.replace("-", "").strip():
                        previous_word = None if sentence_initial else (
                            word_tokens[index - 1].text if index > 0 else None
                        )
                        article_match = article_rules.match_hyphenated_article_after(
                            f"{original_word}-{word_tokens[index + 1].text}",
                            previous=previous_word,
                        )
                        if article_match is not None:
                            corrected_phrase = (
                                self._capitalize_first_letter(article_match.corrected)
                                if sentence_initial
                                else article_match.corrected
                            )
                            is_ambiguous, is_crucial = token_choice_state(
                                article_match.choices,
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
                                    "choices": article_match.choices,
                                }
                            )
                            corrected_parts.append(corrected_phrase)
                            previous_surface_word = self._normalize_word(
                                corrected_phrase.split()[-1]
                            )
                            last_end = matches[index + 1].end()
                            index += 2
                            continue

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
                    if sentence_initial:
                        article_corrected = self._capitalize_first_letter(
                            article_corrected
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
                    corrected_phrase = (
                        self._capitalize_first_letter(article_match.corrected)
                        if sentence_initial
                        else article_match.corrected
                    )
                    is_ambiguous, is_crucial = token_choice_state(
                        article_match.choices,
                        force_crucial=True,
                    )
                    phrase_choices = article_match.choices
                    if sentence_initial:
                        phrase_choices = [
                            {
                                **choice,
                                "word": self._capitalize_first_letter(
                                    choice.get("word", "")
                                ),
                            }
                            for choice in phrase_choices
                        ]

                    tokens.append(
                        {
                            "type": "phrase",
                            "original": original_phrase,
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

                if index + 1 < len(matches):
                    fallback_previous = None if sentence_initial else (
                        word_tokens[index - 1].text if index > 0 else None
                    )
                    fallback_article = self._split_article_unknown_tail(
                        original_word,
                        word_tokens[index + 1].text,
                        previous=fallback_previous,
                    )
                    if fallback_article is not None:
                        corrected_phrase = (
                            self._capitalize_first_letter(fallback_article)
                            if sentence_initial
                            else fallback_article
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
                        previous_surface_word = self._normalize_word(
                            corrected_phrase.split()[-1]
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

                if current_norm not in {
                    "il", "l", "ic", "iċ", "id", "in", "ir", "is", "it", "ix", "iz", "iż",
                }:
                    fallback_previous = None if sentence_initial else (
                        word_tokens[index - 1].text if index > 0 else None
                    )
                    fallback_article = self._split_article_unknown_tail(
                        original_word,
                        next_word,
                        previous=fallback_previous,
                    )
                    if fallback_article is not None:
                        corrected_phrase = (
                            self._capitalize_first_letter(fallback_article)
                            if sentence_initial
                            else fallback_article
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
                        previous_surface_word = self._normalize_word(
                            corrected_phrase.split()[-1]
                        )
                        last_end = matches[index + 1].end()
                        index += 2
                        continue

                if current_norm == "l" and next_word[:1].isupper():
                    corrected_next = self._correct_place_word(next_word)
                    if corrected_next is None:
                        corrected_next = self.correct_word(next_word)
                    corrected_next = self._capitalize_first_letter(
                        self._normalize_word(corrected_next)
                    )
                    prefix_surface = (
                        self._capitalize_first_letter(current_norm)
                        if sentence_initial
                        else current_norm
                    )
                    corrected_phrase = f"{prefix_surface}-{corrected_next}"
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

                if current_norm == "l":
                    next_is_capitalized = bool(next_word[:1].isupper())
                    corrected_next = (
                        self._correct_place_word(next_word)
                        if next_is_capitalized
                        else None
                    )
                    if corrected_next is None:
                        corrected_next = self.correct_word(next_word)
                        if next_is_capitalized:
                            corrected_next = self._capitalize_first_letter(
                                self._normalize_word(corrected_next)
                            )
                    else:
                        corrected_next = self._capitalize_first_letter(
                            self._normalize_word(corrected_next)
                        )
                    if next_is_capitalized:
                        corrected_next = self._match_capitalisation(
                            next_word,
                            corrected_next,
                        )
                    corrected_next_norm = self._normalize_word(corrected_next)
                    if (
                        self._starts_vowel_gh_or_h(next_norm)
                        or self._is_noun_tagged_word(corrected_next_norm)
                        or self._is_adjective_tagged_word(corrected_next_norm)
                        or self._is_probable_noun(corrected_next_norm)
                        or self._capitalized_name_kind(corrected_next)
                        or next_is_capitalized
                    ):
                        prefix_surface = (
                            self._capitalize_first_letter(current_norm)
                            if sentence_initial
                            else current_norm
                        )
                        corrected_phrase = f"{prefix_surface}-{corrected_next}"
                        if next_is_capitalized:
                            corrected_phrase = self._match_hyphenated_tail_capitalisation(
                                next_word,
                                corrected_phrase,
                            )
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

                fixed_time_phrase = self._fixed_time_phrase_match(
                    word_tokens,
                    index,
                    sentence_initial,
                )
                if fixed_time_phrase is not None:
                    corrected_phrase, phrase_choices, consumed = fixed_time_phrase
                    is_ambiguous, is_crucial = token_choice_state(
                        phrase_choices,
                        force_crucial=False,
                    )
                    tokens.append(
                        {
                            "type": "phrase",
                            "original": text[
                                matches[index].start() : matches[index + consumed - 1].end()
                            ],
                            "corrected": corrected_phrase,
                            "ambiguous": is_ambiguous,
                            "crucial": is_crucial,
                            "choices": phrase_choices,
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
                    current_norm in {"f", "b"}
                    and next_norm not in {"l", "il"}
                    and not next_norm.startswith(("l-", "il-"))
                ):
                    corrected_next = self.correct_word(next_word)
                    if self._is_probable_noun(corrected_next):
                        display_prefix = self._apply_surface_case(
                            original_word,
                            current_norm,
                            sentence_initial=sentence_initial,
                        )
                        corrected_phrase = f"{display_prefix}'{corrected_next}"
                        phrase_choices = [
                            {
                                "word": corrected_phrase,
                                "meaning": self.meaning_for(corrected_next),
                            }
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

                if current_norm == "dal":
                    corrected_next = self.correct_word(next_word)
                    corrected_phrase = self._match_capitalisation(
                        text[matches[index].start() : matches[index + 1].end()],
                        f"dal-{self._normalize_word(corrected_next)}",
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
                            "choices": [
                                {
                                    "word": corrected_phrase,
                                    "meaning": self.meaning_for(
                                        self._normalize_word(corrected_next)
                                    ),
                                }
                            ],
                        }
                    )
                    corrected_parts.append(corrected_phrase)
                    previous_surface_word = self._normalize_word(corrected_phrase)
                    last_end = matches[index + 1].end()
                    index += 2
                    continue

                # Special case: fi xħin → fi x'ħin, while always exposing xħin too.
                if current_norm == "fi" and next_norm in {"xhin", "xħin", "x'ħin"}:
                    corrected_phrase = self._apply_surface_case(
                        original_word,
                        "fi x'ħin",
                        sentence_initial=sentence_initial,
                    )
                    alternate_phrase = self._apply_surface_case(
                        original_word,
                        "fi xħin",
                        sentence_initial=sentence_initial,
                    )
                    phrase_choices = [
                        {"word": corrected_phrase, "meaning": self.meaning_for("x'ħin")},
                        {"word": alternate_phrase, "meaning": self.meaning_for("xħin")},
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
                    previous_surface_word = self._normalize_word("x'ħin")
                    last_end = matches[index + 1].end()
                    index += 2
                    continue

                if current_norm in {"xi", "bi", "fi"}:
                    corrected_next = next_word
                    if current_norm == "xi":
                        corrected_next = self.correct_word(next_word)
                    elif fused_preposition_rules is not None:
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
                        corrected_prefix = self._apply_surface_case(
                            original_word,
                            current_norm,
                            sentence_initial=sentence_initial,
                        )
                        corrected_phrase = f"{corrected_prefix} {corrected_next}"
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

                if current_norm in {"il", "l"} and self._accepted_exact_english(
                    next_word
                ):
                    corrected_phrase = self._apply_surface_case(
                        original_word,
                        f"{current_norm}-{self._normalize_word(next_word)}",
                        sentence_initial=sentence_initial,
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
                    previous_surface_word = self._normalize_word(
                        corrected_phrase.split()[-1]
                    )
                    last_end = matches[index + 1].end()
                    index += 2
                    continue

                if current_norm == "ta" and index + 1 < len(matches):
                    corrected_next = self.correct_word(next_word)
                    previous_tail = (
                        previous_surface_word.split("-", 1)[-1]
                        if previous_surface_word and "-" in previous_surface_word
                        else previous_surface_word
                    )
                    prev_allows_ta = (
                        previous_surface_word is None
                        or self._is_noun_tagged_word(previous_surface_word)
                        or self._is_pronoun_tagged_word(previous_surface_word)
                        or self._is_noun_tagged_word(previous_tail or "")
                        or self._is_adjective_tagged_word(previous_tail or "")
                        or self._capitalized_name_kind(previous_surface_word)
                        or self._accepted_exact_english(previous_surface_word)
                        or (
                            previous_surface_word
                            and "-" in previous_surface_word
                            and self._accepted_exact_english(
                                previous_surface_word.split("-", 1)[-1]
                            )
                        )
                    )
                    next_is_nominal = (
                        self._is_noun_tagged_word(corrected_next)
                        or self._is_adjective_tagged_word(corrected_next)
                        or self._is_pronoun_tagged_word(corrected_next)
                        or self._capitalized_name_kind(corrected_next)
                        or next_norm in {"min", "minn"}
                    )
                    next_is_adverb = self._is_adverb_tagged_word(corrected_next)
                    if (
                        (prev_allows_ta and next_is_nominal)
                        or (next_is_adverb and not self._is_verb_tagged_word(corrected_next))
                    ):
                        ta_surface = "t'" if self._starts_vowel_gh_or_h(
                            self._normalize_word(corrected_next)
                        ) else "ta'"
                        corrected_phrase = self._apply_surface_case(
                            original_word,
                            f"{ta_surface} {corrected_next}" if ta_surface == "ta'" else f"{ta_surface}{corrected_next}",
                            sentence_initial=sentence_initial,
                        )
                        phrase_choices = [
                            {
                                "word": corrected_phrase,
                                "meaning": self.meaning_for(corrected_next),
                            },
                            {
                                "word": self._apply_surface_case(
                                    original_word,
                                    f"ta {corrected_next}",
                                    sentence_initial=sentence_initial,
                                ),
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
                    current_norm == "x"
                    and separator_between.isspace()
                    and next_norm.startswith("l")
                    and len(next_norm) > 1
                ):
                    tail = next_norm[1:].lstrip("-")
                    corrected_tail = (
                        article_rules._strict_dictionary_tail(tail)
                        if article_rules is not None
                        else None
                    )
                    corrected_tail = corrected_tail or self.correct_word(tail)
                    corrected_tail_norm = self._normalize_word(corrected_tail)
                    if corrected_tail_norm and (
                        self._is_noun_tagged_word(corrected_tail_norm)
                        or self._is_adjective_tagged_word(corrected_tail_norm)
                        or self._is_pronoun_tagged_word(corrected_tail_norm)
                    ):
                        corrected_phrase = self._apply_surface_case(
                            original_word,
                            f"x'l-{corrected_tail_norm}",
                            sentence_initial=sentence_initial,
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
                        previous_surface_word = corrected_tail_norm
                        last_end = matches[index + 1].end()
                        index += 2
                        continue

                if (
                    match.end() == matches[index + 1].start()
                    and current_norm == "x"
                    and next_norm
                ):
                    combined_original = text[
                        matches[index].start() : matches[index + 1].end()
                    ]
                    restored_combined = self._medial_guttural_vowel_restoration_variants(
                        combined_original
                    )
                    if restored_combined:
                        corrected_word = self._apply_surface_case(
                            combined_original,
                            restored_combined[0],
                            sentence_initial=sentence_initial,
                        )
                        tokens.append(
                            {
                                "type": "word",
                                "original": combined_original,
                                "corrected": corrected_word,
                                "ambiguous": False,
                                "choices": [],
                                "name_like": False,
                            }
                        )
                        corrected_parts.append(corrected_word)
                        previous_surface_word = self._normalize_word(corrected_word)
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
                        or self._is_preposition_tagged_word(corrected_next)
                    )

                    if is_target_pos:
                        primary_form = self._xi_form_for_word(corrected_next)

                        # A preceding pronoun can keep the long form.  Finite
                        # verbs do not on their own block x' before a valid
                        # vowel/CV continuation (ma kellix x'niekol).
                        if previous_surface_word and self._is_pronoun_tagged_word(
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

                previous_word = None if sentence_initial else (
                    word_tokens[index - 1].text if index > 0 else None
                )
                article_match = article_rules.match_compact_definite_article(
                    original_word,
                    previous=previous_word,
                )
                # bix-/fix- are compact assimilated prepositions.  Their
                # verified x-tail repair is handled by Phase X; parsing them
                # here first would incorrectly split bixiraq as "bi xieraq".
                if (
                    "-" not in original_norm
                    and original_norm.startswith(("bix", "fix"))
                ) or (
                    original_norm.startswith("inm")
                ):
                    article_match = None
                elif original_norm not in self.dictionary_set:
                    article_match = article_match or article_rules.match_compact_preposition_article(
                        original_word,
                    )

                if article_match is not None:
                    compact_corrected = self._match_hyphenated_tail_capitalisation(
                        original_word.split("-", 1)[-1],
                        article_match.corrected,
                    )
                    if sentence_initial or self._is_initial_capitalized(original_word):
                        compact_corrected = self._capitalize_first_letter(
                            compact_corrected
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
                    if sentence_initial or self._is_initial_capitalized(original_word):
                        hyphenated_corrected = self._capitalize_first_letter(
                            hyphenated_corrected
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

            if (
                fused_preposition_rules is not None
                and not (
                    "-" not in original_norm
                    and original_norm.startswith(("bix", "fix"))
                )
            ):
                fused_match = fused_preposition_rules.match(original_word)

                if fused_match is not None:
                    fused_corrected = self._apply_surface_case(
                        original_word,
                        self._contract_negative_ma(
                            fused_match.corrected
                        ),
                        sentence_initial=sentence_initial,
                    )
                    fused_choices = [
                        {
                            **choice,
                            "word": self._apply_surface_case(
                                original_word,
                                self._contract_negative_ma(
                                    choice.get("word", "")
                                ),
                                sentence_initial=sentence_initial,
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

            if (
                self._is_initial_capitalized(original_word)
                and not original_word.isupper()
                and not sentence_initial
                and original_norm not in self.MANUAL_WORD_REPAIRS
            ):
                final_capitalized = self._conservative_capitalized_word(original_word)
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

            repaired_initial_name = (
                self._capitalized_name_repair(original_word)
                if sentence_initial and self._is_initial_capitalized(original_word)
                else None
            )
            corrected_word = (
                repaired_initial_name
                or (
                    self._conservative_capitalized_word(original_word)
                    if (
                        sentence_initial
                        and self._is_initial_capitalized(original_word)
                        and self._capitalized_name_kind(original_word)
                        and self._initial_i_surface_repair(original_norm) is None
                        and not original_norm.startswith("x'")
                    )
                    else self.correct_word(original_word)
                )
            )
            if (
                not sentence_initial
                and self._is_initial_capitalized(original_word)
                and not original_word.isupper()
                and not self._capitalized_name_kind(original_word)
            ):
                corrected_word = self._normalize_word(corrected_word)
            restorative_norms = set()
            if not self._is_recognized_surface(original_norm):
                restorative_norms = {
                    self._normalize_word(candidate)
                    for candidate in self._medial_guttural_vowel_restoration_variants(
                        original_word
                    )
                }
            if restorative_norms and self._normalize_word(corrected_word) in restorative_norms:
                surface_word = self._apply_surface_case(
                    original_word,
                    corrected_word,
                    sentence_initial=sentence_initial,
                )
                tokens.append(
                    {
                        "type": "word",
                        "original": original_word,
                        "corrected": surface_word,
                        "meaning": self.meaning_for(corrected_word),
                        "ambiguous": False,
                        "choices": [],
                        "name_like": False,
                    }
                )
                corrected_parts.append(surface_word)
                previous_surface_word = self._normalize_word(surface_word)
                last_end = match.end()
                index += 1
                continue

            corrected_word = self._contract_negative_ma(corrected_word)
            choices = self.ambiguity_choices(
                original_word,
                corrected_word,
                limit=3,
                edit_distance_tolerance=effective_tolerance,
            ) if not bulk_mode else self._bulk_ambiguity_choices(
                original_word,
                corrected_word,
                limit=3,
            )
            if (
                sentence_initial
                and self._is_initial_capitalized(original_word)
                and self._normalize_word(corrected_word)
                == self._normalize_word(original_word)
                and self._normalize_word(original_word)
                not in self.EXACT_SUGGESTION_OVERRIDES
            ):
                choices = []
            preferred_apostrophe = self._preferred_apostrophe_choice(choices)
            if (
                preferred_apostrophe
                and "'" in original_norm
                and original_norm not in self.dictionary_set
                and self._normalize_word(
                    preferred_apostrophe
                ) != self._normalize_word(corrected_word)
                and self._normalize_word(
                    corrected_word
                ) not in restorative_norms
            ):
                corrected_word = preferred_apostrophe
                choices = self.ambiguity_choices(
                    original_word,
                    corrected_word,
                    limit=3,
                    edit_distance_tolerance=effective_tolerance,
                ) if not bulk_mode else self._bulk_ambiguity_choices(
                    original_word,
                    corrected_word,
                    limit=3,
                )

            corrected_word = self._feminine_imperfect_continuation(
                original_word,
                corrected_word,
                previous_surface_word,
            )
            surface_word, corrected_word = self._phase_z_finalize_surface_word(
                original_word,
                corrected_word,
                previous_surface_word=previous_surface_word,
                sentence_initial=sentence_initial,
                prefer_initial_vowel_surface=prefer_initial_vowel_surface,
            )

            if self._normalize_word(surface_word) != self._normalize_word(corrected_word):
                choices = []

            if sentence_initial:
                capitalized_choices = []
                for choice in choices:
                    capitalized_choices.append(
                        {
                            **choice,
                            "word": self._capitalize_first_letter(
                                choice.get("word", "")
                            ),
                        }
                    )
                choices = capitalized_choices

            is_name_like = (
                self._is_initial_capitalized(original_word) and not sentence_initial
            )

            is_ambiguous = len(choices) >= 2 and self._normalize_word(
                choices[0]["word"]
            ) != self._normalize_word(choices[1]["word"])

            is_crucial = is_ambiguous and (
                original_norm in self.MANUAL_WORD_SUGGESTIONS
            )

            tokens.append(
                {
                    "type": "word",
                    "original": original_word,
                    "corrected": surface_word,
                    "meaning": self.meaning_for(corrected_word),
                    "ambiguous": is_ambiguous,
                    "crucial": is_crucial,
                    "choices": choices if is_ambiguous else [],
                    "name_like": is_name_like,
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

        corrected_parts = [part for part in corrected_parts if part is not None]
        corrected_text = self._ensure_terminal_period("".join(corrected_parts))
        self._add_country_translation_choices(tokens)
        self._mark_unrecognized_tokens(tokens)

        # Do not attach meanings to ordinary corrected tokens.  Re-enable
        # lookup only for visible choices, which keeps the UI quiet and avoids
        # suffix-meaning work across every word in a long text.
        self._local.suppress_meanings = False
        self.meaning_for.cache_clear()
        for token in tokens:
            choices = token.get("choices", []) if isinstance(token, dict) else []
            if not choices:
                if isinstance(token, dict):
                    token["meaning"] = ""
                continue
            for choice in choices:
                if not isinstance(choice, dict):
                    continue
                if not choice.get("meaning"):
                    choice["meaning"] = self.meaning_for(choice.get("word", ""))

        return {
            "corrected_text": corrected_text,
            "tokens": tokens,
        }


# -----------------------------------------------------------------------------

def _trim_request_caches() -> None:
    """
    Partially clear high-churn, per-request caches to keep memory bounded.
    Called after every check-text request. Only clears caches that fill up
    with user-input-derived keys (word pairs etc.), not dictionary-derived ones.
    """
    import gc
    spellchecker._word_distance.cache_clear()
    spellchecker._damerau_levenshtein_distance.cache_clear()
    spellchecker._extract_consonant_anchor.cache_clear()
    spellchecker._vowel_slots.cache_clear()
    spellchecker._count_vowels.cache_clear()
    spellchecker._get_candidates_cached.cache_clear()
    gc.collect()


def _annotated_usage_output(tokens: list[dict], corrected_text: str) -> str:
    parts: list[str] = []
    saw_structured_token = False
    for token in tokens:
        if not isinstance(token, dict):
            continue
        if token.get("type") == "text":
            parts.append(str(token.get("text", "")))
            continue

        corrected = str(token.get("corrected", ""))
        if not corrected:
            continue

        saw_structured_token = True
        labels: list[str] = []
        if token.get("choices"):
            labels.append("(suggestion)")
        if token.get("unrecognized"):
            labels.append("(unrecognized)")
        parts.append(corrected + (" " + " ".join(labels) if labels else ""))

    annotated = "".join(parts)
    if not saw_structured_token:
        return corrected_text
    if corrected_text.endswith((".", "?", "!")) and not annotated.rstrip().endswith((".", "?", "!")):
        annotated = annotated.rstrip() + corrected_text[-1]
    return annotated


def _append_usage_log(
    *,
    request_id: str,
    original_text: str,
    corrected_text: str,
    tokens: list[dict],
) -> None:
    if not USAGE_LOG_ENABLED:
        return

    annotated = _annotated_usage_output(tokens, corrected_text)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S %z")
    entry = (
        f"[{timestamp}] request_id={request_id}\n"
        "INPUT:\n"
        f"{original_text}\n\n"
        "OUTPUT:\n"
        f"{corrected_text}\n\n"
        "ANNOTATED OUTPUT:\n"
        f"{annotated}\n"
        + ("-" * 80)
        + "\n"
    )

    try:
        USAGE_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with USAGE_LOG_LOCK:
            with USAGE_LOG_FILE.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(entry)
    except Exception:
        try:
            app.logger.exception(
                "SPELLCHECK request_id=%s stage=usage_log_write_failed",
                request_id,
            )
        except NameError:
            pass

# Flask app
# -----------------------------------------------------------------------------

app = Flask(__name__)

_startup_started = time.perf_counter()
spellchecker = UniversalMalteseSpellchecker(dictionary_files=DICTIONARY_FILES)
meaning_index = MeaningIndex()
meaning_index.load_entries(spellchecker.raw_entries, include_verbs=True)
spellchecker.raw_entries = []
article_phrase_rules = MalteseArticlePhraseRules(
    meaning_index=meaning_index,
    normalizer=spellchecker._normalize_word,
    noun_words=spellchecker.tagged_words_with_marker("NOUN"),
    num_words=spellchecker.tagged_words_with_marker("NUM"),
)
article_phrase_rules.spellchecker = spellchecker
spellchecker.article_phrase_rules = article_phrase_rules

grammar_rule_engine = MalteseGrammarRuleEngine(
    rules_path=BASE_DIR / "grammar" / "grammar_rules_measured.json",
    spellchecker=spellchecker,
    meaning_index=meaning_index,
    article_rules=article_phrase_rules,
)
spellchecker.grammar_rule_engine = grammar_rule_engine
sentence_context_analyzer = OptionalSentenceContextAnalyzer(
    backend=SENTENCE_CONTEXT_BACKEND,
    enabled=ENABLE_SENTENCE_CONTEXT_ANALYZER,
)

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

spellchecker.clear_disposable_startup_caches()

log_spellcheck_event(
    event="SPELLCHECK_STARTUP",
    stage="startup_complete",
    instance_id=spellchecker.instance_id,
    elapsed_ms=(time.perf_counter() - _startup_started) * 1000,
    dictionary_words=len(spellchecker.dictionary),
    paradigms=len(spellchecker.paradigm_forms),
    suffix_verb_records=(
        spellchecker.suffix_generator.verb_index.record_count()
        if hasattr(spellchecker, "suffix_generator")
        else None
    ),
    rss_mb=rss_mb(),
)


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
    return jsonify(
        {
            "ok": True,
            "status": "ok",
            "dictionary_words": len(spellchecker.dictionary),
            "paradigms": len(spellchecker.paradigm_forms),
        }
    ), 200


@app.post("/check-text")
def check_text():
    profiler = RequestProfiler()
    profiler_token = set_current_profiler(profiler)
    token_count = 0
    unique_tokens = 0
    data = request.get_json(silent=True) or {}
    try:
        log_spellcheck_event(
            event="SPELLCHECK_REQUEST",
            instance_id=spellchecker.instance_id,
            request_id=profiler.request_id,
        )
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

        request_words = [
            match.group(0) for match in spellchecker.WORD_PATTERN.finditer(text)
        ]
        token_count = len(request_words)
        unique_tokens = len(
            {spellchecker._normalize_word(word) for word in request_words}
        )

        edit_distance_tolerance = int(data.get("edit_distance_tolerance", 1))
        include_grammar = bool(data.get("include_grammar", True))
        with profiler.span(
            "correct_text_rich",
            tokens=token_count,
            unique_tokens=unique_tokens,
        ):
            result = spellchecker.correct_text_rich(
                text, edit_distance_tolerance=edit_distance_tolerance
        )
        corrected_text = result["corrected_text"]
        tokens = result["tokens"]
        grammar_findings: list[dict[str, object]] = []
        grammar_enabled = bool(include_grammar and grammar_rule_engine is not None)
        if grammar_enabled:
            corrected_request_words = [
                match.group(0)
                for match in spellchecker.WORD_PATTERN.finditer(corrected_text)
            ]
            grammar_findings = grammar_rule_engine.analyze(
                text=corrected_text,
                request_words=corrected_request_words,
                tokens=tokens,
            )
            corrected_text, tokens, _ = grammar_rule_engine.apply_safe_rewrites(
                original_text=corrected_text,
                corrected_text=corrected_text,
                tokens=tokens,
            )

        context_shadow = None
        if sentence_context_analyzer.enabled:
            with profiler.span(
                "sentence_context_analyzer",
                backend=sentence_context_analyzer.backend,
            ):
                context_shadow = asdict(sentence_context_analyzer.analyze(corrected_text))

        _append_usage_log(
            request_id=profiler.request_id,
            original_text=text,
            corrected_text=corrected_text,
            tokens=tokens,
        )

        response_payload = {
            "original_text": text,
            "corrected_text": corrected_text,
            "changed": corrected_text != text,
            "tokens": tokens,
        }
        if grammar_enabled:
            response_payload["grammar_enabled"] = True
            response_payload["grammar_findings"] = grammar_findings
        if context_shadow is not None:
            response_payload["context_analyzer"] = context_shadow
        return jsonify(response_payload)
    except Exception:
        app.logger.exception(
            "SPELLCHECK request_id=%s stage=exception", profiler.request_id
        )
        return jsonify({"error": "Internal spell-checking error."}), 500
    finally:
        profiler.finish(token_count=token_count, unique_tokens=unique_tokens)
        reset_current_profiler(profiler_token)
        # Trim request-scoped caches to prevent unbounded memory growth across requests.
        # We only trim caches whose entries are request-specific (word pairs, scores).
        # Dictionary-based caches (_normalize, _graphemes, tag lookups) are kept warm
        # because they store pre-computed facts about the dictionary words, not user text.
        _trim_request_caches()


@app.post("/suggest-word")
def suggest_word():
    data = request.get_json(silent=True) or {}
    word = data.get("word", "")
    spellchecker._reset_request_token_cache()

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
    spellchecker.correct_word(word)

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
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False,
        use_reloader=False,
    )
