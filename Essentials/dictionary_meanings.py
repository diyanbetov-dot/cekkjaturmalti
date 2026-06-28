#!/usr/bin/env python3
"""
Dictionary meaning lookup for the Maltese spell-checker.

Supported examples
------------------
    abaku/SINGNOUNM-abacus
    ikrah/SINGADJM-ugly
    kiser/T-ksr-PERF-3SM-to break or divert
    kisirx/T-ksr-PERF-3SM-to not break or not divert-N

For ordinary entries, the meaning follows the final morphological tag.
For negative verbs, the final ``N`` is a negative marker, so the meaning
precedes it.

Unlike a plain ``split("-")[-1]``, this parser preserves meanings containing
hyphens, such as ``built-up area`` or ``short-toed snake eagle``.
"""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence


PERSON_TAGS = {
    "1S", "2S", "3SM", "3SF", "1P", "2P", "3P", "12S", "12P",
}

# Match the POS marker itself. The meaning starts after the following dash.
POS_MARKER_RE = re.compile(
    r"(?:^|-)"
    r"(?:MLT-)?"
    r"(?:"
    r"SINGNOUN(?:M|F|FM)?|PLUNOUN|PAUCNOUN|COLLNOUN|DUALNOUN|"
    r"SINGADJM|SINGADJF|PLUADJ|SINGADJM\+SINGNOUN|"
    r"ADVERB(?:-\d(?:S|P|SM|SF))?|"
    r"PRON(?:-\d(?:S|P|SM|SF))?|"
    r"PLACE|DNYMM|DYNMF|DYNMPL"
    r")-",
    re.IGNORECASE,
)

# Person marker in a verb payload. The meaning starts after this marker.
PERSON_MARKER_RE = re.compile(
    r"(?:^|-)(?:1S|2S|3SM|3SF|1P|2P|3P|12S|12P)-",
    re.IGNORECASE,
)

VERB_FORM_CLASS_ONLY_RE = re.compile(
    r"^(?:T|Q|S|AS|IS)-[^-]+-F\d+$",
    re.IGNORECASE,
)

VERB_TENSES = {"IMP", "PERF", "MPERF"}
SUBJECT_PRONOUNS = {
    "1S": "I",
    "2S": "you",
    "3SM": "he",
    "3SF": "she",
    "1P": "we",
    "2P": "you all",
    "3P": "they",
    "12S": "we",
    "12P": "we",
}
DO_SUFFIX_GLOSSES = {
    "1S": "me",
    "2S": "you",
    "3SM": "him",
    "3SF": "her",
    "1P": "us",
    "2P": "you all",
    "3P": "them",
}
IDO_SUFFIX_GLOSSES = {
    "1S": "my",
    "2S": "your",
    "3SM": "his",
    "3SF": "her",
    "1P": "our",
    "2P": "your (plural)",
    "3P": "their",
}
IRREGULAR_PRESENT_3SG = {
    "be": "is",
    "have": "has",
    "do": "does",
    "go": "goes",
}
IRREGULAR_PAST = {
    "abide": "abode",
    "be": "was",
    "become": "became",
    "begin": "began",
    "break": "broke",
    "bring": "brought",
    "build": "built",
    "buy": "bought",
    "catch": "caught",
    "choose": "chose",
    "come": "came",
    "cut": "cut",
    "dig": "dug",
    "do": "did",
    "draw": "drew",
    "drink": "drank",
    "drive": "drove",
    "eat": "ate",
    "fall": "fell",
    "feel": "felt",
    "fight": "fought",
    "find": "found",
    "forget": "forgot",
    "forgive": "forgave",
    "freeze": "froze",
    "get": "got",
    "give": "gave",
    "go": "went",
    "grow": "grew",
    "have": "had",
    "hear": "heard",
    "hide": "hid",
    "hold": "held",
    "keep": "kept",
    "know": "knew",
    "lead": "led",
    "leave": "left",
    "let": "let",
    "lose": "lost",
    "make": "made",
    "mean": "meant",
    "meet": "met",
    "pay": "paid",
    "put": "put",
    "read": "read",
    "ride": "rode",
    "ring": "rang",
    "rise": "rose",
    "run": "ran",
    "say": "said",
    "see": "saw",
    "sell": "sold",
    "send": "sent",
    "set": "set",
    "shake": "shook",
    "shoot": "shot",
    "sing": "sang",
    "sit": "sat",
    "sleep": "slept",
    "speak": "spoke",
    "spend": "spent",
    "stand": "stood",
    "swim": "swam",
    "take": "took",
    "teach": "taught",
    "tell": "told",
    "think": "thought",
    "understand": "understood",
    "wear": "wore",
    "win": "won",
    "write": "wrote",
}


def normalize_word(value: str) -> str:
    return (
        unicodedata.normalize("NFC", str(value).strip().casefold())
        .replace("\u2019", "'")
        .replace("\u02bc", "'")
        .replace("\u2018", "'")
    )


def normalize_word_exact(value: str) -> str:
    return (
        unicodedata.normalize("NFC", str(value).strip())
        .replace("\u2019", "'")
        .replace("\u02bc", "'")
        .replace("\u2018", "'")
    )


def clean_meaning(value: str) -> str:
    value = re.sub(r"\s+", " ", str(value or "")).strip()
    return value.strip(" \t\r\n:;,")


def _split_or_clauses(value: str) -> list[str]:
    return [part.strip() for part in re.split(r"\s+or\s+", value) if part.strip()]


def _normalise_infinitive_gloss(gloss: str, *, negative: bool) -> list[str]:
    cleaned = clean_meaning(gloss)
    if cleaned.casefold().startswith("to "):
        cleaned = cleaned[3:].strip()

    clauses = _split_or_clauses(cleaned) or [cleaned]
    normalized_clauses: list[str] = []
    for clause in clauses:
        if negative and clause.casefold().startswith("not "):
            clause = clause[4:].strip()
        normalized_clauses.append(clause)
    return normalized_clauses


def _present_third_person(verb: str) -> str:
    lowered = verb.casefold()
    if lowered in IRREGULAR_PRESENT_3SG:
        return IRREGULAR_PRESENT_3SG[lowered]
    if lowered.endswith("y") and len(lowered) > 1 and lowered[-2] not in "aeiou":
        return verb[:-1] + "ies"
    if lowered.endswith(("s", "sh", "ch", "x", "z", "o")):
        return verb + "es"
    return verb + "s"


def _regular_past(verb: str) -> str:
    lowered = verb.casefold()
    if lowered.endswith("e"):
        return verb + "d"
    if lowered.endswith("y") and len(lowered) > 1 and lowered[-2] not in "aeiou":
        return verb[:-1] + "ied"
    return verb + "ed"


def _past_form(verb: str) -> str:
    return IRREGULAR_PAST.get(verb.casefold(), _regular_past(verb))


def _conjugate_be(rest: str, tense: str, person: str, negative: bool) -> str:
    trimmed_rest = rest.strip()

    if tense == "MPERF":
        positive = {
            "1S": "am",
            "2S": "are",
            "3SM": "is",
            "3SF": "is",
            "1P": "are",
            "2P": "are",
            "3P": "are",
            "12S": "are",
            "12P": "are",
        }[person]
        negative_form = {
            "1S": "am not",
            "2S": "are not",
            "3SM": "is not",
            "3SF": "is not",
            "1P": "are not",
            "2P": "are not",
            "3P": "are not",
            "12S": "are not",
            "12P": "are not",
        }[person]
        verb_phrase = negative_form if negative else positive
    elif tense == "PERF":
        positive = {
            "1S": "was",
            "2S": "were",
            "3SM": "was",
            "3SF": "was",
            "1P": "were",
            "2P": "were",
            "3P": "were",
            "12S": "were",
            "12P": "were",
        }[person]
        negative_form = {
            "1S": "was not",
            "2S": "were not",
            "3SM": "was not",
            "3SF": "was not",
            "1P": "were not",
            "2P": "were not",
            "3P": "were not",
            "12S": "were not",
            "12P": "were not",
        }[person]
        verb_phrase = negative_form if negative else positive
    else:
        verb_phrase = "don't be" if negative else "be"

    return f"{verb_phrase} {trimmed_rest}".strip()


def _conjugate_clause(clause: str, tense: str, person: str, negative: bool) -> str:
    words = clause.split(None, 1)
    if not words:
        return clause

    verb = words[0]
    rest = words[1] if len(words) > 1 else ""

    if verb.casefold() == "be":
        return _conjugate_be(rest, tense, person, negative)

    if tense == "IMP":
        prefix = "don't " if negative else ""
        return f"{prefix}{clause}".strip()

    if tense == "PERF":
        if negative:
            return f"didn't {clause}".strip()
        return f"{_past_form(verb)} {rest}".strip()

    if negative:
        helper = "doesn't" if person in {"3SM", "3SF"} else "don't"
        return f"{helper} {clause}".strip()

    if person in {"3SM", "3SF"}:
        return f"{_present_third_person(verb)} {rest}".strip()
    return clause


def _combine_negative_predicates(clauses: list[str], tense: str, person: str) -> str:
    split_clauses = [clause.split(None, 1) for clause in clauses if clause.strip()]
    if not split_clauses:
        return ""

    if all(parts and parts[0].casefold() == "be" for parts in split_clauses):
        rests = [parts[1].strip() if len(parts) > 1 else "" for parts in split_clauses]
        if tense == "PERF":
            helper = "was not" if person in {"1S", "3SM", "3SF"} else "were not"
        elif tense == "MPERF":
            helper = "is not" if person in {"3SM", "3SF"} else (
                "am not" if person == "1S" else "are not"
            )
        else:
            helper = "don't be"
        return f"{helper} {' or '.join(part for part in rests if part).strip()}".strip()

    if tense == "PERF":
        helper = "didn't"
    elif tense == "MPERF":
        helper = "doesn't" if person in {"3SM", "3SF"} else "don't"
    else:
        helper = "don't"
    return f"{helper} {' or '.join(clauses)}".strip()


def _combine_predicates(clauses: list[str], tense: str, person: str, negative: bool) -> str:
    if negative:
        return _combine_negative_predicates(clauses, tense, person)
    converted = [
        _conjugate_clause(clause, tense, person, negative)
        for clause in clauses
    ]
    return " or ".join(converted)


def _suffix_phrase(kind: str | None, person: str | None) -> str:
    if not kind or not person:
        return ""
    if kind == "DO":
        return DO_SUFFIX_GLOSSES.get(person, "")
    if kind == "IDO":
        return IDO_SUFFIX_GLOSSES.get(person, "")
    if kind == "DO_IDO":
        direct_person, _, possessive_person = person.partition("+")
        direct = DO_SUFFIX_GLOSSES.get(direct_person, "").strip()
        possessive = IDO_SUFFIX_GLOSSES.get(possessive_person, "").strip()
        return " ".join(part for part in (direct, possessive) if part).strip()
    return ""


def format_verb_meaning_from_gloss(
    gloss: str,
    *,
    tense: str,
    person: str,
    negative: bool = False,
    suffix_kind: str | None = None,
    suffix_person: str | None = None,
) -> str:
    subject = SUBJECT_PRONOUNS.get(person, "it")
    clauses = _normalise_infinitive_gloss(gloss, negative=negative)
    predicate = _combine_predicates(clauses, tense, person, negative)
    suffix = _suffix_phrase(suffix_kind, suffix_person)

    if tense == "IMP":
        command_predicate = f"{predicate} {suffix}".strip() if suffix else predicate
        return f"{subject}, {command_predicate} (command)"

    finite = f"{subject} {predicate}".strip()
    if suffix:
        finite = f"{finite} {suffix}"
    return finite


def parse_verb_payload(payload: str) -> dict[str, str | bool] | None:
    raw = str(payload or "").strip()
    if not raw:
        return None

    if "/" in raw:
        raw = raw.split("/", 1)[1].strip()

    negative = raw.upper().endswith("-N")
    if negative:
        raw = raw[:-2].rstrip("-")

    parts = raw.split("-")
    for index in range(len(parts) - 2):
        tense = parts[index].upper()
        person = parts[index + 1].upper()
        if tense in VERB_TENSES and person in PERSON_TAGS:
            gloss = clean_meaning("-".join(parts[index + 2:]))
            if gloss:
                return {
                    "tense": tense,
                    "person": person,
                    "gloss": gloss,
                    "negative": negative,
                }
    return None


def format_verb_payload_meaning(payload: str) -> str:
    parsed = parse_verb_payload(payload)
    if not parsed:
        return ""
    return format_verb_meaning_from_gloss(
        str(parsed["gloss"]),
        tense=str(parsed["tense"]),
        person=str(parsed["person"]),
        negative=bool(parsed["negative"]),
    )


def format_suffix_candidate_meaning(candidate, fallback_gloss: str = "") -> str:
    payload_meaning = format_verb_payload_meaning(getattr(candidate, "raw_tag", ""))
    parsed = parse_verb_payload(getattr(candidate, "raw_tag", ""))

    if not parsed:
        if not fallback_gloss:
            return payload_meaning
        return fallback_gloss

    gloss = str(parsed["gloss"])
    return format_verb_meaning_from_gloss(
        gloss or fallback_gloss,
        tense=getattr(candidate, "tense", str(parsed["tense"])),
        person=getattr(candidate, "person", str(parsed["person"])),
        negative=bool(parsed["negative"]),
        suffix_kind=getattr(candidate, "suffix_kind", ""),
        suffix_person=getattr(candidate, "suffix_person", ""),
    )


def split_dictionary_line(line: str) -> tuple[str, str]:
    """
    Return ``(surface, payload)``.

    ``payload`` is everything after the first slash. Lines without a slash are
    plain Hunspell words and therefore have no embedded meaning.
    """
    line = str(line).strip()
    if not line or line.isdigit() or "/" not in line:
        return "", ""

    surface, payload = line.split("/", 1)
    return surface.strip(), payload.strip()


def extract_meaning_from_payload(payload: str) -> str:
    """
    Extract the complete meaning from a dictionary payload.

    The final ``-N`` is removed only when it is the negative-verb marker.
    Hyphens inside the meaning are retained.
    """
    raw = str(payload or "").strip()
    if not raw:
        return ""
    if raw.upper() == "MLT-PLACE":
        return ""

    if "/" in raw:
        raw = raw.split("/", 1)[1].strip()

    if VERB_FORM_CLASS_ONLY_RE.fullmatch(raw):
        return ""

    # Nouns, adjectives, adverbs, and pronouns.
    match = POS_MARKER_RE.search(raw)
    if match:
        return clean_meaning(raw[match.end():])

    verb_meaning = format_verb_payload_meaning(raw)
    if verb_meaning:
        return verb_meaning

    if raw.upper().endswith("-N"):
        raw = raw[:-2].rstrip("-")

    # Verbs compatibility fallback: locate the person tag and keep the complete remaining suffix.
    matches = list(PERSON_MARKER_RE.finditer(raw))
    if matches:
        return clean_meaning(raw[matches[-1].end():])

    # Compatibility fallback following the user's stated convention.
    parts = raw.split("-")
    if parts and re.fullmatch(r"F\d+", parts[-1], re.IGNORECASE):
        return ""
    return clean_meaning(parts[-1] if parts else "")


def extract_meaning_from_line(line: str) -> tuple[str, str]:
    surface, payload = split_dictionary_line(line)
    if not surface or not payload:
        return "", ""
    return surface, extract_meaning_from_payload(payload)


class MeaningIndex:
    """Word -> one or more meanings collected from one or more .dic files."""

    def __init__(self, dictionary_files: Iterable[str | Path] = ()) -> None:
        self._meanings: dict[str, list[str]] = defaultdict(list)
        self._exact_meanings: dict[str, list[str]] = defaultdict(list)
        self.load_files(dictionary_files)

    def add(self, word: str, meaning: str) -> None:
        key = normalize_word(word)
        exact_key = normalize_word_exact(word)
        meaning = clean_meaning(meaning)
        if not key or not meaning:
            return
        if meaning not in self._meanings[key]:
            self._meanings[key].append(meaning)
        if exact_key and meaning not in self._exact_meanings[exact_key]:
            self._exact_meanings[exact_key].append(meaning)

    def load_line(self, line: str) -> None:
        word, meaning = extract_meaning_from_line(line)
        self.add(word, meaning)

    def load_file(self, path: str | Path) -> None:
        path = Path(path)
        try:
            lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
        except FileNotFoundError:
            return

        if lines and lines[0].strip().isdigit():
            lines = lines[1:]

        for line in lines:
            self.load_line(line)

    def load_files(self, paths: Iterable[str | Path]) -> None:
        for path in paths:
            self.load_file(path)

    def load_entries(
        self,
        entries: Iterable[tuple[str, str | None]],
        *,
        include_verbs: bool = False,
    ) -> None:
        for word, payload in entries:
            if not word or not payload:
                continue
            if not include_verbs and str(payload).startswith(
                ("T-", "Q-", "S-", "AS-", "IS-")
            ):
                continue
            self.add(word, extract_meaning_from_payload(payload))

    def meanings_for(self, word: str) -> list[str]:
        exact = self._exact_meanings.get(normalize_word_exact(word), ())
        if exact:
            return list(exact)
        return list(self._meanings.get(normalize_word(word), ()))

    def meaning_for(self, word: str) -> str:
        return " / ".join(self.meanings_for(word))

    def choice(self, word: str) -> dict[str, str]:
        return {
            "word": word,
            "meaning": self.meaning_for(word),
        }

    def enrich_choice(self, choice):
        """
        Accept either a suggestion string or an existing suggestion mapping.
        Existing non-empty meanings are preserved.
        """
        if isinstance(choice, str):
            return self.choice(choice)

        if isinstance(choice, Mapping):
            result = dict(choice)
            word = (
                result.get("word")
                or result.get("candidate")
                or result.get("corrected")
                or result.get("surface")
                or ""
            )
            result.setdefault("word", word)

            if not clean_meaning(result.get("meaning", "")):
                result["meaning"] = self.meaning_for(word)

            return result

        return {"word": str(choice), "meaning": ""}

    def enrich_choices(self, choices: Sequence) -> list[dict]:
        return [self.enrich_choice(choice) for choice in choices]


def _self_test() -> None:
    examples = {
        "abaku/SINGNOUNM-abacus": ("abaku", "abacus"),
        "ikrah/SINGADJM-ugly": ("ikrah", "ugly"),
        "koroh/PLUADJ-ugly": ("koroh", "ugly"),
        "kiser/T-ksr-PERF-3SM-to break or divert": (
            "kiser", "he broke or diverted"
        ),
        "kisirx/T-ksr-PERF-3SM-to not break or not divert-N": (
            "kisirx", "he did not break or divert"
        ),
        "abitat/SINGNOUNM-built-up area": (
            "abitat", "built-up area"
        ),
        "ajkla bajda/SINGNOUNF-short-toed snake eagle": (
            "ajkla bajda", "short-toed snake eagle"
        ),
        "kiteb/F1-T-ktb-PERF-3SM-to write": (
            "kiteb", "he wrote"
        ),
    }

    for line, expected in examples.items():
        actual = extract_meaning_from_line(line)
        assert actual == expected, (line, actual, expected)

    index = MeaningIndex()
    for line in examples:
        index.load_line(line)

    assert index.choice("kisirx") == {
        "word": "kisirx",
        "meaning": "he did not break or divert",
    }

    assert format_verb_payload_meaning("T-ghml-MPERF-1S-to make or do") == (
        "I make or do"
    )
    assert format_verb_payload_meaning("T-ghml-IMP-2P-to make or do") == (
        "you all, make or do (command)"
    )

    print("dictionary_meanings.py self-test passed.")


if __name__ == "__main__":
    _self_test()
