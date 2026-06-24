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
    r"(?:"
    r"SINGNOUN(?:M|F|FM)?|PLUNOUN|PAUCNOUN|COLLNOUN|DUALNOUN|"
    r"SINGADJM|SINGADJF|PLUADJ|SINGADJM\+SINGNOUN|"
    r"ADVERB(?:-\d(?:S|P|SM|SF))?|"
    r"PRON(?:-\d(?:S|P|SM|SF))?"
    r")-",
    re.IGNORECASE,
)

# Person marker in a verb payload. The meaning starts after this marker.
PERSON_MARKER_RE = re.compile(
    r"(?:^|-)(?:1S|2S|3SM|3SF|1P|2P|3P|12S|12P)-",
    re.IGNORECASE,
)


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

    if "/" in raw:
        raw = raw.split("/", 1)[1].strip()

    if raw.upper().endswith("-N"):
        raw = raw[:-2].rstrip("-")

    # Nouns, adjectives, adverbs, and pronouns.
    match = POS_MARKER_RE.search(raw)
    if match:
        return clean_meaning(raw[match.end():])

    # Verbs: locate the person tag and keep the complete remaining suffix.
    matches = list(PERSON_MARKER_RE.finditer(raw))
    if matches:
        return clean_meaning(raw[matches[-1].end():])

    # Compatibility fallback following the user's stated convention.
    parts = raw.split("-")
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
            "kiser", "to break or divert"
        ),
        "kisirx/T-ksr-PERF-3SM-to not break or not divert-N": (
            "kisirx", "to not break or not divert"
        ),
        "abitat/SINGNOUNM-built-up area": (
            "abitat", "built-up area"
        ),
        "ajkla bajda/SINGNOUNF-short-toed snake eagle": (
            "ajkla bajda", "short-toed snake eagle"
        ),
        "kiteb/F1-T-ktb-PERF-3SM-to write": (
            "kiteb", "to write"
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
        "meaning": "to not break or not divert",
    }

    print("dictionary_meanings.py self-test passed.")


if __name__ == "__main__":
    _self_test()
