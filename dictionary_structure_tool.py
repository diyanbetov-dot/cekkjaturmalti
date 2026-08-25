from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from flask import Blueprint, jsonify, request


VOWELS = frozenset({"a", "e", "i", "o", "u", "à", "è", "ì", "ò", "ù", "ie"})
SPECIAL_GRAPHEMES = ("għ", "ie")
ALT_ROUTE = re.compile(
    r"(?:^|,\s*)[A-Z]+:\s*([^,]+?)\s+-\s+(.+?)(?=(?:,\s*[A-Z]+:)|$)"
)
STRUCTURAL_TAG = re.compile(
    r"^(?:T|F\d+|M?PERF|IMP|PRES|PAST|FUT|\d[SPMF]|DO.*|IDO.*|N|"
    r"SING.*|PLU.*|COLL.*|PAUC.*|ADJ.*|ADVERB|PREP.*|PRON.*|ARTICLE.*)$",
    re.IGNORECASE,
)


def normalize(value: str) -> str:
    return unicodedata.normalize("NFC", value).casefold()


def graphemes(word: str) -> tuple[str, ...]:
    normalized = normalize(word)
    result: list[str] = []
    index = 0
    while index < len(normalized):
        matched = False
        for special in SPECIAL_GRAPHEMES:
            if normalized.startswith(special, index):
                result.append(special)
                index += len(special)
                matched = True
                break
        if not matched:
            result.append(normalized[index])
            index += 1
    return tuple(result)


def is_searchable_word(word: str) -> bool:
    return bool(word) and all(grapheme.isalpha() for grapheme in graphemes(word))


def meaning_from_tag(tag: str, filename: str) -> str:
    parts = [part.strip() for part in tag.split("-") if part.strip()]
    if not parts:
        return ""

    if parts[0] == "T":
        for index, part in enumerate(parts[2:], start=2):
            if part.casefold().startswith("to "):
                meaning_parts = parts[index:]
                if meaning_parts and meaning_parts[-1] == "N":
                    meaning_parts = meaning_parts[:-1]
                return "-".join(meaning_parts)

    for index, part in enumerate(parts[1:], start=1):
        if not STRUCTURAL_TAG.fullmatch(part) and (
            " " in part or any(character.islower() for character in part)
        ):
            return "-".join(parts[index:])

    if filename == "names.dic":
        return "Name"
    if filename == "places.dic":
        return "Place"
    return ""


def fallback_meaning(filename: str) -> str:
    labels = {
        "names.dic": "Name",
        "places.dic": "Place",
        "eu_countries.dic": "Country, place or demonym",
        "nopossessionnouns.dic": "Noun",
        "maltese_articles.dic": "Article",
        "prepositions.dic": "Preposition",
        "maltese_pronouns.dic": "Pronoun",
    }
    return labels.get(filename, "Meaning unavailable")


@dataclass(frozen=True, slots=True)
class StructureEntry:
    word: str
    units: tuple[str, ...]
    meanings: tuple[str, ...]
    sources: tuple[str, ...]


class DictionaryStructureIndex:
    def __init__(self, dics_root: Path) -> None:
        aggregated: dict[str, dict[str, object]] = {}
        for path in sorted(dics_root.rglob("*.dic")):
            relative_source = str(path.relative_to(dics_root)).replace("\\", "/")
            for word, meaning in self._entries_from_file(path):
                if not is_searchable_word(word):
                    continue
                key = normalize(word)
                record = aggregated.setdefault(
                    key,
                    {"word": word, "meanings": set(), "sources": set()},
                )
                if word.islower() and not str(record["word"]).islower():
                    record["word"] = word
                if meaning:
                    record["meanings"].add(meaning)  # type: ignore[union-attr]
                record["sources"].add(relative_source)  # type: ignore[union-attr]

        self.entries = tuple(
            StructureEntry(
                word=str(record["word"]),
                units=graphemes(str(record["word"])),
                meanings=tuple(sorted(record["meanings"])),  # type: ignore[arg-type]
                sources=tuple(sorted(record["sources"])),  # type: ignore[arg-type]
            )
            for record in aggregated.values()
        )

    @staticmethod
    def _entries_from_file(path: Path):
        for raw in path.read_text(encoding="utf-8-sig").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue

            if path.name == "alt.dic":
                for source, target in ALT_ROUTE.findall(line):
                    yield source.strip(), f"Correction route: {target.strip()}"
                continue

            if path.name in {"single.dic", "english.dic"}:
                if "-" in line:
                    source, target = line.split("-", 1)
                    label = "Bil-Malti" if path.name == "english.dic" else "Correction route"
                    yield source.strip(), f"{label}: {target.strip()}"
                continue

            if "/" in line:
                word, tag = line.split("/", 1)
                if tag.startswith("MALTI-VERB-"):
                    yield word.strip(), f"Bil-Malti: {tag[len('MALTI-VERB-'):].strip()}"
                elif tag.startswith("EN-MALTI-"):
                    yield word.strip(), f"Bil-Malti: {tag[len('EN-MALTI-'):].strip()}"
                else:
                    meaning = meaning_from_tag(tag, path.name) or fallback_meaning(path.name)
                    yield word.strip(), meaning
            else:
                yield line, fallback_meaning(path.name)

    @staticmethod
    def _slot_matches(slot: str, grapheme: str) -> bool:
        normalized_slot = normalize(slot).strip()
        if normalized_slot == "v":
            return grapheme in VOWELS
        if normalized_slot == "k":
            return grapheme.isalpha() and grapheme not in VOWELS
        return normalized_slot == grapheme

    def search(self, slots: list[str], limit: int = 500) -> tuple[list[dict[str, object]], int]:
        matches: list[StructureEntry] = []
        for entry in self.entries:
            if len(entry.units) != len(slots):
                continue
            if all(self._slot_matches(slot, unit) for slot, unit in zip(slots, entry.units)):
                matches.append(entry)
        matches.sort(key=lambda entry: (entry.word.casefold(), entry.word))
        total = len(matches)
        payload = [
            {
                "word": entry.word,
                "meaning": "; ".join(entry.meanings) or "Meaning unavailable",
                "sources": list(entry.sources),
            }
            for entry in matches[:limit]
        ]
        return payload, total


def create_dictionary_structure_blueprint(dics_root: Path) -> Blueprint:
    index = DictionaryStructureIndex(dics_root)
    blueprint = Blueprint("dictionary_structure", __name__)

    @blueprint.post("/dictionary-structure/search")
    def search_dictionary_structure():
        payload = request.get_json(silent=True) or {}
        slots = payload.get("slots")
        if not isinstance(slots, list) or not 1 <= len(slots) <= 32:
            return jsonify(error="Provide between 1 and 32 structure slots."), 400
        cleaned = [str(slot).strip() for slot in slots]
        if any(not slot for slot in cleaned):
            return jsonify(error="Every structure box must contain a value."), 400
        if any(len(graphemes(slot)) != 1 and normalize(slot) not in {"k", "v"} for slot in cleaned):
            return jsonify(error="Each box must contain one letter, għ, ie, K or V."), 400
        results, total = index.search(cleaned)
        return jsonify(results=results, total=total, shown=len(results), truncated=total > len(results))

    return blueprint
