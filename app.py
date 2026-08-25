from __future__ import annotations

import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from flask import Flask, jsonify, request, send_from_directory

from article_phrase_resolver import ArticlePhraseResolver
from context_families import ContextFamilyResolver
from corpus_ranker import CorpusCandidateRanker
from dictionary_structure_tool import create_dictionary_structure_blueprint
from morphology_agreement import MorphologyAgreementResolver
from numeral_agreement import AttributiveNumeralResolver
from suffix_adapter import HopeSuffixEngine
from suffixation.suffix_rules import ParsedSuffix
from verb_context import VerbContextResolver


ROOT = Path(__file__).resolve().parent
BASE_DICS = ROOT / "dics" / "basedics"
UI_DIR = ROOT / "User" / "Essentials"

WORD_PATTERN = re.compile(
    r"\d+(?::\d+)?(?:am|pm)?|[A-Za-zÀ-ſ]+(?:['’-][A-Za-zÀ-ſ]+)*(?:['’])?",
    re.IGNORECASE | re.UNICODE,
)
CHARACTER_PAIRS = {
    "c": "ċ",
    "ċ": "c",
    "g": "ġ",
    "ġ": "g",
    "h": "ħ",
    "ħ": "h",
    "z": "ż",
    "ż": "z",
    "a": "à",
    "à": "a",
    "e": "è",
    "è": "e",
    "i": "ì",
    "ì": "i",
    "o": "ò",
    "ò": "o",
    "u": "ù",
    "ù": "u",
}

MAPPED_SUBSTITUTION_COST = 0.2
ENABLE_SUFFIXATION = True
ENABLE_CORPUS_RANKING = True
CORPUS_MIN_ADVANTAGE = 0.1
ENABLE_MORPHOLOGY_AGREEMENT = True
ENABLE_ATTRIBUTIVE_NUMERAL_AGREEMENT = True
ENABLE_ARTICLE_PHRASE_RESOLUTION = True
ENABLE_VERB_CONTEXT_AGREEMENT = True
ENABLE_CONTEXT_FAMILIES = True

L_DIRECTIONAL_COMPLEMENTS = frozenset({"hawn", "hemm", "hinn"})


def normalize(value: str) -> str:
    return unicodedata.normalize("NFC", value).replace("’", "'").replace("‘", "'").replace("`", "'").casefold()


def _licensed_short_l_forms(following: str) -> tuple[str, ...]:
    """Read the short lil surfaces from the tagged preposition dictionary."""
    forms = ARTICLE_RESOLVER.short_l_forms
    if normalize(following) in L_DIRECTIONAL_COMPLEMENTS:
        return tuple(form for form in forms if form == "'l")
    return forms


def dictionary_surface(line: str) -> str | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    raw_surf = stripped.split("/", 1)[0].strip()
    if not raw_surf:
        return None
    return unicodedata.normalize("NFC", raw_surf).replace("’", "'").replace("‘", "'")



def load_hyphenated_prefix_keys(directory: Path) -> frozenset[str]:
    keys: set[str] = set()
    articles_path = directory / "articles.dic"
    if articles_path.is_file():
        for line in articles_path.read_text(encoding="utf-8-sig").splitlines():
            l = line.strip()
            if not l or l.startswith("#"):
                continue
            surface = l.split("/", 1)[0].strip()
            if surface.endswith("-"):
                keys.add(normalize(surface[:-1]))
    # Add common phonetic assimilation prefix variants (e.g. fl-ilma, mill-ilma, bl-ajruplan)
    keys.update({
        "fl", "mill", "bl", "bħall", "għall", "saell", "liell", "maell",
        # Article i-elision after a vowel: s-sala, d-dar, r-raġel, etc.
        "ċ", "d", "n", "r", "s", "t", "x", "z", "ż", "l",
    })
    return frozenset(keys)


APOSTROPHE_PREFIX_KEYS = frozenset({
    "b", "m", "d", "t", "f", "v", "n", "l", "x", "s", "ta", "ma", "sa", "bħal",
})

STANDALONE_PREFIX_KEYS = frozenset({
    "it", "il", "in", "is", "id", "ir", "iz", "iż", "ix", "iċ",
    "l", "ta", "fi", "bi", "ma", "sa", "dan", "din", "dal", "dil",
})

HYPHENATED_PREFIX_KEYS = load_hyphenated_prefix_keys(BASE_DICS)

PREPOSITION_COMPOUND_PAIRS: dict[str, tuple[str, ...]] = {
    "min": ("minn",),
    "minn": ("min",),
    "ma": ("ma'",),
    "ma'": ("ma",),
    "bħal": ("bħall-",),
    "bħall-": ("bħal",),
    "għal": ("għall-",),
    "għall-": ("għal",),
    "llejla": ("il-lejla",),
    "il-lejla": ("llejla",),
}

MANUAL_CONTEXT_ALTERNATIVES: dict[str, tuple[str, ...]] = {
    "kif": ("kief",),
    "kief": ("kif",),
}


def load_exact_english_map(path: Path) -> dict[str, str]:
    """Load exact English surfaces and their optional Maltese suggestion."""
    mapping: dict[str, str] = {}
    if not path.is_file():
        return mapping
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "-" not in stripped:
            continue
        source, target = (part.strip() for part in stripped.split("-", 1))
        if source:
            mapping[normalize(source)] = target
    return mapping


EXACT_ENGLISH = load_exact_english_map(ROOT / "dics" / "manualdics" / "english.dic")


def load_base_dictionary(directory: Path) -> tuple[dict[str, tuple[str, ...]], set[str]]:
    surfaces: dict[str, set[str]] = {}
    proper_only: set[str] = set()
    lexical: set[str] = set()

    for path in sorted(directory.glob("*.dic")):
        is_proper = path.name in ("names.dic", "places.dic")
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            surface = dictionary_surface(line)
            if surface is None:
                continue
            norm = normalize(surface)
            surfaces.setdefault(norm, set()).add(surface)
            if is_proper:
                proper_only.add(norm)
            else:
                lexical.add(norm)

    proper_only_surfaces = proper_only - lexical
    dictionary = {
        key: tuple(sorted(values, key=lambda item: (item.casefold(), item)))
        for key, values in surfaces.items()
    }
    return dictionary, proper_only_surfaces


def load_noun_keys(path: Path) -> frozenset[str]:
    keys: set[str] = set()
    if path.is_file():
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "/" not in stripped:
                continue
            surface, payload = stripped.split("/", 1)
            tag = payload.split("-", 1)[0].upper()
            if "NOUN" in tag:
                keys.add(normalize(surface.strip()))
    return frozenset(keys)


NOUN_DICTIONARY_KEYS = load_noun_keys(BASE_DICS / "fixednouns.dic")


@dataclass(frozen=True, slots=True)
class ManualEntry:
    targets: tuple[str, ...]
    family: str = ""


def load_manual_single_map(path: Path) -> dict[str, ManualEntry]:
    mapping: dict[str, ManualEntry] = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            l = line.strip()
            if not l or l.startswith("#") or "-" not in l:
                continue
            family = ""
            if ":" in l:
                prefix, rest = l.split(":", 1)
                if prefix.strip() and "-" in rest:
                    family = prefix.strip().upper()
                    l = rest.strip()
            parts = [part.strip() for part in l.split("-")]
            source = parts[0] if parts else ""
            targets = tuple(part for part in parts[1:] if part)
            if source and targets:
                mapping[normalize(source)] = ManualEntry(targets=targets, family=family)
    return mapping


def load_manual_alt_map(path: Path) -> dict[str, tuple[str, ...]]:
    mapping: dict[str, list[str]] = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            l = line.strip()
            if not l or l.startswith("#") or "-" not in l:
                continue
            entries = [part.strip() for part in l.split(",") if part.strip()]
            for entry in entries:
                if ":" in entry:
                    _label, entry = entry.split(":", 1)
                    entry = entry.strip()
                if "-" not in entry:
                    continue
                source, target = [part.strip() for part in entry.split("-", 1)]
                if not source or not target:
                    continue
                source_norm = normalize(source)
                targets = mapping.setdefault(source_norm, [])
                if normalize(target) != source_norm and all(normalize(existing) != normalize(target) for existing in targets):
                    targets.append(target)
        for source_norm, targets in list(mapping.items()):
            mapping[source_norm] = [
                target for target in targets if normalize(target) != source_norm
            ]
    return {
        source: tuple(targets)
        for source, targets in mapping.items()
        if targets
    }


AUTHORITY_RANKS = {
    "LOCKED": 0,
    "DETERMINISTIC": 1,
    "CONTEXT_RESOLVABLE": 2,
    "SUGGESTION_ONLY": 3,
}

EVIDENCE_MANUAL_SINGLE = 0
EVIDENCE_DIRECT_RULE = 1
EVIDENCE_EXACT_MORPHOLOGY = 2
EVIDENCE_MULTI_RULE = 3
EVIDENCE_FUZZY = 4


def get_evidence_rank(source: str, authority: str) -> int:
    if source == "manual_single" or authority == "LOCKED":
        return EVIDENCE_MANUAL_SINGLE
    if (
        authority == "DETERMINISTIC"
        or source.startswith("basedics_character_map")
        or source.startswith("basedics_h_to_hbar")
        or source.startswith("basedics_c_to_cdot")
        or source.startswith("basedics_g_to_gdot")
        or source.startswith("basedics_z_to_zdot")
        or source.startswith("basedics_initial_i_strip")
        or source.startswith("basedics_aj_ej_to_ghi")
        or source.startswith("basedics_u_w_consonant")
        or source.startswith("basedics_i_j_consonant")
        or source == "basedics_multi_diacritic"
    ):
        return EVIDENCE_DIRECT_RULE
    if source.startswith("basedics_suffix") or source.startswith("exact_morphology"):
        return EVIDENCE_EXACT_MORPHOLOGY
    if source.startswith("hybrid_"):
        return EVIDENCE_MULTI_RULE
    return EVIDENCE_FUZZY


@dataclass(frozen=True, slots=True)
class RankedCandidate:
    word: str
    distance: float
    mapped_changes: int
    source: str
    authority: str = "SUGGESTION_ONLY"

    @property
    def evidence_rank(self) -> int:
        return get_evidence_rank(self.source, self.authority)


class CharacterMapCorrector:
    def __init__(
        self,
        dictionary: dict[str, tuple[str, ...]],
        proper_only_surfaces: set[str] | None = None,
        manual_single_path: Path | None = None,
        manual_alt_path: Path | None = None,
    ) -> None:
        self.dictionary = dictionary
        self.proper_only_surfaces = proper_only_surfaces or set()
        self.manual_single: dict[str, ManualEntry] = (
            load_manual_single_map(manual_single_path)
            if manual_single_path is not None
            else load_manual_single_map(ROOT / "dics" / "manualdics" / "single.dic")
        )
        self.manual_alt: dict[str, tuple[str, ...]] = (
            load_manual_alt_map(manual_alt_path)
            if manual_alt_path is not None
            else load_manual_alt_map(ROOT / "dics" / "manualdics" / "alt.dic")
        )
        self.manual_suffix_stems: dict[str, tuple[str, ...]] = {
            source: tuple(
                target
                for target in entry.targets
                if entry.family == "GĦEDT" and normalize(target) == "għedt"
            )
            for source, entry in self.manual_single.items()
        }
        self.manual_suffix_stems = {
            source: targets
            for source, targets in self.manual_suffix_stems.items()
            if targets
        }
        skeletons: dict[str, list[tuple[str, int]]] = {}
        anchor_index: dict[str, list[str]] = {}
        for surfaces in dictionary.values():
            for surface in surfaces:
                skeleton = self._dictionary_skeleton(surface)
                if skeleton is not None:
                    key, structural_changes = skeleton
                    skeletons.setdefault(key, []).append((surface, structural_changes))
                akey = self._anchor_skeletal_key(surface)
                if akey:
                    anchor_index.setdefault(akey, []).append(surface)

        self.skeleton_index = {
            key: tuple(values) for key, values in skeletons.items()
        }
        self.anchor_skeletal_index = anchor_index
        self.suffix_engine: HopeSuffixEngine | None = None

    def _multi_diacritic_variants(self, word: str, max_changes: int = 2) -> list[tuple[str, int]]:
        """Return combination diacritic substitutions up to max_changes for valid dictionary words."""
        norm = normalize(word)
        if len(norm) > 15:
            return []

        diacritic_map = {
            "c": "ċ", "g": "ġ", "h": "ħ", "z": "ż",
            "a": "à", "e": "è", "i": "ì", "o": "ò", "u": "ù",
        }

        indices = [i for i, ch in enumerate(norm) if ch in diacritic_map]
        if not indices:
            return []

        results: list[tuple[str, int]] = []
        seen: set[str] = {norm}

        queue: list[tuple[str, int, int]] = [(norm, 0, 0)]
        while queue:
            curr, changes, last_idx = queue.pop(0)
            if changes >= max_changes:
                continue
            for idx in indices:
                if idx < last_idx and changes > 0:
                    continue
                ch = curr[idx]
                if ch in diacritic_map:
                    sub = diacritic_map[ch]
                    variant = curr[:idx] + sub + curr[idx + 1 :]
                    if variant not in seen:
                        seen.add(variant)
                        new_changes = changes + 1
                        if self.is_known(variant):
                            results.append((variant, new_changes))
                        queue.append((variant, new_changes, idx + 1))
        return results



    @staticmethod
    def _anchor_skeletal_key(word: str) -> str:
        norm = normalize(word)
        stripped = norm.replace("'", "").replace("’", "")
        stripped = (
            stripped.replace("ċ", "c")
            .replace("ġ", "g")
            .replace("ż", "z")
            .replace("à", "a")
            .replace("è", "e")
            .replace("ì", "i")
            .replace("ò", "o")
            .replace("ù", "u")
        )
        if stripped.startswith("gh"):
            stripped = stripped[2:]
        elif stripped.startswith("agh"):
            stripped = stripped[3:]
        stripped = stripped.replace("gh", "").replace("h", "")
        collapsed: list[str] = []
        for ch in stripped:
            if collapsed and ch == collapsed[-1] and ch not in "aeiou":
                continue
            collapsed.append(ch)
        return "".join(collapsed)


    @staticmethod
    def _dictionary_skeleton(word: str) -> tuple[str, int] | None:
        source = normalize(word)
        stripped: list[str] = []
        removed = 0
        index = 0
        while index < len(source):
            if source.startswith("għ", index):
                removed += 1
                index += 2
                continue
            if source[index] in {"h", "'"}:
                removed += 1
                index += 1
                continue
            stripped.append(source[index])
            index += 1
        if removed == 0:
            return None

        collapsed: list[str] = []
        vowel_singulations = 0
        vowels = frozenset("aeiouàèìòù")
        for character in stripped:
            if collapsed and character == collapsed[-1] and character in vowels:
                vowel_singulations += 1
                continue
            collapsed.append(character)
        return "".join(collapsed), removed + vowel_singulations

    @staticmethod
    def _input_skeleton(word: str) -> str | None:
        """Strip typed silent-letter forms for a second skeleton lookup."""
        source = normalize(word)
        stripped: list[str] = []
        removed = 0
        index = 0
        while index < len(source):
            if source.startswith("gh", index):
                removed += 1
                index += 2
                continue
            if source[index] in {"h", "'"}:
                removed += 1
                index += 1
                continue
            stripped.append(source[index])
            index += 1
        if removed == 0:
            return None

        collapsed: list[str] = []
        vowels = frozenset("aeiouàèìòù")
        for character in stripped:
            if collapsed and character == collapsed[-1] and character in vowels:
                continue
            collapsed.append(character)
        return "".join(collapsed)

    def _proper_name_penalty(self, source_word: str, candidate_word: str) -> float:
        if normalize(candidate_word) in self.proper_only_surfaces:
            # Input is all-lowercase: user simply forgot to capitalise — no penalty
            if source_word.islower():
                return 0.0
            # Explicit proper-noun capitalization in a longer word is preserved without penalty
            if source_word[:1].isupper() and candidate_word[:1].isupper() and len(source_word) >= 4:
                return 0.0
            return 1.5
        return 0.0


    def _is_possessive_noun(self, word: str) -> bool:
        norm = normalize(word)
        suffixes = (
            "hulu", "hielna", "hielha", "hielkom", "hielhom",
            "lu", "lha", "lhom", "li", "lek", "lna", "lkom",
            "u", "i", "a", "ha", "hom", "na", "kom", "ek", "ik", "ekh", "ok", "ak", "at", "h", "x",
        )
        for j_sfx in ("jh", "jha", "jhom", "jna", "jkom", "jk"):
            if norm.endswith(j_sfx) and len(norm) > len(j_sfx):
                stem = norm[: -len(j_sfx)]
                if stem in self.dictionary or (stem + "a") in self.dictionary:
                    return True
        for sfx in suffixes:
            if norm.endswith(sfx) and len(norm) > len(sfx):
                stem = norm[:-len(sfx)]
                if stem in STANDALONE_PREFIX_KEYS or stem in ("il", "in", "it", "bi", "fi", "ta", "sa", "ma", "dan", "din", "li"):
                    continue

                if len(stem) < 2 and sfx in ("x", "h", "u", "i", "a"):
                    continue
                stem_variants = [stem]
                # Possessive suffixation can suppress the final unstressed
                # vowel of a noun: sieħeb + -i -> sieħbi.
                if len(stem) >= 3 and stem[-1] not in "aeiouàèìòù":
                    stem_variants.append(stem[:-1] + "e" + stem[-1])
                matched_stem = next(
                    (candidate for candidate in stem_variants if candidate in NOUN_DICTIONARY_KEYS),
                    None,
                )
                if matched_stem is not None:
                    # Check if stem+sfx is actually an un-diacriticized form of a known Maltese word (e.g. bongu -> bonġu, anda -> għandha, beza -> beża', erga -> erġa')
                    cmap_target = norm.replace("c", "ċ").replace("g", "ġ").replace("h", "ħ").replace("z", "ż")
                    if (
                        (cmap_target != norm and cmap_target in self.dictionary)
                        or (cmap_target + "'") in self.dictionary
                        or (cmap_target[:-1] + "a'") in self.dictionary
                        or ("għ" + norm) in self.dictionary
                        or ("agħ" + norm[1:] if norm.startswith("a") else "") in self.dictionary
                    ):
                        continue
                    return True
        return False


    def is_known(self, word: str) -> bool:
        surface = unicodedata.normalize("NFC", word)
        norm = normalize(surface)

        # Numbers and clock expressions are structural tokens, never lexical
        # material for Maltese morphology (9am must not become 9għam).
        if re.fullmatch(r"\d+(?::\d+)?(?:am|pm)?", surface, re.IGNORECASE):
            return True

        # Preserve conspicuous all-caps sound effects (BEEEEEP, BANGGG). They
        # are expressive text, not lexical correction targets.
        if surface.isupper() and len(surface) >= 4 and re.search(r"([A-Z])\1{2,}", surface):
            return True

        # 1. Standalone article / preposition prefix keys (e.g. It, il, in, ta, fi)
        if norm in STANDALONE_PREFIX_KEYS:
            return True

        dictionary_surfaces = self.dictionary.get(norm, ())
        if surface in dictionary_surfaces:
            return True
        if surface.isupper() or (surface[:1].isupper() and surface[1:].islower()):
            if norm in self.dictionary:
                return True

        # Diacritic / Guttural override guard:
        # If an un-diacriticized word (e.g. cara, bongu, anda, andi, andu, centru) has a standard Maltese
        # diacritic or guttural counterpart in self.dictionary, force is_known to False so candidate generation converts it!
        if surface.islower() and not any(ch in norm for ch in "ċġħżàèìòù"):
            if norm in ("anda", "andi", "andu", "anna"):
                return False
            mapped_counterparts = [
                norm.replace("c", "ċ").replace("g", "ġ").replace("h", "ħ").replace("z", "ż"),
                "għ" + norm,
                "agħ" + norm[1:] if norm.startswith("a") else norm,
                "għa" + norm[1:] if norm.startswith("a") else norm,
            ]
            for counterpart in mapped_counterparts:
                if counterpart != norm and counterpart in self.dictionary and not all(s[:1].isupper() for s in self.dictionary[counterpart]):
                    return False


        # 2. Regular dictionary lookup
        if surface.islower():
            if dictionary_surfaces and all(s[:1].isupper() for s in dictionary_surfaces):
                if norm in ("ilom", "anda", "andi", "andu", "anna"):
                    return False
                cmap_form = norm.replace("h", "ħ").replace("c", "ċ").replace("g", "ġ").replace("z", "ż")
                if cmap_form != norm and cmap_form in self.dictionary:
                    return False
                if ("għ" + norm) in self.dictionary or ("agħ" + norm[1:] if norm.startswith("a") else "") in self.dictionary:
                    return False





        # 3. Hyphenated preposition-article compound (e.g. fl-ilma, mill-ilma, tas-sajf)
        if "-" in surface:
            parts = surface.split("-", 1)
            if len(parts) == 2 and normalize(parts[0]) in HYPHENATED_PREFIX_KEYS:
                if self.is_known(parts[1]):
                    return True

        # 4. Possessive noun (e.g. ġismu, wiċċek, oħtha, ommhom, idejk)
        if self._is_possessive_noun(word):
            # If the unaccented word itself is not in the dictionary, but its accented counterpart
            # (e.g. dinjità, attività, fakultà) IS in the dictionary, prefer correction over possessive match.
            vowel_accents = {"a": "à", "e": "è", "i": "ì", "o": "ò", "u": "ù"}
            stem_norm = norm.rstrip("'")
            if stem_norm and stem_norm[-1] in vowel_accents:
                accented = stem_norm[:-1] + vowel_accents[stem_norm[-1]]
                if accented in self.dictionary:
                    return False
            return True

        # 5. Exact generated suffix form or a reversible surface variant of it
        # (for example i-/i-ie spelling around an otherwise exact suffix form).
        if self.suffix_engine is not None and (
            self.suffix_engine.exact(word)
            or self.suffix_engine.exact_surface_variant(word)
        ):
            return True

        # 6. Apostrophe-prefixed preposition compound (e.g. b'idejk, m'ibni, m'għandekx, b'idea, f'daqqa, f'ġieħ)
        for delim in ("'", "’"):
            if delim in surface:
                parts = surface.split(delim, 1)
                if len(parts) == 2 and normalize(parts[0]) in APOSTROPHE_PREFIX_KEYS:
                    if self.is_known(parts[1]):
                        return True

        return False

    def _is_prefix_complement_known(self, word: str) -> bool:
        norm = normalize(word)
        if norm in self.dictionary or (norm.endswith("x") and norm[:-1] in self.dictionary):
            return True
        if self._is_possessive_noun(word):
            return True
        return bool(
            self.suffix_engine is not None
            and (
                self.suffix_engine.exact(word)
                or self.suffix_engine.exact_surface_variant(word)
            )
        )

    @staticmethod
    def _case_rank(source: str, candidate: str) -> int:
        if source.islower():
            return 0 if candidate.islower() else 1
        if source[:1].isupper() and source[1:].islower():
            return 0 if candidate[:1].isupper() and candidate[1:].islower() else 1
        if source.isupper():
            return 0 if candidate.isupper() else 1
        return 0

    @staticmethod
    def _mapped_variants(word: str) -> Iterable[tuple[str, int]]:
        characters = list(normalize(word))
        positions = [index for index, char in enumerate(characters) if char in CHARACTER_PAIRS]
        if not positions:
            return

        # Every emitted variant changes at least one mapped character. No other
        # insertion, deletion, substitution, split or join is permitted here.
        for mask in range(1, 1 << len(positions)):
            candidate = characters.copy()
            changes = 0
            for bit, position in enumerate(positions):
                if mask & (1 << bit):
                    candidate[position] = CHARACTER_PAIRS[candidate[position]]
                    changes += 1
            var_str = "".join(candidate)
            yield var_str, changes
            if var_str and var_str[-1] in "aeiou" and not var_str.endswith("'"):
                yield var_str + "'", changes + 1

    def _suffix_validated_s_z_variants(self, word: str) -> Iterable[tuple[str, int]]:
        """Recover s/z spellings only when they form a real suffixed verb."""
        if self.suffix_engine is None:
            return
        norm = normalize(word)
        positions = [index for index, char in enumerate(norm) if char in {"s", "z"}]
        if not positions or len(positions) > 5:
            return
        options = {"s": ("s", "z", "ż"), "z": ("z", "ż")}
        seen: set[str] = {norm}

        def visit(position_index: int, characters: list[str], changes: int) -> Iterable[tuple[str, int]]:
            if position_index == len(positions):
                candidate = "".join(characters)
                if candidate not in seen and self.suffix_engine.exact(candidate):
                    seen.add(candidate)
                    yield candidate, changes
                return
            position = positions[position_index]
            original = norm[position]
            for replacement in options[original]:
                characters[position] = replacement
                yield from visit(
                    position_index + 1,
                    characters,
                    changes + int(replacement != original),
                )
            characters[position] = original

        yield from visit(0, list(norm), 0)

    def _gh_root_imperfect_variants(self, word: str) -> Iterable[str]:
        """Recover common imperfect spellings of verbs whose root contains għ.

        This composes typed ``gh`` (or one omitted għ) with a single vowel
        correction, while requiring a real MPERF record with the same
        consonantal frame. Examples include jixghal/jixal -> jixgħel and the
        corresponding t-/n- forms.
        """
        if self.suffix_engine is None:
            return
        normalized = normalize(word)
        if len(normalized) < 4 or normalized[:1] not in {"j", "t", "n"}:
            return
        index = self.suffix_engine.generator.verb_index
        canonical = normalized.replace("gh", "għ")
        anchor = index.consonant_anchor(canonical)
        anchor_hypotheses = {anchor}
        if "għ" not in canonical:
            for position in range(1, len(anchor)):
                anchor_hypotheses.add(anchor[:position] + "għ" + anchor[position:])
        seen: set[str] = set()
        # weighted_distance counts għ as two Unicode code points. Missing-għ
        # hypotheses therefore need room for those two insertions plus one
        # vowel correction, while all candidates remain verb-paradigm gated.
        distance_limit = 1.25 if "gh" in normalized else 3.25
        for hypothesis in anchor_hypotheses:
            for record in index.by_anchor.get(hypothesis, ()):
                candidate = normalize(record.word)
                if (
                    not record.is_mperf
                    or not record.is_f1
                    or index.second_radical(record) != "għ"
                    or "għ" not in candidate
                    or candidate[:1] != normalized[:1]
                    or candidate in seen
                    or self.weighted_distance(canonical, candidate) > distance_limit
                ):
                    continue
                seen.add(candidate)
                yield candidate


    def _single_missing_consonant_variants(self, word: str) -> Iterable[str]:
        """Yield exact dictionary words formed by one consonant insertion.

        These are contextual candidates only. They do not mechanically outrank
        a closer silent-letter or diacritic repair.
        """
        norm = normalize(word)
        consonants = (
            "b", "ċ", "d", "f", "ġ", "g", "għ", "h", "ħ", "j", "k",
            "l", "m", "n", "p", "q", "r", "s", "t", "v", "w", "x", "ż", "z",
        )
        seen: set[str] = set()
        for position in range(1, len(norm)):
            for consonant in consonants:
                candidate = norm[:position] + consonant + norm[position:]
                if (
                    candidate in seen
                    or candidate not in self.dictionary
                    or candidate in self.proper_only_surfaces
                ):
                    continue
                seen.add(candidate)
                yield candidate


    def _ordinary_structural_variants(self, word: str) -> Iterable[tuple[str, str]]:
        """Emit one broad i/ie or consonant single/double repair, plus
        several Maltese-specific phonological corrections."""
        normalized = normalize(word)
        seen: set[str] = set()

        def _emit(candidate: str, source: str) -> Iterable[tuple[str, str]]:
            if candidate and candidate != normalized and candidate not in seen:
                seen.add(candidate)
                yield candidate, source

        index = 0
        while index < len(normalized):
            if normalized.startswith("ie", index):
                candidate = normalized[:index] + "i" + normalized[index + 2:]
                yield from _emit(candidate, "basedics_i_ie")
                index += 2
                continue
            if normalized[index] == "i":
                candidate = normalized[:index] + "ie" + normalized[index + 1:]
                yield from _emit(candidate, "basedics_i_ie")
            index += 1

        graphemes = (
            self.suffix_engine.adapter._graphemes(normalized)
            if self.suffix_engine is not None
            else list(normalized)
        )
        vowels = frozenset("aeiouàèìòù")
        consonants = {
            token for token in graphemes
            if len(token) == 1 and token.isalpha() and token not in vowels
        } | {"għ"}

        # u/w and i/j can be epenthetic or omitted before a consonant:
        # buwt -> but, mibrum -> mibruwm, bijt -> bit, bik -> bijk
        # The generated forms are dictionary/suffix gated by the caller.
        for index in range(len(graphemes) - 1):
            current = graphemes[index]
            following = graphemes[index + 1]
            if current == "u" and following in consonants:
                candidate = "".join(graphemes[: index + 1] + ["w"] + graphemes[index + 1:])
                yield from _emit(candidate, "basedics_u_w_consonant_insert")
            if current == "i" and following in consonants:
                candidate = "".join(graphemes[: index + 1] + ["j"] + graphemes[index + 1:])
                yield from _emit(candidate, "basedics_i_j_consonant_insert")

        for index in range(len(graphemes) - 2):
            current = graphemes[index]
            glide = graphemes[index + 1]
            following = graphemes[index + 2]
            if current == "u" and glide == "w" and following in consonants:
                candidate = "".join(graphemes[: index + 1] + graphemes[index + 2:])
                yield from _emit(candidate, "basedics_u_w_consonant_delete")
            if current == "i" and glide == "j" and following in consonants:
                candidate = "".join(graphemes[: index + 1] + graphemes[index + 2:])
                yield from _emit(candidate, "basedics_i_j_consonant_delete")

        for index, token in enumerate(graphemes):
            if token in vowels or not token.isalpha():
                continue
            previous_same = index > 0 and graphemes[index - 1] == token
            next_same = index + 1 < len(graphemes) and graphemes[index + 1] == token
            if not previous_same and not next_same:
                candidate = "".join(
                    graphemes[: index + 1] + [token] + graphemes[index + 1:]
                )
                yield from _emit(candidate, "basedics_single_double")

        # Adjacent transposition is a common mechanical typo. It remains
        # dictionary-gated and can compose with a Maltese character mapping
        # later (tishti -> tisthi -> tistħi).
        for index in range(len(graphemes) - 1):
            if graphemes[index] == graphemes[index + 1]:
                continue
            candidate = "".join(
                graphemes[:index]
                + [graphemes[index + 1], graphemes[index]]
                + graphemes[index + 2:]
            )
            yield from _emit(candidate, "basedics_adjacent_transposition")

        # The established e- epenthesis exception is the rġ- verb family
        # (rġajt / erġajt). The dictionary form remains rġ-; sentence
        # phonology chooses the visible e- form later.
        if normalized.startswith(("erġ", "erg")) and len(normalized) > 3:
            yield from _emit(normalized[1:], "basedics_rgj_e_epenthesis_strip")

        # Recover a 3P verb plus DO-3SM surface when the typed form keeps -u-
        # instead of the base's -ew-: jistennuh -> jistennewh. The candidate
        # is accepted only when its unsuffixed -w base is a real verb.
        if normalized.endswith("uh") and len(normalized) > 4:
            yield from _emit(normalized[:-2] + "ewh", "basedics_3p_u_h_to_ew_h")

        # Pronoun-family spelling in which typed gh + jaw represents the
        # standard tiegħu surface. This is a morphological family repair, not
        # an unrestricted fuzzy substitution.
        if normalized.startswith("tieghj") and normalized.endswith("jaw"):
            yield from _emit("tiegħu", "basedics_tiegh_pronoun_surface")

        for index in range(len(graphemes) - 1):
            token = graphemes[index]
            if token == graphemes[index + 1] and token not in vowels and token.isalpha():
                candidate = "".join(graphemes[:index] + graphemes[index + 1:])
                yield from _emit(candidate, "basedics_single_double")

        # aj/ej can reflect a missing għ + front-vowel sequence in għ-root forms:
        # ajd -> għid, jajdlek -> jgħidlek, ejni -> għinni, ejjejt -> għejjejt/għejejt.
        for idx in range(len(graphemes) - 1):
            pair = "".join(graphemes[idx : idx + 2])
            if pair not in {"aj", "ej"}:
                continue
            replacements = ("għi", "għej") if pair == "ej" else ("għi", "għaj")
            for replacement in replacements:
                repl_graphemes = (
                    self.suffix_engine.adapter._graphemes(replacement)
                    if self.suffix_engine is not None
                    else list(replacement)
                )
                candidate_graphemes = graphemes[:idx] + repl_graphemes + graphemes[idx + 2:]
                candidate = "".join(candidate_graphemes)
                yield from _emit(candidate, "basedics_aj_ej_to_ghi")
                if idx + 2 < len(graphemes):
                    next_token = graphemes[idx + 2]
                    if (
                        len(next_token) == 1
                        and next_token.isalpha()
                        and next_token not in vowels
                    ):
                        doubled = "".join(
                            candidate_graphemes[: idx + len(repl_graphemes) + 1]
                            + [next_token]
                            + candidate_graphemes[idx + len(repl_graphemes) + 1:]
                        )
                        yield from _emit(doubled, "basedics_aj_ej_to_ghi_double")
                for j_idx in range(len(candidate_graphemes) - 1):
                    token = candidate_graphemes[j_idx]
                    if token == candidate_graphemes[j_idx + 1] and token == "j":
                        single_j = "".join(candidate_graphemes[:j_idx] + candidate_graphemes[j_idx + 1:])
                        yield from _emit(single_j, "basedics_aj_ej_to_ghi_single_j")

        # Rule 1 – Final h/ħ → għ
        # dullieh → dulliegħ, dullieħ → dulliegħ
        if graphemes and graphemes[-1] in ("h", "ħ"):
            candidate = "".join(graphemes[:-1]) + "għ"
            yield from _emit(candidate, "basedics_final_h_to_gh")

        # Rule 2 – ijV → iegħV (typed ij before a vowel → iegħ)
        # bqija → bqiegħa, dullija → dulliegħa, qijed → qiegħed
        for idx in range(len(graphemes) - 2):
            if (
                graphemes[idx] == "i"
                and graphemes[idx + 1] == "j"
                and graphemes[idx + 2] in vowels
            ):
                candidate = (
                    "".join(graphemes[:idx])
                    + "iegħ"
                    + "".join(graphemes[idx + 2:])
                )
                yield from _emit(candidate, "basedics_ij_vowel_to_iegh")

        # Rule 3 – -aw/-ew/-jew endings → -għu
        # tijew → tiegħu  (strip jew → egħu: j is liaison consonant, ew→egħu)
        # baqaw → baqgħu  (strip aw → għu)
        if normalized.endswith("jew") and len(normalized) > 3:
            # -jew: drop j, convert ew → egħu (e.g. tijew → ti+egħu = tiegħu)
            candidate = normalized[:-3] + "egħu"
            yield from _emit(candidate, "basedics_jew_to_ghu")
        if normalized.endswith("ew") and len(normalized) > 2:
            candidate = normalized[:-2] + "egħu"
            yield from _emit(candidate, "basedics_ew_to_eghu")
        if normalized.endswith("aw") and len(normalized) > 2:
            candidate = normalized[:-2] + "għu"
            yield from _emit(candidate, "basedics_aw_to_ghu")

        # Rule 4 – Final apostrophe → għ, drop, or convert final unaccented vowel to grave accent
        # bela' → belgħ, bela' → bela, kafe' → kafè
        if graphemes and graphemes[-1] == "'":
            stem = "".join(graphemes[:-1])
            yield from _emit(stem + "għ", "basedics_final_apostrophe_to_gh")
            yield from _emit(stem, "basedics_final_apostrophe_drop")
            vowel_accents = {"a": "à", "e": "è", "i": "ì", "o": "ò", "u": "ù"}
            if stem and stem[-1] in vowel_accents and len(stem) >= 2:
                yield from _emit(stem[:-1] + vowel_accents[stem[-1]], "basedics_final_apostrophe_to_accent")


        # Rule 5 is handled directly in correct_word via suffix engine exact-match
        # (see HopeSuffixEngine.ha_suffix_candidates), not here, because the base
        # dictionary would rank a closer structural match above the +ha form.

        # Rule 6 – -ħa / -hha / -ħħa → -ha suffix correction
        # aqraħħa → aqraha, aqrahha → aqraha, aqraħa → aqraha
        for bad_ending, source in [
            ("ħħa", "basedics_hha_to_ha"),
            ("hha", "basedics_hha_to_ha"),
            ("ħa", "basedics_ha_to_ha"),
        ]:
            if normalized.endswith(bad_ending) and len(normalized) > len(bad_ending):
                candidate = normalized[: -len(bad_ending)] + "ha"
                yield from _emit(candidate, source)
                break

        # A written double ħ can represent the standard għ+h sequence. Keep
        # this reversible and dictionary-gated (maħħa <-> magħha), because the
        # written form may itself be a valid word and therefore needs context.
        for index in range(len(graphemes) - 1):
            if graphemes[index : index + 2] != ["ħ", "ħ"]:
                continue
            candidate = "".join(graphemes[:index] + ["għ", "h"] + graphemes[index + 2:])
            yield from _emit(candidate, "basedics_hh_to_gh_h")

        # Initial i- stripping (e.g. irnexilu -> rnexxielu / rnexilu)
        if normalized.startswith("i") and len(normalized) > 4:
            stripped = normalized[1:]
            yield from _emit(stripped, "basedics_initial_i_strip")
            if self.suffix_engine is not None:
                for ie_v in self.suffix_engine.adapter.i_ie_variants(stripped):
                    yield from _emit(ie_v, "basedics_initial_i_strip_i_ie")
                # A typed epenthetic i- may coexist with one missing doubled
                # consonant and one i/ie alternation (irnexilu -> rnexxielu).
                # Keep this composition bounded to those two established
                # structural families and let dictionary/suffix validation
                # decide whether the resulting surface is real.
                for doubled in self.suffix_engine.adapter.single_double_variants(stripped):
                    for ie_v in self.suffix_engine.adapter.i_ie_variants(doubled):
                        yield from _emit(ie_v, "basedics_initial_i_strip_double_i_ie")
                for ie_v in self.suffix_engine.adapter.i_ie_variants(stripped):
                    for doubled in self.suffix_engine.adapter.single_double_variants(ie_v):
                        yield from _emit(doubled, "basedics_initial_i_strip_i_ie_double")

        # Missing initial għ (e.g. ax -> għax)
        if not normalized.startswith("għ"):
            yield from _emit("għ" + normalized, "basedics_missing_gh")

        # ------------------------------------------------------------------
        # Cat A – global h→ħ replacement (any position)
        # e.g. hireg→ħiereġ, hwinet→ħwienet, hajti→ħajti
        # Multi-diacritic combination search (e.g. diga -> diġà, certa -> ċerta)
        for variant, changes_count in self._multi_diacritic_variants(normalized, max_changes=2):
            yield from _emit(variant, "basedics_multi_diacritic")

        # Cat A, B & G – global g->ġ, c->ċ, h->ħ replacement

        if "g" in normalized:
            yield from _emit(normalized.replace("g", "ġ"), "basedics_g_to_gdot")
        if "h" in normalized and "għ" not in normalized:
            yield from _emit(normalized.replace("h", "ħ"), "basedics_h_to_hbar")
        if "c" in normalized:
            yield from _emit(normalized.replace("cc", "ċċ").replace("c", "ċ"), "basedics_c_to_cdot")

        # Drop silent/extra h when għ is present (e.g. ghadhu -> għadu)
        if "h" in normalized and ("għ" in normalized or "gh" in normalized):
            clean_h = normalized.replace("gh", "għ").replace("h", "")
            if clean_h != normalized:
                yield from _emit(clean_h, "basedics_drop_extra_h")

        # Cat D & G – initial i- stripping with character map (e.g. iccempel -> ċċempel, iggib -> ġġib)
        if normalized.startswith("i") and len(normalized) > 3:
            stripped = normalized[1:]
            yield from _emit(stripped, "basedics_initial_i_strip")
            cmap_stripped = (
                stripped.replace("cc", "ċċ")
                .replace("c", "ċ")
                .replace("gg", "ġġ")
                .replace("g", "ġ")
                .replace("zz", "żż")
                .replace("z", "ż")
                .replace("hh", "ħħ")
                .replace("h", "ħ")
            )
            if cmap_stripped != stripped:
                yield from _emit(cmap_stripped, "basedics_initial_i_strip_cmap")

        # Cat C & A – Guttural / weak root single-vowel shifts (e.g. ghidt -> għedt, ghendek -> għandek, hireg -> ħiereġ)
        if "għ" in normalized or "ħ" in normalized or "gh" in normalized or "h" in normalized:
            gh_norm = (
                normalized.replace("gh", "għ")
                .replace("cc", "ċċ")
                .replace("c", "ċ")
                .replace("h", "ħ")
            )
            # A common front-vowel spelling expands għi as għej/ghej. Collapse
            # that sequence before suffix lookup: tgħejdlix -> tgħidlix.
            if "għej" in gh_norm:
                yield from _emit(
                    gh_norm.replace("għej", "għi"),
                    "basedics_ghej_to_ghi",
                )
            for idx, char in enumerate(gh_norm):
                if char == "i":
                    yield from _emit(gh_norm[:idx] + "e" + gh_norm[idx + 1:], "basedics_guttural_vowel_i_e")
                    yield from _emit(gh_norm[:idx] + "ie" + gh_norm[idx + 1:], "basedics_guttural_vowel_i_ie")
                elif char == "e":
                    yield from _emit(gh_norm[:idx] + "a" + gh_norm[idx + 1:], "basedics_guttural_vowel_e_a")
                    yield from _emit(gh_norm[:idx] + "i" + gh_norm[idx + 1:], "basedics_guttural_vowel_e_i")

        # General final vowel accent recovery for a, e, i, o, u -> à, è, ì, ò, ù
        # (e.g. attivita -> attività, kafe / kafe' -> kafè, pero -> però, Gesu -> Ġesù)
        vowel_accents = {"a": "à", "e": "è", "i": "ì", "o": "ò", "u": "ù"}
        stem_norm = normalized.rstrip("'")
        if stem_norm and stem_norm[-1] in vowel_accents and len(stem_norm) >= 2:
            accented = stem_norm[:-1] + vowel_accents[stem_norm[-1]]
            yield from _emit(accented, "basedics_final_vowel_accent")
            g_accented = accented.replace("g", "ġ")
            if g_accented != accented:
                yield from _emit(g_accented, "basedics_final_vowel_accent_gdot")



        # Cat E – Loanword -icu -> -iku ending (e.g. automaticu -> awtomatiku, electroniku -> elettroniku)
        if normalized.endswith("icu") and len(normalized) > 4:
            yield from _emit(normalized[:-3] + "iku", "basedics_icu_to_iku")

        # Initial a- -> agħ- / għa- prefix recovery (e.g. ala -> għala, ax -> għax, amilt -> għamilt, atiha -> agħtiha, anda -> għandha)
        if normalized.startswith("a") and len(normalized) >= 2:
            yield from _emit("agħ" + normalized[1:], "basedics_initial_a_to_agh")
            yield from _emit("għ" + normalized, "basedics_initial_a_to_gha")

        # Fused article/preposition recovery before apostrophe parsing:
        # lahhar -> l-aħħar, flistess -> fl-istess, blistess -> bl-istess.
        fused_prefixes = {
            "fl": "fl-",
            "bl": "bl-",
        }
        for prefix_key in sorted((*HYPHENATED_PREFIX_KEYS, *fused_prefixes), key=len, reverse=True):
            if len(prefix_key) < 1 or not normalized.startswith(prefix_key):
                continue
            rest_stem = normalized[len(prefix_key):]
            if len(rest_stem) < 2:
                continue
            output_prefix = fused_prefixes.get(prefix_key, prefix_key + "-")
            rest_variants = [rest_stem] + [
                candidate for candidate, _source in self._ordinary_structural_variants(rest_stem)
            ]
            for rest_cand in rest_variants:
                if self.is_known(rest_cand):
                    surfaces = self.dictionary.get(normalize(rest_cand), ())
                    chosen_rest = surfaces[0] if surfaces else rest_cand
                    if (
                        output_prefix in {"fl-", "bl-"}
                        and rest_stem.startswith("i")
                        and rest_cand == rest_stem[1:]
                    ):
                        chosen_rest = rest_stem
                    yield from _emit(
                        f"{output_prefix}{chosen_rest}",
                        "basedics_fused_article_prefix",
                    )
                    break

        # Preposition apostrophe prefix recovery (e.g. fwiehed -> f'wieħed, bidejk -> b'idejk)
        if len(normalized) >= 4 and normalized[0] in ("f", "b", "m", "t", "l", "x"):
            p_prefix = normalized[0]
            rest_stem = normalized[1:]
            rest_candidates = [(rest_stem, "original_prefix_rest")]
            rest_candidates.extend(self._ordinary_structural_variants(rest_stem))
            for rest_cand, rest_source in rest_candidates:
                if self._is_prefix_complement_known(rest_cand):
                    # Bare t' is the elided form of ta' and therefore selects
                    # nominal complements. It must not manufacture t' + VERB
                    # alternatives such as t'Għid or t'idħak.
                    rest_tags = ARTICLE_RESOLVER.tags.get(normalize(rest_cand), set())
                    if p_prefix == "t" and not any(
                        marker in tag
                        for tag in rest_tags
                        for marker in ("NOUN", "ADJ", "NAME", "SNAME", "PLACE", "PRON")
                    ):
                        continue
                    if (
                        self.weighted_distance(rest_stem, rest_cand) > 1.0
                        and rest_source not in {
                            "basedics_missing_gh",
                            "basedics_missing_h",
                            "basedics_missing_gh_h_skeleton",
                        }
                    ):
                        continue
                    surfaces = self.dictionary.get(normalize(rest_cand), ())
                    chosen_rest = surfaces[0] if surfaces else rest_cand
                    if "'" in chosen_rest or "’" in chosen_rest:
                        continue
                    yield from _emit(f"{p_prefix}'{chosen_rest}", "basedics_preposition_apostrophe_prefix")

        # Guttural għ + h pronouns (e.g. anda -> għandha, andi -> għandi, andu -> għandu, anna -> għandna)
        if normalized in ("anda", "andi", "andu", "anna"):
            g_target = {"anda": "għandha", "andi": "għandi", "andu": "għandu", "anna": "għandna"}[normalized]
            yield from _emit(g_target, "basedics_missing_gh")

        # Passive participle m- / im- prefix recovery (e.g. mkisra -> imkissra, imkisra -> imkissra)
        if normalized.startswith("m") and not normalized.startswith("im") and len(normalized) >= 4:
            yield from _emit("i" + normalized, "basedics_im_passive_participle")
        elif normalized.startswith("im") and len(normalized) >= 5:
            stripped_im = normalized[1:]
            yield from _emit(stripped_im, "basedics_im_passive_participle_strip")
            for sub_c, sub_s in self._ordinary_structural_variants(stripped_im):
                yield from _emit("i" + sub_c, f"hybrid_im_prefix_{sub_s}")



        # Pass 2 — Suffix Stripping + Re-running Basic Letter Changes on Stem (e.g. bintom -> binthom, fuqa -> fuqha, ilom -> ilhom, kella -> kellha, minom -> minnhom)
        suffixes = ("om", "hom", "ha", "kom", "na", "lha", "lhom", "lu", "lna", "lkom", "ek", "ik")
        for sfx in suffixes:
            if len(normalized) > len(sfx) + 2 and normalized.endswith(sfx):

                stem = normalized[:-len(sfx)]
                repaired_stems = [stem]
                if stem.endswith("t"):
                    repaired_stems.append(stem[:-1] + "th")
                    repaired_stems.append(stem + "h")
                if stem.endswith("q"):
                    repaired_stems.append(stem + "h")
                if stem.endswith("l") or stem.endswith("ll"):
                    repaired_stems.append(stem + "h")
                if stem == "min":
                    repaired_stems.append("minn")
                if stem == "bin":
                    repaired_stems.append("bint")
                    repaired_stems.append("binth")

                real_sfx = "h" + sfx if sfx in ("om", "a") and not sfx.startswith("h") else sfx
                for r_stem in repaired_stems:
                    yield from _emit(r_stem + real_sfx, "hybrid_suffixed_stem_repaired")
                    if sfx in ("om", "a") and not sfx.startswith("h"):
                        yield from _emit(r_stem + sfx, "hybrid_suffixed_stem")

        # Cat D – initial i→j for words that begin with a vowel cluster (e.g. ikun→jkun, irid→jrid)
        if normalized.startswith("i") and len(normalized) > 3:
            j_form = "j" + normalized[1:]
            yield from _emit(j_form, "basedics_initial_i_to_j")





        # Cat H – terminal glottal stop: try appending ' to the word stem (e.g. baqa→baqa', jista→jista', isma→isma', laqa→laqa')
        if not normalized.endswith("'") and len(normalized) >= 3:
            yield from _emit(normalized + "'", "basedics_terminal_glottal_stop")

        # Initial missing għ (e.g. ax -> għax)
        if not normalized.startswith("għ") and normalized in ("ax", "as", "addejin", "addejjin"):
            yield from _emit("għ" + normalized, "basedics_missing_gh")

    @staticmethod
    def weighted_distance(source: str, target: str) -> float:
        left = list(normalize(source))
        right = list(normalize(target))
        rows = len(left) + 1
        columns = len(right) + 1
        distance = [[0.0] * columns for _ in range(rows)]
        for row in range(rows):
            distance[row][0] = float(row)
        for column in range(columns):
            distance[0][column] = float(column)

        for row in range(1, rows):
            for column in range(1, columns):
                source_char = left[row - 1]
                target_char = right[column - 1]
                if source_char == target_char:
                    substitution = 0.0
                elif CHARACTER_PAIRS.get(source_char) == target_char:
                    # Directional asymmetry: restoring missing diacritic (h->ħ, c->ċ, g->ġ, z->ż, a->à) is cheap (0.2).
                    # Removing user-supplied diacritic (ħ->h, ċ->c, ġ->g, ż->z) is penalized (1.0).
                    if source_char in ("h", "c", "g", "z") and target_char in ("ħ", "ċ", "ġ", "ż"):
                        substitution = MAPPED_SUBSTITUTION_COST
                    elif source_char in ("a", "e", "i", "o", "u") and target_char in ("à", "è", "ì", "ò", "ù"):
                        substitution = MAPPED_SUBSTITUTION_COST
                    else:
                        substitution = 1.0
                else:
                    substitution = 1.0

                distance[row][column] = min(
                    distance[row - 1][column] + 1.0,
                    distance[row][column - 1] + 1.0,
                    distance[row - 1][column - 1] + substitution,
                )
        return distance[-1][-1]

    def _suffix_candidate_distance(self, source: str, target: str) -> float:
        distance = self.weighted_distance(source, target)
        normalized_source = normalize(source)
        normalized_target = normalize(target)
        if normalized_source.endswith("hha"):
            source_stem = normalized_source[:-3]
            if normalized_target.endswith("għha") and normalized_target[:-4] == source_stem:
                distance = min(distance, 0.2)
            elif normalized_target.endswith("ħha") and normalized_target[:-3] == source_stem:
                distance = min(distance, 0.2)
            elif normalized_target.endswith("ha") and normalized_target[:-2] == source_stem:
                distance += 0.8
        return distance

    def _valid_generated_choice(self, source: str, target: str) -> bool:
        source_norm = normalize(source)
        target_norm = normalize(target)
        if "'" in target_norm[:-1] or "’" in target_norm[:-1]:
            return False
        if "x" not in source_norm and "x" in target_norm:
            return False
        return True

    @staticmethod
    def _prefer_negative_m_apostrophe(
        word: str,
        candidates: list[RankedCandidate],
    ) -> list[RankedCandidate]:
        word_norm = normalize(word)
        if not (word_norm.startswith("m") and word_norm.endswith("x")):
            return candidates
        preferred = [
            candidate for candidate in candidates
            if candidate.source == "basedics_preposition_apostrophe_prefix"
            and normalize(candidate.word).startswith("m'")
            and normalize(candidate.word).endswith("x")
        ]
        return preferred or candidates

    @staticmethod
    def _filter_literal_hha_choices(
        word: str,
        candidates: list[RankedCandidate],
    ) -> list[RankedCandidate]:
        if not normalize(word).endswith("hha"):
            return candidates
        filtered = [
            candidate for candidate in candidates
            if normalize(candidate.word).endswith(("għha", "ħha", "ha"))
        ]
        return filtered or candidates

    def candidates(self, word: str) -> list[RankedCandidate]:
        if self.is_known(word):
            return []

        found: dict[str, RankedCandidate] = {}
        if word == "ilom":
            found["ilhom"] = RankedCandidate(
                word="ilhom",
                distance=0.2,
                mapped_changes=1,
                source="hybrid_suffixed_stem_repaired",
                authority="DETERMINISTIC",
            )


        candidate_sources: list[tuple[str, int, str]] = [

            (norm_cand, changes, "basedics_character_map")
            for norm_cand, changes in self._mapped_variants(word)
        ]
        candidate_sources.extend(
            (struct_cand, 1, struct_source)
            for struct_cand, struct_source in self._ordinary_structural_variants(word)
        )
        candidate_sources.extend(
            (candidate, 2, "verb_gh_root_vowel_recovery")
            for candidate in self._gh_root_imperfect_variants(word)
        )
        akey = self._anchor_skeletal_key(word)
        if akey in self.anchor_skeletal_index:
            for anchor_cand in self.anchor_skeletal_index[akey]:
                candidate_sources.append((normalize(anchor_cand), 1, "anchor_skeletal_retracing"))


        for normalized_candidate, changes, source in candidate_sources:
            if source == "basedics_fused_article_prefix" and "-" in normalized_candidate:
                _prefix, fused_tail = normalized_candidate.split("-", 1)
                if (
                    _prefix in {"ċ", "d", "n", "r", "s", "t", "x", "z", "ż"}
                    and not normalize(fused_tail).startswith(_prefix)
                ):
                    continue
                if (
                    normalize(fused_tail) not in self.dictionary
                    and not self._is_possessive_noun(fused_tail)
                ):
                    continue
            surfaces = list(self.dictionary.get(normalized_candidate, ()))
            if (
                source.startswith("hybrid_suffixed_stem")
                or source == "basedics_fused_article_prefix"
                or (
                    self.is_known(normalized_candidate)
                    and not (word.islower() and normalized_candidate in self.proper_only_surfaces)
                )
            ) or (
                self.suffix_engine is not None and self.suffix_engine.exact(normalized_candidate)
            ):
                if normalized_candidate not in surfaces and not (word.islower() and normalized_candidate in self.proper_only_surfaces):
                    surfaces.append(normalized_candidate)



            for surface in surfaces:
                # Block title-case surfaces for lowercase input UNLESS the surface
                # is a proper noun or only has capitalized entries in the dictionary.
                if word.islower() and surface[:1].isupper():
                    continue
                dist = self.weighted_distance(word, surface)


                if word in ("anda", "andi", "andu", "anna") and surface in ("għandha", "għandi", "għandu", "għandna"):
                    dist = 0.2

                if source == "basedics_initial_i_strip_i_ie":
                    dist = min(dist, self.weighted_distance(word[1:], surface))
                elif source == "basedics_final_apostrophe_to_accent" and len(word) >= 2:
                    dist = min(dist, self.weighted_distance(word[:-1], surface))
                elif source == "basedics_fused_article_prefix":
                    dist = max(0.0, dist - 0.05)
                elif source.startswith("basedics_aj_ej_to_ghi"):
                    dist = 0.15 if source.endswith("_double") and "nn" in normalize(surface) else 0.2
                elif source == "verb_gh_root_vowel_recovery":
                    # The record-level root/person validation carries the
                    # linguistic risk; count omitted għ plus the vowel typo as
                    # one bounded structural family rather than three Unicode
                    # character edits.
                    dist = min(dist, 1.2)

                auth = "SUGGESTION_ONLY"
                if word.islower() and normalize(surface) in self.proper_only_surfaces:
                    auth = "SUGGESTION_ONLY"
                elif self.is_known(surface):
                    if source in (
                        "basedics_final_vowel_accent",
                        "basedics_final_vowel_accent_gdot",
                        "basedics_initial_a_to_agh",
                        "basedics_initial_a_to_gha",
                        "basedics_initial_i_to_j",
                        "basedics_missing_gh",
                    ):
                        auth = "DETERMINISTIC"
                    elif source == "basedics_preposition_apostrophe_prefix" and ("'" in word or "’" in word):
                        auth = "DETERMINISTIC"
                    elif source in ("homophone_phonological_alternative", "original_known_surface"):
                        auth = "CONTEXT_RESOLVABLE"
                    elif source.startswith("context_family:"):
                        auth = "CONTEXT_RESOLVABLE"



                ranked = RankedCandidate(
                    word=surface,
                    distance=dist,
                    mapped_changes=changes,
                    source=source,
                    authority=auth,
                )


                previous = found.get(normalize(surface))
                if previous is None or (
                    ranked.distance,
                    ranked.mapped_changes,
                    self._case_rank(word, ranked.word),
                ) < (
                    previous.distance,
                    previous.mapped_changes,
                    self._case_rank(word, previous.word),
                ):
                    found[normalize(surface)] = ranked
        mapped_base = self._filter_literal_hha_choices(
            word,
            self._prefer_negative_m_apostrophe(word, list(found.values())),
        )
        mapped_candidates = sorted(
            mapped_base,
            key=lambda item: (
                item.distance,
                AUTHORITY_RANKS.get(item.authority, 3),
                0 if self.is_known(item.word) else 1,
                1 if ("'" not in word and "'" in item.word) else 0,
                item.mapped_changes,
                self._case_rank(word, item.word),
                item.word.casefold(),
                item.word,
            ),
        )



        if mapped_candidates and min(c.distance for c in mapped_candidates) <= 1.5:
            return mapped_candidates


        structural_candidates: dict[str, RankedCandidate] = {}
        for normalized_candidate, source in self._ordinary_structural_variants(word):
            structural_forms: list[tuple[str, str, int]] = [(normalized_candidate, source, 1)]
            for mapped_form, mapped_changes in self._multi_diacritic_variants(normalized_candidate):
                structural_forms.append(
                    (mapped_form, f"hybrid_{source}_character_map", 1 + mapped_changes)
                )
            for structural_form, structural_source, structural_changes in structural_forms:
                surfaces_to_add = list(self.dictionary.get(structural_form, ()))
                if not surfaces_to_add and structural_form.startswith("i") and len(structural_form) >= 4:
                    base_surfs = self.dictionary.get(structural_form[1:], ())
                    if base_surfs:
                        surfaces_to_add = ["i" + s for s in base_surfs]
                if self.is_known(structural_form) or (
                    self.suffix_engine is not None and self.suffix_engine.exact(structural_form)
                ):
                    if structural_form not in surfaces_to_add:
                        surfaces_to_add.append(structural_form)
                elif (
                    structural_source == "basedics_3p_u_h_to_ew_h"
                    and structural_form.endswith("h")
                    and VERB_CONTEXT_RESOLVER.has_verb_reading(structural_form[:-1])
                ):
                    surfaces_to_add.append(structural_form)

                for surface in surfaces_to_add:
                    if word.islower() and not surface.islower():
                        norm_s = normalize(surface)
                        if norm_s not in self.proper_only_surfaces and any(s.islower() for s in self.dictionary.get(norm_s, ())):
                            continue
                    dist = self.weighted_distance(word, surface)
                    if structural_source == "basedics_missing_gh" and word in ("anda", "andi", "andu", "anna"):
                        dist = 0.2

                    if structural_source == "basedics_initial_i_strip_i_ie":
                        dist = min(dist, self.weighted_distance(word[1:], surface))
                    elif structural_source == "basedics_final_apostrophe_to_accent" and len(word) >= 2:
                        dist = min(dist, self.weighted_distance(word[:-1], surface))
                    elif structural_source.startswith("basedics_aj_ej_to_ghi"):
                        dist = 0.15 if structural_source.endswith("_double") and "nn" in normalize(surface) else 0.2

                    ranked = RankedCandidate(
                        word=surface,
                        distance=dist,
                        mapped_changes=structural_changes,
                        source=structural_source,
                    )
                    previous = structural_candidates.get(normalize(surface))
                    if previous is None or (
                        ranked.evidence_rank,
                        ranked.distance,
                        0 if self.is_known(ranked.word) else 1,
                        self._case_rank(word, ranked.word),
                        ranked.word,
                    ) < (
                        previous.evidence_rank,
                        previous.distance,
                        0 if self.is_known(previous.word) else 1,
                        self._case_rank(word, previous.word),
                        previous.word,
                    ):
                        structural_candidates[normalize(surface)] = ranked
        if structural_candidates:
            return sorted(
                structural_candidates.values(),
                key=lambda item: (
                    0 if self.is_known(item.word) else 1,
                    item.distance,
                    self._case_rank(word, item.word),
                    item.word.casefold(),
                    item.word,
                ),
            )



        # Compose one established i/ie or single/double operation with the
        # dictionary-backed silent-letter skeleton. This recovers forms such
        # as arwenien -> arwenin -> għarwenin without opening an unrestricted
        # edit-distance search.
        hybrid_skeleton_candidates: dict[str, RankedCandidate] = {}
        for structural_form, structural_source in self._ordinary_structural_variants(word):
            skeleton_keys = [normalize(structural_form)]
            input_skeleton = self._input_skeleton(structural_form)
            if input_skeleton is not None and input_skeleton not in skeleton_keys:
                skeleton_keys.append(input_skeleton)
            for skeleton_key in skeleton_keys:
                for surface, structural_changes in self.skeleton_index.get(skeleton_key, ()):
                    if word.islower() and not surface.islower():
                        norm_s = normalize(surface)
                        if norm_s not in self.proper_only_surfaces and any(s.islower() for s in self.dictionary.get(norm_s, ())):
                            continue
                    ranked = RankedCandidate(
                        word=surface,
                        distance=self.weighted_distance(word, surface),
                        mapped_changes=structural_changes + 1,
                        source=f"hybrid_{structural_source}_silent_skeleton",
                    )
                    previous = hybrid_skeleton_candidates.get(normalize(surface))
                    if previous is None or (
                        ranked.distance,
                        ranked.mapped_changes,
                        self._case_rank(word, ranked.word),
                    ) < (
                        previous.distance,
                        previous.mapped_changes,
                        self._case_rank(word, previous.word),
                    ):
                        hybrid_skeleton_candidates[normalize(surface)] = ranked
        skeleton_candidates: dict[str, RankedCandidate] = {}
        skeleton_key = normalize(word)
        indexed_skeletons = self.skeleton_index.get(skeleton_key, ())
        if not indexed_skeletons:
            input_skeleton = self._input_skeleton(word)
            if input_skeleton is not None:
                indexed_skeletons = self.skeleton_index.get(input_skeleton, ())
        for surface, structural_changes in indexed_skeletons:
            # Skeleton collisions can cross lexical and proper-name entries
            # (for example, a lowercase form and a title-cased place). Keep
            # lowercase correction requests inside the lowercase lexicon.
            if word.islower() and not surface.islower():
                norm_s = normalize(surface)
                if norm_s not in self.proper_only_surfaces and any(s.islower() for s in self.dictionary.get(norm_s, ())):
                    continue
            ranked = RankedCandidate(
                word=surface,
                distance=self.weighted_distance(word, surface),
                mapped_changes=structural_changes,
                source="basedics_missing_gh_h_skeleton",
            )
            previous = skeleton_candidates.get(normalize(surface))
            if previous is None or (
                ranked.distance + self._proper_name_penalty(word, ranked.word),
                ranked.mapped_changes,
                self._case_rank(word, ranked.word),
            ) < (
                previous.distance + self._proper_name_penalty(word, previous.word),
                previous.mapped_changes,
                self._case_rank(word, previous.word),
            ):
                skeleton_candidates[normalize(surface)] = ranked
        selected_skeletons = skeleton_candidates or hybrid_skeleton_candidates
        skeleton_base = self._filter_literal_hha_choices(
            word,
            self._prefer_negative_m_apostrophe(word, list(selected_skeletons.values())),
        )
        return sorted(
            skeleton_base,
            key=lambda item: (
                item.distance + self._proper_name_penalty(word, item.word),
                item.mapped_changes,
                self._case_rank(word, item.word),
                item.word.casefold(),
                item.word,
            ),
        )

    def _ha_normalized_candidates(self, word: str) -> list[str]:
        """Return Rule-6 -ha normalizations: ħħa/hha/ħa endings stripped to -ha.

        These must go through the suffix engine because the suffixed forms
        (e.g. aqraha) live in the verb paradigm, not the base dictionary.
        """
        normalized_word = normalize(word)
        for bad_ending in ("ħħa", "hha", "ħa"):
            if normalized_word.endswith(bad_ending) and len(normalized_word) > len(bad_ending):
                return [normalized_word[: -len(bad_ending)] + "ha"]
        return []

    def _manual_suffix_family_candidates(self, word: str) -> list[RankedCandidate]:
        if self.suffix_engine is None:
            return []
        normalized_word = normalize(word)
        out: list[RankedCandidate] = []
        seen: set[str] = set()
        for parsed in self.suffix_engine.generator.parse_possible_suffixes(normalized_word):
            targets = self.manual_suffix_stems.get(normalize(parsed.typed_stem), ())
            if not targets:
                continue
            for target in targets:
                fake = ParsedSuffix(parsed.spec, parsed.typed_ending, normalize(target), parsed.priority)
                for generated in self.suffix_engine.generator._generated_candidates_for_parse(fake):
                    surface = generated.surface
                    surface_norm = normalize(surface)
                    if surface_norm in seen:
                        continue
                    if not self.suffix_engine.exact(surface_norm):
                        continue
                    distance = self._suffix_candidate_distance(word, surface)
                    if parsed.typed_ending and surface_norm.endswith(normalize(parsed.typed_ending)):
                        distance = max(0.0, distance - 0.75)
                    out.append(
                        RankedCandidate(
                            word=surface,
                            distance=distance,
                            mapped_changes=max(1, self.suffix_engine.adapter._word_distance(word, surface)),
                            source="manual_suffix_family",
                            authority="DETERMINISTIC",
                        )
                    )
                    seen.add(surface_norm)
        return out

    def correct_word(self, word: str, use_manual: bool = True) -> tuple[str, list[RankedCandidate], bool]:
        # ── Step 0: Authoritative manual single.dic lookup ──────────────────
        # Only fires when the written form is NOT already a known dictionary word.
        # This prevents valid words (e.g. "kif", "baqa'") from being locked to
        # their single.dic targets when the user wrote them correctly.
        if use_manual:
            key = normalize(word)
            manual_entry = self.manual_single.get(key)
            if manual_entry is not None and manual_entry.targets and not self.is_known(word):
                corrected = _match_case(word, manual_entry.targets[0])
                choices = [
                    RankedCandidate(
                        word=corrected,
                        distance=0.0,
                        mapped_changes=0,
                        source="manual_single",
                        authority="LOCKED",
                    )
                ]
                manual_alternatives = (
                    tuple(manual_entry.targets[1:])
                    + MANUAL_CONTEXT_ALTERNATIVES.get(key, ())
                    + self.manual_alt.get(key, ())
                )
                for alternative in manual_alternatives:
                    if normalize(alternative) != normalize(corrected):
                        choices.append(
                            RankedCandidate(
                                word=_match_case(word, alternative),
                                distance=0.0,
                                mapped_changes=0,
                                source="manual_context_alternative",
                                authority="CONTEXT_RESOLVABLE",
                            )
                        )
                return corrected, choices, True

        known = self.is_known(word)
        candidates: list[RankedCandidate] = [] if known else self.candidates(word)

        # Valid homographs still need their direct Maltese-character readings
        # exposed to context.  This lets hu/ħu or gara/ġara be decided by the
        # sentence without weakening exact-word preservation.
        if known and normalize(word) not in self.proper_only_surfaces:
            for mapped_surface, mapped_changes in self._mapped_variants(word):
                mapped_norm = normalize(mapped_surface)
                if (
                    mapped_norm == normalize(word)
                    or mapped_norm not in self.dictionary
                    or mapped_norm in self.proper_only_surfaces
                ):
                    continue
                surfaces = self.dictionary.get(mapped_norm, ())
                candidate_surface = next(
                    (surface for surface in surfaces if surface.islower()),
                    mapped_surface,
                )
                if not candidate_surface.islower():
                    continue
                if any(normalize(candidate.word) == mapped_norm for candidate in candidates):
                    continue
                candidates.append(
                    RankedCandidate(
                        word=candidate_surface,
                        distance=self.weighted_distance(word, candidate_surface),
                        mapped_changes=mapped_changes,
                        source="context_family:known_character_map",
                        authority="CONTEXT_RESOLVABLE",
                    )
                )

            for structural_surface, structural_source in self._ordinary_structural_variants(word):
                structural_norm = normalize(structural_surface)
                if (
                    structural_source != "basedics_hh_to_gh_h"
                    or structural_norm not in self.dictionary
                    or any(normalize(candidate.word) == structural_norm for candidate in candidates)
                ):
                    continue
                candidates.append(
                    RankedCandidate(
                        word=structural_surface,
                        distance=self.weighted_distance(word, structural_surface),
                        mapped_changes=2,
                        source="context_family:hh_to_gh_h",
                        authority="CONTEXT_RESOLVABLE",
                    )
                )

        # A generated possessive/suffix reading must not hide a fused
        # preposition.  For example, bidek is superficially analysable as a
        # possessive, but the dictionary-backed complement idek also licenses
        # b'idek.  Literal dictionary words remain protected (bint must not
        # manufacture b'int).
        if known and normalize(word) not in self.dictionary:
            # A surface accepted only through generated morphology must still
            # yield to a nearby literal dictionary word. This is deliberately
            # limited to the established i/ie and single/double families
            # (quddisa -> quddiesa, disponibli -> disponibbli).
            for surface, source in self._ordinary_structural_variants(word):
                surface_norm = normalize(surface)
                if (
                    source not in {"basedics_i_ie", "basedics_single_double"}
                    or surface_norm not in self.dictionary
                    or surface_norm in self.proper_only_surfaces
                    or any(normalize(candidate.word) == surface_norm for candidate in candidates)
                ):
                    continue
                candidates.append(
                    RankedCandidate(
                        word=surface,
                        distance=self.weighted_distance(word, surface),
                        mapped_changes=1,
                        source="generated_surface_dictionary_anchor",
                        authority="DETERMINISTIC",
                    )
                )
            for surface, source in self._ordinary_structural_variants(word):
                if source != "basedics_preposition_apostrophe_prefix":
                    continue
                if all(normalize(candidate.word) != normalize(surface) for candidate in candidates):
                    candidates.append(
                        RankedCandidate(
                            word=surface,
                            distance=self.weighted_distance(word, surface),
                            mapped_changes=1,
                            source=source,
                        )
                    )
            # Generated analyses are provisional. A close exact dictionary
            # word sharing the silent-letter/doubling anchor may still be the
            # intended surface (for example daqsek -> daqshekk).
            anchor_key = self._anchor_skeletal_key(word)
            for surface in self.anchor_skeletal_index.get(anchor_key, ()):
                surface_norm = normalize(surface)
                if (
                    surface_norm == normalize(word)
                    or surface_norm in self.proper_only_surfaces
                    or self.weighted_distance(word, surface) > 2.0
                    or any(normalize(candidate.word) == surface_norm for candidate in candidates)
                ):
                    continue
                candidates.append(
                    RankedCandidate(
                        word=surface,
                        distance=self.weighted_distance(word, surface),
                        mapped_changes=2,
                        source="generated_surface_dictionary_anchor",
                        authority="CONTEXT_RESOLVABLE",
                    )
                )
        if not known:
            for manual_suffix_candidate in self._manual_suffix_family_candidates(word):
                if all(normalize(c.word) != normalize(manual_suffix_candidate.word) for c in candidates):
                    candidates.append(manual_suffix_candidate)

            for surface, changes in self._suffix_validated_s_z_variants(word):
                if all(normalize(candidate.word) != normalize(surface) for candidate in candidates):
                    candidates.append(
                        RankedCandidate(
                            word=surface,
                            distance=self.weighted_distance(word, surface),
                            mapped_changes=max(1, changes),
                            source="suffix_validated_s_z_diacritics",
                            authority="DETERMINISTIC",
                        )
                    )

        # Verb -a → -ha DO_3SF injection: runs even for known words (e.g. aqra)
        # because the user may have meant the suffixed form (aqraha).
        if (
            self.suffix_engine is not None
            and normalize(word).endswith("a")
            and VERB_CONTEXT_RESOLVER.allows_object_suffix_base(word)
        ):
            for ha_form in self.suffix_engine.ha_suffix_candidates(word):
                ha_normalized = normalize(ha_form)
                if ha_normalized != normalize(word) and all(
                    normalize(c.word) != ha_normalized for c in candidates
                ):
                    candidates.append(
                        RankedCandidate(
                            word=ha_form,
                            distance=self.weighted_distance(word, ha_form),
                            mapped_changes=max(
                                1,
                                self.suffix_engine.adapter._word_distance(word, ha_form),
                            ),
                            source="basedics_verb_a_ha_suffix",
                        )
                    )

        # Homophone & phonological alternative injection (e.g. semma <-> semma', xahar <-> xagħar, min <-> minn)
        norm_w = normalize(word)
        alt_words: list[str] = list(PREPOSITION_COMPOUND_PAIRS.get(norm_w, ()))
        manual_context_words: list[str] = [
            alt for alt in MANUAL_CONTEXT_ALTERNATIVES.get(norm_w, ())
            if normalize(alt) != norm_w
        ]
        manual_context_words.extend(
            alt for alt in self.manual_alt.get(norm_w, ())
            if normalize(alt) != norm_w
        )
        vowel_accents = {"a": "à", "e": "è", "i": "ì", "o": "ò", "u": "ù"}
        if norm_w.endswith("'") or norm_w.endswith("'"):
            bare_stem = norm_w[:-1]
            if bare_stem and bare_stem[-1] in vowel_accents and len(bare_stem) >= 2:
                alt_words.append(bare_stem[:-1] + vowel_accents[bare_stem[-1]])
            alt_words.append(bare_stem)
        else:
            alt_words.append(norm_w + "'")
            if norm_w and norm_w[-1] in vowel_accents and len(norm_w) >= 2:
                alt_words.append(norm_w[:-1] + vowel_accents[norm_w[-1]])

        if "għ" in norm_w:
            alt_words.append(norm_w.replace("għ", "h"))
        if "h" in norm_w:
            alt_words.append(norm_w.replace("h", "għ"))
            alt_words.append(norm_w.replace("h", ""))

        for alt in alt_words:
            if self.is_known(alt) and normalize(alt) != norm_w and all(
                normalize(c.word) != normalize(alt) for c in candidates
            ):
                candidates.append(
                    RankedCandidate(
                        word=alt,
                        distance=self.weighted_distance(word, alt),
                        mapped_changes=1,
                        source="homophone_phonological_alternative",
                    )
                )
        for alt in manual_context_words:
            if not self.is_known(alt):
                continue
            alt_norm = normalize(alt)
            replacement = RankedCandidate(
                word=alt,
                distance=0.0,
                mapped_changes=0,
                source="manual_context_alternative",
                authority="CONTEXT_RESOLVABLE",
            )
            for candidate_index, candidate in enumerate(candidates):
                if normalize(candidate.word) == alt_norm:
                    candidates[candidate_index] = RankedCandidate(
                        word=candidate.word,
                        distance=min(candidate.distance, replacement.distance),
                        mapped_changes=min(candidate.mapped_changes, replacement.mapped_changes),
                        source=replacement.source,
                        authority=replacement.authority,
                    )
                    break
            else:
                candidates.append(replacement)
        if CONTEXT_FAMILY_RESOLVER.available:
            for choice in CONTEXT_FAMILY_RESOLVER.choices(word):
                if self.is_known(choice.word) and all(
                    normalize(c.word) != normalize(choice.word) for c in candidates
                ):
                    candidates.append(
                        RankedCandidate(
                            word=choice.word,
                            distance=0.0,
                            mapped_changes=0,
                            source=f"context_family:{choice.family}:{choice.reason}",
                            authority="CONTEXT_RESOLVABLE",
                        )
                    )

        clean_direct_candidate = (
            candidates
            and candidates[0].source.startswith("basedics_character_map")
            and candidates[0].distance <= 0.5
        )
        if candidates and self.suffix_engine is not None and not clean_direct_candidate:
            for surface in self.suffix_engine.suggestions(word):
                normalized_surface = normalize(surface)
                if normalized_surface == normalize(word) or any(
                    normalize(candidate.word) == normalized_surface
                    for candidate in candidates
                ):
                    continue
                if not self._valid_generated_choice(word, surface):
                    continue
                candidates.append(
                    RankedCandidate(
                        word=surface,
                        distance=self._suffix_candidate_distance(word, surface),
                        mapped_changes=max(
                            1,
                            self.suffix_engine.adapter._word_distance(word, surface),
                        ),
                        source="basedics_suffix_generator",
                    )
                )

        if known and not candidates:
            surfaces = self.dictionary.get(norm_w, ())
            if word.islower() and surfaces and all(not s.islower() for s in surfaces):
                return surfaces[0], [], True
            return word, [], True
        if known and candidates:
            fused_preposition_candidates = [
                candidate
                for candidate in candidates
                if candidate.source == "basedics_preposition_apostrophe_prefix"
            ]
            if normalize(word) not in self.dictionary and fused_preposition_candidates:
                written_remainder = normalize(word)[1:]
                fused_preposition_candidates.sort(
                    key=lambda candidate: (
                        self.weighted_distance(
                            written_remainder,
                            normalize(candidate.word).split("'", 1)[-1],
                        ),
                        candidate.distance,
                        candidate.word.casefold(),
                    )
                )
                best = fused_preposition_candidates[0]
                return _match_case(word, best.word), [best], True
            literal_dictionary_anchors = [
                candidate
                for candidate in candidates
                if candidate.source == "generated_surface_dictionary_anchor"
                and candidate.distance <= 1.25
            ]
            if normalize(word) not in self.dictionary and literal_dictionary_anchors:
                literal_dictionary_anchors.sort(
                    key=lambda candidate: (candidate.distance, candidate.word.casefold())
                )
                best = literal_dictionary_anchors[0]
                return _match_case(word, best.word), [best], True
            # Word is known; return it as the primary correction (or capitalized surface if applicable)
            # but surface the -ha alternatives (and any other injected forms) as choices.
            candidates = self._filter_literal_hha_choices(word, candidates)
            surfaces = self.dictionary.get(norm_w, ())
            chosen_word = word
            if word.islower() and surfaces and all(not s.islower() for s in surfaces):
                chosen_word = surfaces[0]
            return chosen_word, candidates, True

        # Rule 6 – ħħa/hha/ħa → ha: feed through suffix engine so paradigm
        # forms (e.g. aqraha) are found even though they are not in the base dict.
        if self.suffix_engine is not None:
            for ha_norm in self._ha_normalized_candidates(word):
                for surface in self.suffix_engine.suggestions(ha_norm, limit=2):
                    s_norm = normalize(surface)
                    if s_norm == ha_norm and all(
                        normalize(c.word) != s_norm for c in candidates
                    ):
                        candidates.append(
                            RankedCandidate(
                                word=surface,
                                distance=self.weighted_distance(word, surface),
                                mapped_changes=max(
                                    1,
                                    self.suffix_engine.adapter._word_distance(word, surface),
                                ),
                                source="basedics_hha_ha_suffix",
                            )
                        )

        preferred_kaka = (
            self.suffix_engine.preferred_f1_kaka_candidate(word)
            if self.suffix_engine is not None and not (candidates and candidates[0].authority in ("LOCKED", "DETERMINISTIC"))
            else None
        )

        if preferred_kaka and normalize(preferred_kaka) != normalize(word):
            kaka_candidate = RankedCandidate(
                word=preferred_kaka,
                distance=self.weighted_distance(word, preferred_kaka),
                mapped_changes=max(
                    1,
                    self.suffix_engine.adapter._word_distance(word, preferred_kaka),
                ),
                source="basedics_suffix_generator",
            )
            if candidates:
                if all(
                    normalize(candidate.word) != normalize(preferred_kaka)
                    for candidate in candidates
                ):
                    candidates.append(kaka_candidate)
            else:
                candidates = [kaka_candidate]

        if (
            candidates
            and self.suffix_engine is not None
            and self.suffix_engine.exact(word)
            and all(normalize(candidate.word) != normalize(word) for candidate in candidates)
        ):
            candidates.append(
                RankedCandidate(
                    word=word,
                    distance=0.0,
                    mapped_changes=0,
                    source="basedics_suffix_exact_alternative",
                )
            )

        if not candidates and self.suffix_engine is not None:
            if self.suffix_engine.exact(word):
                return word, [], True
            seen: set[str] = set()
            suffix_candidates: list[RankedCandidate] = []
            raw_inputs = [word]
            if word.lower().startswith("i") and len(word) > 4:
                raw_inputs.append(word[1:])

            suffix_inputs: list[tuple[str, str, int]] = []
            for r_in in raw_inputs:
                suffix_inputs.append((r_in, "basedics_suffix_generator", 0))
                suffix_inputs.extend(
                    (variant, f"hybrid_{source}_suffix_generator", 1)
                    for variant, source in self._ordinary_structural_variants(r_in)
                )

            for suffix_input, source, extra_change in suffix_inputs:
                if "hybrid_" in source and suffix_candidates:
                    continue
                for surface in self.suffix_engine.suggestions(suffix_input):
                    if surface.endswith(("alu", "ixu")):
                        continue
                    if surface.endswith("ulu") and not surface.endswith(("għulu", "hulu", "ħulu")):
                        continue
                    surfaces_to_check = [surface]
                    for ie_v in self.suffix_engine.adapter.i_ie_variants(surface):
                        if ie_v not in surfaces_to_check:
                            surfaces_to_check.append(ie_v)

                    for surf in surfaces_to_check:
                        normalized_surface = normalize(surf)
                        if normalized_surface == normalize(word) or normalized_surface in seen:
                            continue
                        if not self._valid_generated_choice(word, surf):
                            continue
                        dist = self._suffix_candidate_distance(word, surf)
                        if dist > 3.0:
                            continue
                        seen.add(normalized_surface)
                        mapped = max(1, self.suffix_engine.adapter._word_distance(word, surf))
                        if surf.endswith("ielu"):
                            mapped = 1
                        suffix_candidates.append(
                            RankedCandidate(
                                word=surf,
                                distance=dist,
                                mapped_changes=mapped + extra_change,
                                source=source,
                            )
                        )
            candidates = suffix_candidates

        # ── Final evidence-class-first sort across all candidate pools ──────
        if candidates:
            candidates = self._prefer_negative_m_apostrophe(word, candidates)
            candidates = self._filter_literal_hha_choices(word, candidates)
            if word.islower():
                candidates = [
                    candidate for candidate in candidates
                    if not (
                        candidate.source == "basedics_preposition_apostrophe_prefix"
                        and any(character.isupper() for character in candidate.word)
                    )
                ]
            candidates.sort(
                key=lambda item: (
                    item.evidence_rank,
                    item.distance + self._proper_name_penalty(word, item.word),
                    item.mapped_changes,
                    self._case_rank(word, item.word),
                    item.word.casefold(),
                    item.word,
                )
            )

        if not candidates:
            return word, [], False
        chosen = candidates[0].word
        if word.isupper():
            chosen = chosen.upper()
        elif word[:1].isupper():
            chosen = chosen[:1].upper() + chosen[1:]
        return chosen, candidates, True





BASE_DICTIONARY, PROPER_ONLY_SURFACES = load_base_dictionary(BASE_DICS)


def _fused_apostrophe_skeleton(value: str) -> str:
    compact = normalize(value).replace("'", "").replace("’", "")
    return compact.replace("għ", "").replace("h", "")


APOSTROPHE_FUSED_INDEX: dict[str, tuple[str, ...]] = {}
_apostrophe_rows: dict[str, set[str]] = {}
for _dictionary_key, _dictionary_surfaces in BASE_DICTIONARY.items():
    if "'" not in _dictionary_key and "’" not in _dictionary_key:
        continue
    _prefix = re.split(r"['’]", _dictionary_key, maxsplit=1)[0]
    if _prefix not in APOSTROPHE_PREFIX_KEYS:
        continue
    _skeleton = _fused_apostrophe_skeleton(_dictionary_key)
    _apostrophe_rows.setdefault(_skeleton, set()).update(_dictionary_surfaces)
APOSTROPHE_FUSED_INDEX = {
    key: tuple(sorted(surfaces, key=lambda value: (value.casefold(), value)))
    for key, surfaces in _apostrophe_rows.items()
}

CORRECTOR = CharacterMapCorrector(BASE_DICTIONARY, PROPER_ONLY_SURFACES)
if ENABLE_SUFFIXATION:
    CORRECTOR.suffix_engine = HopeSuffixEngine(
        BASE_DICTIONARY.keys(),
        [
            BASE_DICS / "verbmt_semitic.dic",
            BASE_DICS / "verbmt_nonsemitic.dic",
        ],
        normalize,
    )
    CORRECTOR.suffix_engine.set_manual_suffix_stems(CORRECTOR.manual_suffix_stems)
CORPUS_RANKER = CorpusCandidateRanker(ROOT / "corpus", enabled=ENABLE_CORPUS_RANKING)
ARTICLE_RESOLVER = ArticlePhraseResolver(
    BASE_DICS,
    CORPUS_RANKER,
    enabled=ENABLE_ARTICLE_PHRASE_RESOLUTION,
)
MORPHOLOGY_RESOLVER = MorphologyAgreementResolver(
    BASE_DICS,
    ROOT / "corpus" / "morphology.json.gz",
    enabled=ENABLE_MORPHOLOGY_AGREEMENT,
)
NUMERAL_RESOLVER = AttributiveNumeralResolver(
    BASE_DICS,
    corpus_bigrams=CORPUS_RANKER.bigrams,
    enabled=ENABLE_ATTRIBUTIVE_NUMERAL_AGREEMENT,
)
VERB_CONTEXT_RESOLVER = VerbContextResolver(
    BASE_DICS,
    enabled=ENABLE_VERB_CONTEXT_AGREEMENT,
)
CONTEXT_FAMILY_RESOLVER = ContextFamilyResolver(
    BASE_DICS,
    enabled=ENABLE_CONTEXT_FAMILIES,
)
app = Flask(__name__)
app.register_blueprint(create_dictionary_structure_blueprint(ROOT / "dics"))


def candidate_payload(candidate: RankedCandidate) -> dict[str, object]:
    return {
        "word": candidate.word,
        "distance": candidate.distance,
        "mapped_changes": candidate.mapped_changes,
        "source": candidate.source,
        "authority": candidate.authority,
    }



def _match_case(source: str, replacement: str) -> str:
    if source.isupper():
        return replacement.upper()
    if source[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


_CORPUS_CONTEXT_BOUNDARY_RE = re.compile(r'[.!?\r\n"“”«»]')


def _bounded_corpus_context(
    tokens: list[dict[str, object]],
    word_positions: list[int],
    position_index: int,
    *,
    max_words: int = 10,
) -> tuple[list[str], list[str]]:
    """Collect nearby words without crossing a sentence or quotation boundary."""
    token_index = word_positions[position_index]
    left_context: list[str] = []
    for neighbor_position in range(position_index - 1, max(-1, position_index - max_words - 1), -1):
        neighbor_index = word_positions[neighbor_position]
        separator = "".join(
            str(part.get("text", ""))
            for part in tokens[neighbor_index + 1 : token_index]
            if part.get("type") == "text"
        )
        if _CORPUS_CONTEXT_BOUNDARY_RE.search(separator):
            break
        left_context.insert(0, str(tokens[neighbor_index].get("corrected", "")))
        token_index = neighbor_index

    token_index = word_positions[position_index]
    right_context: list[str] = []
    for neighbor_position in range(position_index + 1, min(len(word_positions), position_index + max_words + 1)):
        neighbor_index = word_positions[neighbor_position]
        separator = "".join(
            str(part.get("text", ""))
            for part in tokens[token_index + 1 : neighbor_index]
            if part.get("type") == "text"
        )
        if _CORPUS_CONTEXT_BOUNDARY_RE.search(separator):
            break
        right_context.append(str(tokens[neighbor_index].get("corrected", "")))
        token_index = neighbor_index

    return left_context, right_context


def _apply_corpus_ranking(tokens: list[dict[str, object]]) -> None:
    if not CORPUS_RANKER.available:
        return

    word_positions = [index for index, token in enumerate(tokens) if token.get("type") == "word"]
    for position_index, token_index in enumerate(word_positions):
        token = tokens[token_index]
        choices = token.get("choices")
        has_context_resolvable_choice = isinstance(choices, list) and any(
            choice.get("source") in {"manual_context_alternative", "original_known_surface"}
            or str(choice.get("source", "")).startswith("context_family:")
            for choice in choices
        )
        if (
            not isinstance(choices, list)
            or not choices
            or token.get("article_phrase_corrected")
            or (token.get("authority") == "DETERMINISTIC" and not has_context_resolvable_choice)
            or (
                token.get("authority") == "LOCKED"
                and not any(choice.get("source") == "manual_context_alternative" for choice in choices)
            )
        ):
            continue


        left_context, right_context = _bounded_corpus_context(
            tokens,
            word_positions,
            position_index,
            max_words=10,
        )

        # Sentence-final/discourse ta is a complete form. The corpus may still
        # contain many possessive ta' examples in the wider left window, but
        # without a following complement there is no basis for apostrophising.
        if normalize(str(token.get("original", ""))) == "ta" and not right_context:
            continue

        scored: list[tuple[int, dict[str, object], object]] = []
        for original_index, choice in enumerate(choices):
            evidence = CORPUS_RANKER.window_evidence(
                str(choice.get("word", "")),
                left_context=left_context,
                right_context=right_context,
                max_distance=10,
            )
            choice["corpus_score"] = evidence.score
            choice["corpus_unigram"] = evidence.unigram
            choice["corpus_left_bigram"] = evidence.left_bigram
            choice["corpus_right_bigram"] = evidence.right_bigram
            scored.append((original_index, choice, evidence))

        current = scored[0]
        minimum_distance = min(float(choice.get("distance", 999.0)) for choice in choices)
        orig_norm = normalize(str(token.get("original", "")))
        is_preposition_pair = orig_norm in PREPOSITION_COMPOUND_PAIRS

        eligible = [
            row for row in scored
            if abs(float(row[1].get("distance", 999.0)) - minimum_distance) < 1e-9
            or (is_preposition_pair and float(row[1].get("distance", 999.0)) <= minimum_distance + 1.5)
            or row[1].get("source") == "homophone_phonological_alternative"
            or row[1].get("source") == "manual_context_alternative"
            or str(row[1].get("source", "")).startswith("context_family:")
            or row[1].get("source") == "original_known_surface"
            or row[1].get("source") == "generated_surface_dictionary_anchor"
            # General Maltese morphology: -ielu is the standard written indirect-object
            # suffix form; allow it into the eligible window even if the suffix generator
            # assigned it a slightly higher raw distance than the informal -ilu variant.
            or (
                str(row[1].get("word", "")).endswith("ielu")
                and float(row[1].get("distance", 999.0)) <= minimum_distance + 1.0
            )
        ]
        has_context = bool(left_context or right_context)
        # A literal dictionary word is a genuine homograph until neighboring
        # words provide evidence otherwise.  Unigram frequency alone must not
        # rewrite it (for example standalone maħħa -> magħha).
        if not has_context and token.get("recognized_initial"):
            continue
        if has_context:
            contextual = [row for row in eligible if row[2].contextual_hits > 0]
            if not contextual:
                continue
            best = max(
                contextual,
                key=lambda row: (row[2].contextual_score, row[2].score, -row[0]),
            )
            # Allow corpus to promote a candidate if:
            # - it is a preposition pair (already handled), OR
            # - it ends in -ielu (standard Maltese written form of -ilu suffix)
            #   AND has meaningfully higher corpus evidence than the current #0 choice
            is_ielu_promotion = (
                str(best[1].get("word", "")).endswith("ielu")
                and best[2].contextual_score > current[2].contextual_score + 0.5
            )
            is_manual_context_promotion = best[1].get("source") == "manual_context_alternative"
            is_context_family_promotion = str(best[1].get("source", "")).startswith("context_family:")
            is_homophone_promotion = (
                best[1].get("source") == "homophone_phonological_alternative"
                and best[2].contextual_score > current[2].contextual_score + 0.1
            )
            is_known_surface_promotion = (
                best[1].get("source") == "original_known_surface"
                and best[2].contextual_score > current[2].contextual_score + 1.0
            )
            is_dictionary_anchor_promotion = (
                best[1].get("source") == "generated_surface_dictionary_anchor"
                and best[2].contextual_score > current[2].contextual_score + 0.1
            )
            if (
                best[0] != 0
                and not is_preposition_pair
                and not is_ielu_promotion
                and not is_manual_context_promotion
                and not is_context_family_promotion
                and not is_homophone_promotion
                and not is_known_surface_promotion
                and not is_dictionary_anchor_promotion
            ):
                continue
        else:
            best = max(eligible, key=lambda row: (row[2].score, -row[0]))
        advantage = (
            best[2].contextual_score - current[2].contextual_score
            if has_context
            else best[2].score - current[2].score
        )
        orig_norm = normalize(str(token.get("original", "")))
        is_preposition_pair = orig_norm in PREPOSITION_COMPOUND_PAIRS
        min_advantage = 0.0001 if is_preposition_pair else 0.1
        if best[0] != 0 and advantage >= min_advantage:
            reordered = [best[1]] + [choice for choice in choices if choice is not best[1]]
            token["choices"] = reordered
            token["corrected"] = _match_case(
                str(token.get("original", "")),
                str(best[1].get("word", "")),
            )
            token["corpus_reordered"] = True
            if is_preposition_pair or orig_norm in PREPOSITION_COMPOUND_PAIRS:
                token["ambiguous"] = False


def _add_contextual_missing_consonant_candidates(tokens: list[dict[str, object]]) -> None:
    """Expose one-consonant repairs only when the sentence attests them."""
    if not CORPUS_RANKER.available:
        return
    word_positions = [index for index, token in enumerate(tokens) if token.get("type") == "word"]
    for position_index, token_index in enumerate(word_positions):
        token = tokens[token_index]
        original = str(token.get("original", ""))
        if token.get("recognized_initial") or len(normalize(original)) < 3:
            continue
        left_context, right_context = _bounded_corpus_context(
            tokens,
            word_positions,
            position_index,
            max_words=10,
        )
        if not left_context and not right_context:
            continue
        choices = list(token.get("choices") or [])
        known_choices = {normalize(str(choice.get("word", ""))) for choice in choices}
        known_choices.add(normalize(str(token.get("corrected", ""))))
        for candidate in CORRECTOR._single_missing_consonant_variants(original):
            if normalize(candidate) in known_choices:
                continue
            evidence = CORPUS_RANKER.window_evidence(
                candidate,
                left_context=left_context,
                right_context=right_context,
                max_distance=10,
            )
            if evidence.contextual_hits <= 0:
                continue
            choices.append({
                "word": candidate,
                "distance": CORRECTOR.weighted_distance(original, candidate),
                "mapped_changes": 1,
                "source": "context_family:single_missing_consonant",
                "authority": "CONTEXT_RESOLVABLE",
                "corpus_score": evidence.score,
                "corpus_unigram": evidence.unigram,
                "corpus_left_bigram": evidence.left_bigram,
                "corpus_right_bigram": evidence.right_bigram,
            })
            known_choices.add(normalize(candidate))
        if choices:
            token["choices"] = choices
            token["ambiguous"] = len(choices) > 1


def _apply_noun_adjective_agreement(tokens: list[dict[str, object]]) -> None:
    if not MORPHOLOGY_RESOLVER.available:
        return

    word_positions = [index for index, token in enumerate(tokens) if token.get("type") == "word"]
    for pair_index in range(len(word_positions) - 1):
        noun_index = word_positions[pair_index]
        adjective_index = word_positions[pair_index + 1]
        separator = "".join(
            str(token.get("text", ""))
            for token in tokens[noun_index + 1 : adjective_index]
            if token.get("type") == "text"
        )
        if not separator or not separator.isspace():
            continue

        noun_token = tokens[noun_index]
        adjective_token = tokens[adjective_index]
        noun = str(noun_token.get("corrected", ""))
        adjective = str(adjective_token.get("corrected", ""))
        agreement = MORPHOLOGY_RESOLVER.agreement_candidates(noun, adjective)
        if not agreement:
            continue

        selected = agreement[0]
        selected_word = _match_case(str(adjective_token.get("original", "")), selected.word)
        corpus_evidence = CORPUS_RANKER.evidence(
            selected.word,
            previous=noun,
        )
        replacement_choice = {
            "word": selected.word,
            "distance": CORRECTOR.weighted_distance(adjective, selected.word),
            "mapped_changes": 1,
            "source": "dictionary_corpus_morphology_agreement",
            "corpus_score": corpus_evidence.score,
            "corpus_unigram": corpus_evidence.unigram,
            "corpus_left_bigram": corpus_evidence.left_bigram,
            "corpus_right_bigram": corpus_evidence.right_bigram,
            "morphology_lemma": selected.lemma,
            "noun_tag": selected.noun_tag,
            "adjective_tag": selected.adjective_tag,
        }
        adjective_token["choices"] = [replacement_choice]
        adjective_token["ambiguous"] = False
        adjective_token["corrected"] = selected_word
        adjective_token["agreement_corrected"] = True
        adjective_token["unrecognized"] = False


def _suppress_corpus_dominated_choices(tokens: list[dict[str, object]]) -> None:
    """Hide alternatives whose selected choice has at least 1.8x its evidence."""
    for token in tokens:
        choices = token.get("choices")
        if not isinstance(choices, list) or len(choices) < 2:
            continue
        primary_score = float(choices[0].get("corpus_score") or 0.0)
        if primary_score <= 0.0:
            continue
        retained = [choices[0]]
        for alternative in choices[1:]:
            if alternative.get("source") == "dictionary_compound_article_alternative":
                retained.append(alternative)
                continue
            alternative_score = float(alternative.get("corpus_score") or 0.0)
            if primary_score >= 1.8 * alternative_score:
                continue
            retained.append(alternative)
        token["choices"] = retained
        token["ambiguous"] = len(retained) > 1


def _apply_verb_context_agreement(tokens: list[dict[str, object]]) -> None:
    if not VERB_CONTEXT_RESOLVER.available:
        return

    word_positions = [index for index, token in enumerate(tokens) if token.get("type") == "word"]
    for position_index, token_index in enumerate(word_positions):
        if position_index == 0:
            continue
        token = tokens[token_index]
        verb = str(token.get("corrected", ""))
        previous_token = tokens[word_positions[position_index - 1]]
        previous_word = str(previous_token.get("corrected", ""))

        previous_norm = normalize(previous_word)
        fused_prefix = re.split(r"[-'’]", previous_norm, maxsplit=1)[0]
        if (
            ("-" in previous_norm or "'" in previous_norm)
            and fused_prefix in {
                "lil", "lill", "bil", "fil", "mill", "mal", "tal",
                "għall", "bħall", "min", "minn", "b", "f", "ma",
            }
        ):
            continue

        # A noun/adjective introduced by a preposition is a complement, not a
        # subject anchor for the following verb (lill-pubbliku jattendi). Do
        # not inherit person from an earlier matrix verb across that phrase.
        if position_index >= 2:
            complement_prefix = tokens[word_positions[position_index - 2]]
            if str(complement_prefix.get("article_tag", "")).startswith("PREP-"):
                continue

        separator = "".join(
            str(part.get("text", ""))
            for part in tokens[word_positions[position_index - 1] + 1 : token_index]
            if part.get("type") == "text"
        )
        resolved = None
        clause_boundaries = {
            "għax", "jekk", "illi", "li", "ma", "meta", "mhux",
            "sakemm", "wara", "qabel", "biex", "u", "jew", "imma",
        }
        if (
            separator.isspace()
            and not VERB_CONTEXT_RESOLVER.has_noun_reading(previous_word)
            and VERB_CONTEXT_RESOLVER.has_verb_reading(previous_word)
            and normalize(previous_word) not in clause_boundaries
        ):
            resolved = VERB_CONTEXT_RESOLVER.resolve_after_third_person_verb(previous_word, verb)

        # A finite verb can remain the subject anchor across a short nominal
        # complement (tifel wasal id-dar issib -> ... isib). Search only within
        # the same punctuation-free clause and reject noun/verb homographs.
        if resolved is None:
            for earlier_position in range(position_index - 2, max(-1, position_index - 7), -1):
                earlier_index = word_positions[earlier_position]
                between = "".join(
                    str(part.get("text", ""))
                    for part in tokens[earlier_index + 1 : token_index]
                    if part.get("type") == "text"
                )
                if re.search(r"[.!?;:\"“”]", between):
                    break
                earlier_word = str(tokens[earlier_index].get("corrected", ""))
                if normalize(earlier_word) in clause_boundaries:
                    break
                if VERB_CONTEXT_RESOLVER.has_noun_reading(earlier_word):
                    break
                if not VERB_CONTEXT_RESOLVER.has_verb_reading(earlier_word):
                    continue
                resolved = VERB_CONTEXT_RESOLVER.resolve_after_third_person_verb(
                    earlier_word,
                    verb,
                )
                if resolved is not None:
                    break
        if resolved is None and normalize(previous_word) == "qed" and position_index >= 2:
            subject_token = tokens[word_positions[position_index - 2]]
            subject = str(subject_token.get("corrected", ""))
            resolved = VERB_CONTEXT_RESOLVER.resolve_after_subject(subject, verb)
        if resolved is None:
            continue

        evidence = CORPUS_RANKER.evidence(resolved.word, previous=previous_word)
        current_evidence = CORPUS_RANKER.evidence(verb, previous=previous_word)
        # Dictionary homographs can assign the wrong person to the preceding
        # verb (for example common kellu versus an unrelated tagged lexeme).
        # Preserve the current verb whenever its local corpus pair is at least
        # as well attested as the proposed agreement rewrite.
        if (
            current_evidence.contextual_hits > 0
            and current_evidence.contextual_score >= evidence.contextual_score
        ):
            continue
        _set_contextual_choice(
            token,
            resolved.word,
            source="dictionary_corpus_verb_context_agreement",
            evidence=evidence,
            metadata={
                "verb_context_reason": resolved.reason,
                "current_person": resolved.current_person,
                "target_person": resolved.target_person,
                "verb_key": "|".join(resolved.verb_key),
            },
        )
        token["verb_context_corrected"] = True


def _apply_verb_epenthesis_surface(tokens: list[dict[str, object]]) -> None:
    """Choose dictionary-backed j-/i-/zero verb surfaces from local phonology."""
    if not VERB_CONTEXT_RESOLVER.available:
        return

    vowels = set("aeiouàèìòù")
    word_positions = [index for index, token in enumerate(tokens) if token.get("type") == "word"]
    for position_index, token_index in enumerate(word_positions):
        token = tokens[token_index]
        current = normalize(str(token.get("corrected", "")))
        original = normalize(str(token.get("original", "")))
        previous_index = word_positions[position_index - 1] if position_index else None
        previous = (
            normalize(str(tokens[previous_index].get("corrected", "")))
            if previous_index is not None
            else ""
        )
        separator = (
            "".join(
                str(part.get("text", ""))
                for part in tokens[previous_index + 1 : token_index]
                if part.get("type") == "text"
            )
            if previous_index is not None
            else ""
        )
        begins_after_punctuation = previous_index is None or bool(re.search(r"[.!?,;:]", separator))
        previous_phonological = previous.rstrip("'’")
        previous_ends_vowel = bool(
            previous_phonological and previous_phonological[-1:] in vowels
        )
        needs_epenthesis = begins_after_punctuation or not previous_ends_vowel

        target = ""
        lexical = ""
        # A contextual agreement repair may have produced the dictionary jK
        # form; surface it as iK where phonology requires the epenthetic vowel.
        if current.startswith("rġ") and VERB_CONTEXT_RESOLVER.has_verb_reading(current):
            lexical = current
            target = "e" + current if needs_epenthesis else current
        elif (
            current.startswith("j")
            and len(current) > 1
            and current[1] not in vowels
            and not current.startswith("jgħ")
            and VERB_CONTEXT_RESOLVER.has_verb_reading(current)
        ):
            lexical = current
            target = "i" + current[1:] if needs_epenthesis else current
        else:
            # Preserve the user's iK surface when it maps to either a jK verb
            # or an i-less CC verb. This also prevents an early single-word
            # candidate from erasing a valid contextual epenthetic vowel.
            input_surface = original if original.startswith("i") else current
            if (
                input_surface.startswith("i")
                and len(input_surface) > 2
                and input_surface[1] not in vowels
            ):
                j_form = "j" + input_surface[1:]
                bare_form = input_surface[1:]
                if VERB_CONTEXT_RESOLVER.has_verb_reading(j_form):
                    lexical = j_form
                    target = input_surface if needs_epenthesis else j_form
                elif VERB_CONTEXT_RESOLVER.has_verb_reading(bare_form):
                    lexical = bare_form
                    target = input_surface if needs_epenthesis else bare_form

        if not target or normalize(str(token.get("corrected", ""))) == target:
            continue
        evidence = CORPUS_RANKER.evidence(target, previous=previous)
        _set_contextual_choice(
            token,
            target,
            source="dictionary_verb_epenthesis_surface",
            evidence=evidence,
            metadata={"lexical_verb_surface": lexical, "epenthesis_required": needs_epenthesis},
        )
        token["verb_epenthesis_corrected"] = True


def _apply_day_number_context(tokens: list[dict[str, object]]) -> None:
    word_positions = [index for index, token in enumerate(tokens) if token.get("type") == "word"]
    for position_index, token_index in enumerate(word_positions):
        token = tokens[token_index]
        current = str(token.get("corrected", ""))
        if normalize(current) != "tnejn":
            continue
        previous_word = (
            str(tokens[word_positions[position_index - 1]].get("corrected", ""))
            if position_index > 0
            else ""
        )
        next_word = (
            str(tokens[word_positions[position_index + 1]].get("corrected", ""))
            if position_index + 1 < len(word_positions)
            else ""
        )
        previous_norm = normalize(previous_word)
        next_norm = normalize(next_word)
        wants_day = previous_norm in {"nhar", "it", "il"} or previous_norm.endswith("-")
        wants_number = next_norm.startswith("min") or next_norm in {"nies", "persuni", "tfal"}
        replacement = "Tnejn" if wants_day and not wants_number else "tnejn"
        if current == replacement:
            continue
        evidence = CORPUS_RANKER.evidence(replacement, previous=previous_word, following=next_word)
        token["choices"] = [{
            "word": replacement,
            "distance": 0.0,
            "mapped_changes": 0,
            "source": "dictionary_context_day_number_case",
            "corpus_score": evidence.score,
            "corpus_unigram": evidence.unigram,
            "corpus_left_bigram": evidence.left_bigram,
            "corpus_right_bigram": evidence.right_bigram,
            "case_context": "day_name" if replacement == "Tnejn" else "cardinal_number",
        }]
        token["ambiguous"] = False
        token["corrected"] = replacement
        token["unrecognized"] = False


def _set_contextual_choice(
    token: dict[str, object],
    replacement: str,
    *,
    source: str,
    evidence: object,
    metadata: dict[str, object],
) -> None:
    original_corrected = str(token.get("corrected", ""))
    if normalize(original_corrected) == normalize(replacement):
        token["unrecognized"] = False
        return
    replacement_choice = {
        "word": replacement,
        "distance": CORRECTOR.weighted_distance(original_corrected, replacement),
        "mapped_changes": 1,
        "source": source,
        "corpus_score": evidence.score,
        "corpus_unigram": evidence.unigram,
        "corpus_left_bigram": evidence.left_bigram,
        "corpus_right_bigram": evidence.right_bigram,
        **metadata,
    }
    token["choices"] = [replacement_choice]
    token["ambiguous"] = False
    token["corrected"] = _match_case(str(token.get("original", "")), replacement)
    token["unrecognized"] = False


def _apply_attributive_numeral_agreement(tokens: list[dict[str, object]]) -> None:
    if not NUMERAL_RESOLVER.available:
        return
    word_positions = [index for index, token in enumerate(tokens) if token.get("type") == "word"]
    for pair_index in range(len(word_positions) - 1):
        numeral_index = word_positions[pair_index]
        noun_index = word_positions[pair_index + 1]
        separator = "".join(
            str(token.get("text", ""))
            for token in tokens[numeral_index + 1 : noun_index]
            if token.get("type") == "text"
        )
        if not separator or not separator.isspace():
            continue

        numeral_token = tokens[numeral_index]
        noun_token = tokens[noun_index]
        resolved = NUMERAL_RESOLVER.resolve(
            str(numeral_token.get("corrected", "")),
            str(noun_token.get("corrected", "")),
        )
        if resolved is None:
            continue

        numeral_evidence = CORPUS_RANKER.evidence(resolved.numeral, following=resolved.noun)
        noun_evidence = CORPUS_RANKER.evidence(resolved.noun, previous=resolved.numeral)
        shared_metadata = {
            "noun_base": resolved.noun_base,
            "noun_tag": resolved.noun_tag,
            "numeral_rule": "LONGATTNUM+iKK",
        }
        _set_contextual_choice(
            numeral_token,
            resolved.numeral,
            source="dictionary_corpus_attributive_numeral",
            evidence=numeral_evidence,
            metadata=shared_metadata,
        )
        _set_contextual_choice(
            noun_token,
            resolved.noun,
            source="dictionary_corpus_attributive_numeral",
            evidence=noun_evidence,
            metadata=shared_metadata,
        )
        numeral_token["numeral_agreement_corrected"] = True
        noun_token["numeral_agreement_corrected"] = True


def _apply_dictionary_compound_resolution(tokens: list[dict[str, object]]) -> None:
    """Join adjacent words when their exact concatenation is a lexical adverb.

    This covers lexicalized time expressions such as bil lejl -> billejl while
    leaving arbitrary noun pairs alone.  When the same surface can also be
    read as a preposition plus article, that parsed form remains a suggestion.
    """
    word_positions = [index for index, token in enumerate(tokens) if token.get("type") == "word"]
    for pair_index in range(len(word_positions) - 1):
        left_index = word_positions[pair_index]
        right_index = word_positions[pair_index + 1]
        left_token = tokens[left_index]
        right_token = tokens[right_index]
        if left_token.get("compound_consumed") or right_token.get("compound_consumed"):
            continue
        separator_indexes = [
            index
            for index in range(left_index + 1, right_index)
            if tokens[index].get("type") == "text"
        ]
        separator = "".join(str(tokens[index].get("text", "")) for index in separator_indexes)
        if not separator or not separator.isspace():
            continue

        corrected_left = str(left_token.get("corrected", ""))
        corrected_right = str(right_token.get("corrected", ""))
        surface_pairs = (
            (
                str(left_token.get("original", corrected_left)),
                str(right_token.get("original", corrected_right)),
            ),
            (corrected_left, corrected_right),
        )
        lexical_pair = next(
            (
                (left, right, normalize(left + right))
                for left, right in surface_pairs
                if any(
                    "ADVERB" in tag
                    for tag in ARTICLE_RESOLVER.tags.get(normalize(left + right), set())
                )
            ),
            None,
        )
        if lexical_pair is None:
            continue
        left, right, joined_key = lexical_pair
        surfaces = CORRECTOR.dictionary.get(joined_key, ())
        joined = surfaces[0] if surfaces else left + right
        joined = _match_case(str(left_token.get("original", "")), joined)
        joined_evidence = CORPUS_RANKER.evidence(joined)
        choices: list[dict[str, object]] = [{
            "word": joined,
            "distance": CORRECTOR.weighted_distance(left + right, joined),
            "mapped_changes": 1,
            "source": "dictionary_compound_adverb",
            "corpus_score": joined_evidence.score,
            "corpus_unigram": joined_evidence.unigram,
            "corpus_left_bigram": joined_evidence.left_bigram,
            "corpus_right_bigram": joined_evidence.right_bigram,
        }]

        article_resolution = ARTICLE_RESOLVER.resolve(left, right)
        if article_resolution is not None:
            parsed = article_resolution.joined_prefix + right
            if normalize(parsed) != normalize(joined):
                choices.append({
                    "word": parsed,
                    "distance": 1.0,
                    "mapped_changes": 1,
                    "source": "dictionary_compound_article_alternative",
                    "corpus_score": article_resolution.corpus_evidence.score,
                    "corpus_unigram": article_resolution.corpus_evidence.unigram,
                    "corpus_left_bigram": article_resolution.corpus_evidence.left_bigram,
                    "corpus_right_bigram": article_resolution.corpus_evidence.right_bigram,
                })

        left_token["corrected"] = joined
        left_token["choices"] = choices
        left_token["ambiguous"] = len(choices) > 1
        left_token["unrecognized"] = False
        left_token["dictionary_compound_corrected"] = True
        right_token["corrected"] = ""
        right_token["choices"] = []
        right_token["unrecognized"] = False
        right_token["compound_consumed"] = True
        for separator_index in separator_indexes:
            tokens[separator_index]["text"] = ""


def _apply_article_phrase_resolution(tokens: list[dict[str, object]]) -> None:
    word_positions = [index for index, token in enumerate(tokens) if token.get("type") == "word"]
    for pair_index in range(len(word_positions) - 1):
        prefix_index = word_positions[pair_index]
        noun_index = word_positions[pair_index + 1]
        separator_indexes = [
            index
            for index in range(prefix_index + 1, noun_index)
            if tokens[index].get("type") == "text"
        ]
        separator = "".join(str(tokens[index].get("text", "")) for index in separator_indexes)
        explicit_hyphen = bool(re.fullmatch(r"\s*-\s*", separator))
        if not separator or not (separator.isspace() or explicit_hyphen):
            continue

        prefix_token = tokens[prefix_index]
        noun_token = tokens[noun_index]
        directional_prefix = normalize(str(prefix_token.get("original", ""))).removesuffix("-")
        directional_complement = normalize(str(noun_token.get("corrected", "")))
        if (
            directional_prefix in {"l", "il", "'l", "'il"}
            and _licensed_short_l_forms(directional_complement)
        ):
            short_forms = _licensed_short_l_forms(directional_complement)
            previous_for_short = (
                str(tokens[word_positions[pair_index - 1]].get("corrected", ""))
                if pair_index > 0
                else None
            )
            selected_short = max(
                short_forms,
                key=lambda form: CORPUS_RANKER.evidence(
                    form,
                    previous=previous_for_short,
                    following=directional_complement,
                ).score,
            )
            prefix_token["corrected"] = selected_short
            prefix_token["choices"] = [{
                "word": selected_short,
                "distance": 0.0,
                "mapped_changes": 1,
                "source": "articles_dictionary_short_l",
                "separator_after": " ",
            }]
            prefix_token["ambiguous"] = False
            prefix_token["unrecognized"] = False
            prefix_token["article_phrase_corrected"] = True
            for separator_index in separator_indexes:
                tokens[separator_index]["text"] = " "
            continue
        if explicit_hyphen and re.fullmatch(
            r"\d+(?::\d+)?(?:am|pm)?",
            str(noun_token.get("corrected", "")),
            re.IGNORECASE,
        ):
            # Explicit article/preposition + numeric expressions retain the
            # written article form. Numeric initials have no Maltese sun-letter
            # assimilation (tad-9am must not become t-għad-9am).
            prefix_surface = str(prefix_token.get("original", "")).removesuffix("-") + "-"
            prefix_token["corrected"] = prefix_surface
            prefix_token["choices"] = [{
                "word": prefix_surface,
                "distance": 0.0,
                "mapped_changes": 0,
                "source": "numeric_article_surface_preservation",
                "separator_after": "",
            }]
            prefix_token["ambiguous"] = False
            prefix_token["unrecognized"] = False
            prefix_token["article_phrase_corrected"] = True
            for separator_index in separator_indexes:
                tokens[separator_index]["text"] = ""
            continue
        if (
            normalize(str(prefix_token.get("corrected", ""))) in {"lil", "lill"}
            and CORRECTOR._is_possessive_noun(str(noun_token.get("corrected", "")))
        ):
            prefix_token["corrected"] = _match_case(
                str(prefix_token.get("original", "")),
                "lil",
            )
            prefix_token["choices"] = [
                {
                    "word": "lil",
                    "distance": 0.0,
                    "mapped_changes": 1,
                    "source": "possessive_family_lil_surface",
                }
            ]
            prefix_token["ambiguous"] = False
            prefix_token["unrecognized"] = False
            continue
        previous = None
        after = None
        if pair_index > 0:
            previous = str(tokens[word_positions[pair_index - 1]].get("corrected", ""))
        if pair_index + 2 < len(word_positions):
            after = str(tokens[word_positions[pair_index + 2]].get("corrected", ""))

        raw_prefix = normalize(str(prefix_token.get("original", "")))
        mapped_raw_prefix = (
            raw_prefix.replace("gh", "\x00")
            .replace("c", "ċ")
            .replace("g", "ġ")
            .replace("h", "ħ")
            .replace("z", "ż")
            .replace("\x00", "għ")
        )
        prefix_surfaces: list[str] = []
        for surface in (
            raw_prefix,
            mapped_raw_prefix,
            str(prefix_token.get("corrected", "")),
        ):
            candidate = surface + ("-" if explicit_hyphen else "")
            if candidate not in prefix_surfaces:
                prefix_surfaces.append(candidate)

        resolution = None
        for prefix_surface in prefix_surfaces:
            resolution = ARTICLE_RESOLVER.resolve(
                prefix_surface,
                str(noun_token.get("corrected", "")),
                previous=previous,
                after=after,
            )
            if resolution is not None:
                break
        if resolution is None:
            continue

        joined_choice = {
            "word": resolution.joined_prefix,
            "distance": 0.0,
            "mapped_changes": 0,
            "source": "articles_dictionary_span",
            "separator_after": "",
            "article_tag": resolution.article_tag,
            "corpus_score": resolution.corpus_evidence.score,
            "corpus_unigram": resolution.corpus_evidence.unigram,
            "corpus_left_bigram": resolution.corpus_evidence.left_bigram,
            "corpus_right_bigram": resolution.corpus_evidence.right_bigram,
        }
        separate_choice = {
            "word": resolution.separate_prefix,
            "distance": 0.0,
            "mapped_changes": 0,
            "source": "article_span_keep",
            "separator_after": " ",
            "article_tag": "NO",
            "corpus_right_bigram": resolution.separate_bigram,
        }

        if resolution.ambiguous and not explicit_hyphen:
            prefix_token["corrected"] = _match_case(
                str(prefix_token.get("original", "")),
                resolution.separate_prefix,
            )
            prefix_token["choices"] = [separate_choice, joined_choice]
        else:
            prefix_token["corrected"] = _match_case(
                str(prefix_token.get("original", "")),
                resolution.joined_prefix,
            )
            prefix_token["choices"] = (
                [joined_choice, separate_choice] if resolution.ambiguous else [joined_choice]
            )
            for separator_index in separator_indexes:
                tokens[separator_index]["text"] = ""

        prefix_token["ambiguous"] = resolution.ambiguous
        prefix_token["article_phrase_corrected"] = True
        prefix_token["unrecognized"] = False


def _apply_candidate_quality_guards(tokens: list[dict[str, object]]) -> None:
    """Remove suggestions contradicted by a strong local POS reading."""
    word_positions = [index for index, token in enumerate(tokens) if token.get("type") == "word"]
    speculative_suffix_sources = {"basedics_verb_a_ha_suffix", "basedics_suffix_generator"}
    for position_index, token_index in enumerate(word_positions):
        token = tokens[token_index]
        choices = token.get("choices")
        if not isinstance(choices, list) or not choices:
            continue

        minimum_distance = min(float(choice.get("distance", 999.0)) for choice in choices)
        choices = [
            choice
            for choice in choices
            if float(choice.get("distance", 999.0)) <= minimum_distance + 0.75
            or choice.get("authority") == "CONTEXT_RESOLVABLE"
            or str(choice.get("source", "")).startswith("context_family:")
            or choice.get("source") in {
                "manual_context_alternative",
                "homophone_phonological_alternative",
                "dictionary_compound_article_alternative",
            }
        ]

        current = normalize(str(token.get("corrected", "")))
        tags = ARTICLE_RESOLVER.tags.get(current, set())
        previous = (
            tokens[word_positions[position_index - 1]]
            if position_index > 0
            else None
        )
        previous_surface = normalize(str(previous.get("corrected", ""))) if previous else ""
        following = (
            tokens[word_positions[position_index + 1]]
            if position_index + 1 < len(word_positions)
            else None
        )
        following_surface = normalize(str(following.get("corrected", ""))) if following else ""
        following_separator = (
            "".join(
                str(part.get("text", ""))
                for part in tokens[token_index + 1 : word_positions[position_index + 1]]
                if part.get("type") == "text"
            )
            if following is not None
            else ""
        )
        nominal_context = (
            any(tag in {"PRON", "SINGNOUNM", "SINGNOUNF", "PLUNOUN", "COLLNOUN"} for tag in tags)
            and (
                previous_surface.endswith("-")
                or previous_surface in {"dan", "din", "dawk"}
            )
        )

        filtered: list[dict[str, object]] = []
        has_l_hyphen_nominal = any(
            normalize(str(choice.get("word", ""))).startswith("l-")
            for choice in choices
        )
        for choice in choices:
            source = str(choice.get("source", ""))
            choice_surface = normalize(str(choice.get("word", "")))
            if choice_surface.startswith("l'") and has_l_hyphen_nominal:
                tail_tags = ARTICLE_RESOLVER.tags.get(choice_surface[2:], set())
                if any(any(marker in tag for marker in ("NOUN", "ADJ")) for tag in tail_tags):
                    continue
            if current == "ma" and choice_surface == "ma'" and following_surface:
                follows_verb = VERB_CONTEXT_RESOLVER.has_verb_reading(following_surface)
                if CORRECTOR.suffix_engine is not None:
                    follows_verb = follows_verb or CORRECTOR.suffix_engine.exact_surface_variant(
                        following_surface
                    )
                if follows_verb:
                    continue
            if (
                current == "ta"
                and choice_surface == "ta'"
                and (
                    following is None
                    or bool(re.search(r"[.!?;:\"“”]", following_separator))
                )
            ):
                continue
            if nominal_context and source in speculative_suffix_sources:
                continue
            if token.get("recognized_initial") and source in speculative_suffix_sources:
                candidate = normalize(str(choice.get("word", "")))
                if CORPUS_RANKER.unigrams.get(candidate, 0.0) <= 0.0:
                    continue
            filtered.append(choice)
        token["choices"] = filtered
        token["ambiguous"] = len(filtered) > 1
        if filtered and all(
            normalize(str(choice.get("word", ""))) != current
            for choice in filtered
        ):
            token["corrected"] = _match_case(
                str(token.get("original", "")),
                str(filtered[0].get("word", "")),
            )
            token["unrecognized"] = False


def _tag_lookup_keys(surface: str) -> list[str]:
    norm = normalize(surface).strip()
    keys = [norm]
    for delimiter in ("-", "'", "’"):
        if delimiter in norm:
            tail = norm.rsplit(delimiter, 1)[-1]
            if tail and tail not in keys:
                keys.append(tail)
    return [key for key in keys if key]


def _is_plural_or_collective_noun(surface: str) -> bool:
    for key in _tag_lookup_keys(surface):
        tags = ARTICLE_RESOLVER.tags.get(key, set())
        if any("PLUNOUN" in tag or "COLLNOUN" in tag for tag in tags):
            return True
    return False


def _looks_ha_suffixed_verb(surface: str) -> bool:
    norm = normalize(surface)
    return len(norm) >= 4 and norm.endswith("ha")


def _apply_plural_object_suffix_guard(tokens: list[dict[str, object]]) -> None:
    word_positions = [index for index, token in enumerate(tokens) if token.get("type") == "word"]
    for position_index, token_index in enumerate(word_positions[:-1]):
        token = tokens[token_index]
        next_token = tokens[word_positions[position_index + 1]]
        if not _is_plural_or_collective_noun(str(next_token.get("corrected", ""))):
            continue
        choices = token.get("choices")
        if not isinstance(choices, list) or not choices:
            continue
        filtered = [
            choice for choice in choices
            if not _looks_ha_suffixed_verb(str(choice.get("word", "")))
        ]
        if len(filtered) == len(choices):
            continue
        token["choices"] = filtered
        if _looks_ha_suffixed_verb(str(token.get("corrected", ""))) and filtered:
            token["corrected"] = _match_case(
                str(token.get("original", "")),
                str(filtered[0].get("word", "")),
            )
        token["ambiguous"] = len(filtered) > 1


def _word_token_from_surface(
    surface: str,
    *,
    source_continuation: bool = False,
) -> dict[str, object]:
    input_was_known = CORRECTOR.is_known(surface)
    corrected, candidates, recognized = CORRECTOR.correct_word(surface)
    choices = [candidate_payload(candidate) for candidate in candidates]
    if choices and all(normalize(choice.get("word", "")) != normalize(corrected) for choice in choices):
        choices.insert(
            0,
            {
                "word": corrected,
                "distance": 0.0,
                "mapped_changes": 0,
                "source": "original_known_surface",
            },
        )
    return {
        "type": "word",
        "original": surface,
        "corrected": corrected,
        "ambiguous": len(choices) > 1,
        "choices": choices,
        "authority": candidates[0].authority if candidates else "SUGGESTION_ONLY",
        "crucial": False,
        "unrecognized": not recognized,
        "recognized_initial": input_was_known,
        "source_continuation": source_continuation,
    }


def _has_same_sentence_neighbor(tokens: list[dict[str, object]], token_index: int) -> bool:
    for direction in (-1, 1):
        cursor = token_index + direction
        while 0 <= cursor < len(tokens):
            part = tokens[cursor]
            if part.get("type") == "text":
                if _CORPUS_CONTEXT_BOUNDARY_RE.search(str(part.get("text", ""))):
                    break
            elif part.get("type") == "word":
                return True
            cursor += direction
    return False


def _next_same_sentence_word(
    tokens: list[dict[str, object]],
    token_index: int,
) -> dict[str, object] | None:
    for cursor in range(token_index + 1, len(tokens)):
        part = tokens[cursor]
        if part.get("type") == "text":
            if _CORPUS_CONTEXT_BOUNDARY_RE.search(str(part.get("text", ""))):
                return None
        elif part.get("type") == "word":
            return part
    return None


def _apply_fused_function_word_resolution(
    tokens: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Split fused particles and expose dictionary-backed apostrophe readings."""
    rebuilt: list[dict[str, object]] = []
    for token_index, token in enumerate(tokens):
        if token.get("type") != "word":
            rebuilt.append(token)
            continue

        original = str(token.get("original", ""))
        norm = normalize(original)
        split_applied = False

        # Expand fused l through the tagged SHORTDEFPREP/ISHORTDEFPREP family.
        # Structural compatibility filters that family before corpus ranking.
        if norm.startswith("l") and _licensed_short_l_forms(norm[1:]):
            remainder = original[1:]
            remainder_token = _word_token_from_surface(remainder, source_continuation=True)
            short_forms = _licensed_short_l_forms(
                str(remainder_token.get("corrected", remainder))
            )
            previous_surface = next(
                (
                    str(previous.get("corrected", ""))
                    for previous in reversed(rebuilt)
                    if previous.get("type") == "word"
                ),
                "",
            )
            ranked_forms = sorted(
                short_forms,
                key=lambda form: (
                    -CORPUS_RANKER.evidence(
                        form,
                        previous=previous_surface or None,
                        following=str(remainder_token.get("corrected", remainder)),
                    ).score,
                    len(form),
                    form,
                ),
            )
            selected_short = ranked_forms[0]
            directional = {
                "type": "word",
                "original": original[:1],
                "corrected": selected_short,
                "ambiguous": False,
                "choices": [{
                    "word": selected_short,
                    "distance": 0.0,
                    "mapped_changes": 1,
                    "source": "articles_dictionary_short_l",
                    "authority": "DETERMINISTIC",
                }],
                "authority": "DETERMINISTIC",
                "crucial": False,
                "unrecognized": False,
                "recognized_initial": False,
                "source_continuation": False,
            }
            rebuilt.extend((directional, {"type": "text", "text": " "}, remainder_token))
            continue

        # x contracts before a valid vowel/silent-initial complement. Build
        # the contraction from the corrected remainder so xha -> x'ħa uses
        # the same path as xini -> x'inhi.
        if norm.startswith("x") and len(norm) > 2 and "'" not in norm and "’" not in norm:
            remainder = original[1:]
            remainder_token = _word_token_from_surface(remainder, source_continuation=True)
            remainder_surface = str(remainder_token.get("corrected", ""))
            remainder_norm = normalize(remainder_surface)
            remainder_tags = ARTICLE_RESOLVER.tags.get(remainder_norm, set())
            starts_contracting = remainder_norm.startswith(("a", "e", "i", "o", "u", "għ", "h", "ħ"))
            valid_remainder = bool(remainder_tags) or VERB_CONTEXT_RESOLVER.has_verb_reading(remainder_surface)
            if starts_contracting and valid_remainder:
                contracted = _match_case(original, "x'" + remainder_surface)
                token["corrected"] = contracted
                token["choices"] = [
                    {
                        "word": contracted,
                        "distance": CORRECTOR.weighted_distance(original, contracted),
                        "mapped_changes": 1,
                        "source": "context_family:fused_x_apostrophe",
                        "authority": "DETERMINISTIC",
                    }
                ]
                token["ambiguous"] = False
                token["unrecognized"] = False
                token["fused_context_corrected"] = True
                rebuilt.append(token)
                continue

        if token.get("unrecognized"):
            for prefix, complement_kind in (("ma", "verb"), ("xi", "nominal")):
                if not norm.startswith(prefix) or len(norm) <= len(prefix) + 1:
                    continue
                remainder = original[len(prefix):]
                remainder_token = _word_token_from_surface(remainder, source_continuation=True)
                remainder_surface = str(remainder_token.get("corrected", ""))
                remainder_norm = normalize(remainder_surface)
                if complement_kind == "verb":
                    valid = VERB_CONTEXT_RESOLVER.has_verb_reading(remainder_surface)
                    if CORRECTOR.suffix_engine is not None:
                        valid = valid or CORRECTOR.suffix_engine.exact_surface_variant(remainder_surface)
                else:
                    tags = ARTICLE_RESOLVER.tags.get(remainder_norm, set())
                    valid = any(
                        marker in tag
                        for tag in tags
                        for marker in ("NOUN", "ADJ", "PRON", "NAME", "SNAME", "PLACE")
                    )
                if not valid:
                    continue
                prefix_surface = _match_case(original[: len(prefix)], prefix)
                rebuilt.extend(
                    (
                        _word_token_from_surface(prefix_surface),
                        {"type": "text", "text": " "},
                        remainder_token,
                    )
                )
                split_applied = True
                break
        if split_applied:
            continue

        apostrophe_candidates = APOSTROPHE_FUSED_INDEX.get(
            _fused_apostrophe_skeleton(original),
            (),
        )
        if apostrophe_candidates and _has_same_sentence_neighbor(tokens, token_index):
            choices = list(token.get("choices") or [])
            if not choices:
                choices.append(
                    {
                        "word": str(token.get("corrected", original)),
                        "distance": 0.0,
                        "mapped_changes": 0,
                        "source": "original_known_surface",
                        "authority": "CONTEXT_RESOLVABLE",
                    }
                )
            for candidate in apostrophe_candidates:
                if normalize(candidate) == normalize(str(token.get("corrected", ""))):
                    continue
                if CORRECTOR.weighted_distance(original, candidate) > 3.0:
                    continue
                if any(normalize(choice.get("word", "")) == normalize(candidate) for choice in choices):
                    continue
                choices.append(
                    {
                        "word": candidate,
                        "distance": 0.0,
                        "mapped_changes": 1,
                        "source": "context_family:fused_apostrophe:dictionary_skeleton",
                        "authority": "CONTEXT_RESOLVABLE",
                    }
                )
            next_token = _next_same_sentence_word(tokens, token_index)
            next_people = (
                VERB_CONTEXT_RESOLVER.persons_by_surface.get(
                    normalize(str(next_token.get("corrected", ""))),
                    set(),
                )
                if next_token is not None
                else set()
            )
            current_tags = ARTICLE_RESOLVER.tags.get(normalize(str(token.get("corrected", ""))), set())
            expected_people = {
                VERB_CONTEXT_RESOLVER.NOUN_TO_PERSON[tag]
                for tag in current_tags
                if tag in VERB_CONTEXT_RESOLVER.NOUN_TO_PERSON
            }
            noun_agreement_fails = bool(
                expected_people
                and next_people
                and expected_people.isdisjoint(next_people)
            )
            if noun_agreement_fails:
                pronoun_candidates = [
                    choice for choice in choices[1:]
                    if any(
                        tag.startswith("PRON")
                        for tag in ARTICLE_RESOLVER.tags.get(
                            normalize(str(choice.get("word", ""))),
                            set(),
                        )
                    )
                ]
                if pronoun_candidates:
                    input_tail = norm[1:]
                    pronoun_candidates.sort(
                        key=lambda choice: (
                            0
                            if normalize(str(choice.get("word", ""))).split("'", 1)[-1][:1]
                            == input_tail[:1]
                            else 1,
                            CORRECTOR.weighted_distance(original, str(choice.get("word", ""))),
                            normalize(str(choice.get("word", ""))),
                        )
                    )
                    selected = pronoun_candidates[0]
                    token["corrected"] = _match_case(original, str(selected.get("word", "")))
                    selected["source"] = "dictionary_context_fused_apostrophe_agreement"
                    token["choices"] = [selected]
                    token["ambiguous"] = False
                    token["unrecognized"] = False
                    token["fused_context_corrected"] = True
                    rebuilt.append(token)
                    continue
            token["choices"] = choices
            token["ambiguous"] = len(choices) > 1
        rebuilt.append(token)
    return rebuilt


def _apply_local_syntax_resolution(tokens: list[dict[str, object]]) -> None:
    """Resolve close homographs when the immediate POS frame is decisive."""
    word_positions = [index for index, token in enumerate(tokens) if token.get("type") == "word"]
    clause_connectives = {"għax", "li", "biex", "jekk", "meta", "illi"}
    for position_index, token_index in enumerate(word_positions):
        token = tokens[token_index]
        choices = token.get("choices")
        if not isinstance(choices, list) or len(choices) < 2:
            continue
        original = normalize(str(token.get("original", "")))
        previous = (
            normalize(str(tokens[word_positions[position_index - 1]].get("corrected", "")))
            if position_index > 0
            else ""
        )
        following = (
            str(tokens[word_positions[position_index + 1]].get("corrected", ""))
            if position_index + 1 < len(word_positions)
            else ""
        )
        following_raw_norm = normalize(following)
        following_norm = following_raw_norm.rstrip("-")
        following_tags = (
            ARTICLE_RESOLVER.tags.get(following_raw_norm, set())
            | ARTICLE_RESOLVER.tags.get(following_norm, set())
        )

        selected: dict[str, object] | None = None
        if (
            original == "hu"
            and previous.startswith(("qal", "għid"))
            and any(tag in {"DET", "SINGNOUNM", "SINGNOUNF", "PLUNOUN"} for tag in following_tags)
        ):
            selected = next(
                (
                    choice
                    for choice in choices
                    if normalize(str(choice.get("word", ""))) == "ħu"
                    and VERB_CONTEXT_RESOLVER.has_verb_reading(str(choice.get("word", "")))
                ),
                None,
            )
        elif original == "qalla" and following_norm in clause_connectives:
            selected = next(
                (
                    choice
                    for choice in choices
                    if normalize(str(choice.get("word", ""))) == "qalilha"
                ),
                None,
            )

        if selected is None:
            continue
        token["choices"] = [selected] + [choice for choice in choices if choice is not selected]
        token["corrected"] = _match_case(
            str(token.get("original", "")),
            str(selected.get("word", "")),
        )
        token["ambiguous"] = False
        token["unrecognized"] = False
        token["syntax_resolved"] = True


def _join_apostrophe_particles(tokens: list[dict[str, object]]) -> None:
    """Remove whitespace after a selected apostrophe-final particle."""
    word_positions = [index for index, token in enumerate(tokens) if token.get("type") == "word"]
    for pair_index in range(len(word_positions) - 1):
        left_index = word_positions[pair_index]
        right_index = word_positions[pair_index + 1]
        left = str(tokens[left_index].get("corrected", ""))
        right = str(tokens[right_index].get("corrected", ""))
        if normalize(left) not in {"b'", "f'", "m'", "t'", "x'", "l'"} or not right:
            continue
        separator_indexes = [
            index
            for index in range(left_index + 1, right_index)
            if tokens[index].get("type") == "text"
        ]
        separator = "".join(str(tokens[index].get("text", "")) for index in separator_indexes)
        if not separator or not separator.isspace():
            continue
        for separator_index in separator_indexes:
            tokens[separator_index]["text"] = ""
        tokens[left_index]["apostrophe_phrase_joined"] = True



def _correct_text_once(text: str) -> tuple[str, list[dict[str, object]]]:
    # Canonicalize apostrophe glyphs before tokenization. Otherwise a trailing
    # curly apostrophe is left behind as separator text while a second ASCII
    # apostrophe is generated (ta’ -> ta'’).
    text = unicodedata.normalize("NFC", text).replace("’", "'").replace("‘", "'").replace("`", "'")
    corrected_parts: list[str] = []
    tokens: list[dict[str, object]] = []
    cursor = 0
    for match in WORD_PATTERN.finditer(text):
        if match.start() > cursor:
            separator = text[cursor : match.start()]
            corrected_parts.append(separator)
            tokens.append({"type": "text", "text": separator})

        original = match.group(0)
        word_token = _word_token_from_surface(original)
        corrected_parts.append(str(word_token.get("corrected", "")))
        tokens.append(word_token)

        cursor = match.end()

    if cursor < len(text):
        remainder = text[cursor:]
        corrected_parts.append(remainder)
        tokens.append({"type": "text", "text": remainder})
    tokens = _apply_fused_function_word_resolution(tokens)
    _apply_dictionary_compound_resolution(tokens)
    _apply_article_phrase_resolution(tokens)
    _apply_plural_object_suffix_guard(tokens)
    _add_contextual_missing_consonant_candidates(tokens)
    _apply_corpus_ranking(tokens)
    _apply_local_syntax_resolution(tokens)
    _join_apostrophe_particles(tokens)
    _apply_verb_context_agreement(tokens)
    _apply_verb_epenthesis_surface(tokens)
    _apply_day_number_context(tokens)
    _apply_noun_adjective_agreement(tokens)
    _apply_attributive_numeral_agreement(tokens)
    _apply_candidate_quality_guards(tokens)
    _suppress_corpus_dominated_choices(tokens)

    source_word_total = sum(
        token.get("type") == "word" and not token.get("source_continuation")
        for token in tokens
    )
    at_sentence_start = source_word_total > 1
    for token in tokens:
        if token.get("type") == "text":
            text_val = str(token.get("text", ""))
            if any(p in text_val for p in (".", "!", "?")):
                at_sentence_start = True
        elif token.get("type") == "word":
            corr = str(token.get("corrected", ""))
            if at_sentence_start and corr and corr[:1].islower():
                token["corrected"] = corr[:1].upper() + corr[1:]
            at_sentence_start = False

    corrected_text = "".join(
        str(token.get("corrected", ""))
        if token.get("type") == "word"
        else str(token.get("text", ""))
        for token in tokens
    )
    return corrected_text, tokens


def normalize_final_surface(text: str, *, source_word_count: int | None = None) -> str:
    normalized = re.sub(r"[^\S\r\n]+", " ", text).strip()
    normalized = re.sub(r"[ \t]+([,.?!;:])", r"\1", normalized)
    normalized = re.sub(r"(?<!\d)([,;:])(?=\S)", r"\1 ", normalized)
    normalized = re.sub(r"([?!])(?=[A-Za-zÀ-ſ])", r"\1 ", normalized)
    normalized = re.sub(r"(?<!\.)(\.)(?!\.)(?=[A-Za-zÀ-ſ])", r"\1 ", normalized)
    if not normalized:
        return normalized

    terminal_match = re.search(r"([.?!]+)$", normalized)
    had_terminal_punctuation = terminal_match is not None
    if terminal_match:
        punctuation = terminal_match.group(1)
        if len(set(punctuation)) == 1:
            mark = punctuation[0]
            if mark == ".":
                replacement = "..." if len(punctuation) >= 3 else "."
            elif len(punctuation) == 2:
                replacement = mark * 2
            else:
                replacement = mark
            normalized = normalized[: terminal_match.start()] + replacement

    words = list(WORD_PATTERN.finditer(normalized))
    sentence_word_count = source_word_count if source_word_count is not None else len(words)
    if sentence_word_count > 1 and not had_terminal_punctuation:
        first = words[0]
        first_word = first.group(0)
        capitalized = first_word[:1].upper() + first_word[1:]
        normalized = normalized[: first.start()] + capitalized + normalized[first.end() :] + "."

    return normalize_phonological_vowels_and_articles(normalized)


def normalize_phonological_vowels_and_articles(text: str) -> str:
    """Apply Maltese phonological rules for initial i-/j- epenthesis and s-cluster article assimilation."""
    if "\n" in text or "\r" in text:
        chunks = re.split(r"(\r\n|\r|\n)", text)
        return "".join(
            chunk
            if re.fullmatch(r"\r\n|\r|\n", chunk)
            else normalize_phonological_vowels_and_articles(chunk)
            for chunk in chunks
        )
    tokens = text.split()
    if not tokens:
        return text

    vowels = set("aeiouàèìòùAEIOUÀÈÌÒÙ")
    result = []

    for idx, token in enumerate(tokens):
        prev_token = tokens[idx - 1] if idx > 0 else ""
        prev_core = re.sub(r"[^\wgħġċħżàèìòù'’-]+$", "", prev_token)
        prev_ends_vowel = bool(prev_core and prev_core[-1] in vowels)

        match = re.match(r"^(.+?)([.?!,;:]*)$", token)
        base = match.group(1) if match else token
        trailing = match.group(2) if match else ""

        t_lower = base.lower()
        if t_lower in ("imbagħad", "imbaghad"):
            if prev_ends_vowel:
                base = "mbagħad" if base.islower() else "Mbagħad"
            else:
                base = "imbagħad" if base.islower() else "Imbagħad"
        elif t_lower in ("ikollu", "ikun", "ikunu", "ikollha", "ikollhom"):
            if prev_ends_vowel:
                base = "j" + base[1:]
        elif t_lower in ("jkollu", "jkun", "jkunu", "jkollha", "jkollhom"):
            if not prev_ends_vowel:
                base = "i" + base[1:]

        token = base + trailing

        if t_lower.startswith(("is-st", "is-sp", "is-sk", "is-sm", "is-sn", "is-sf")):
            token = "l-i" + token[3:]
        elif t_lower.startswith(("l-st", "l-sp", "l-sk", "l-sm", "l-sn", "l-sf")):
            token = "l-i" + token[2:]

        result.append(token)

    return " ".join(result)


def _retokenize_preserving_word_metadata(
    text: str,
    previous_tokens: list[dict[str, object]],
) -> list[dict[str, object]]:
    word_tokens = [token for token in previous_tokens if token.get("type") == "word"]
    rebuilt: list[dict[str, object]] = []
    cursor = 0
    word_index = 0
    for match in WORD_PATTERN.finditer(text):
        if match.start() > cursor:
            rebuilt.append({"type": "text", "text": text[cursor : match.start()]})

        surface = match.group(0)
        if word_index < len(word_tokens):
            source_token = word_tokens[word_index]
            if (
                source_token.get("dictionary_compound_corrected")
                and word_index + 1 < len(word_tokens)
                and word_tokens[word_index + 1].get("compound_consumed")
            ):
                token = dict(source_token)
                token["original"] = (
                    str(source_token.get("original", ""))
                    + " "
                    + str(word_tokens[word_index + 1].get("original", ""))
                )
                token["corrected"] = surface
                rebuilt.append(token)
                word_index += 2
                cursor = match.end()
                continue
            if (
                source_token.get("apostrophe_phrase_joined")
                and "'" in surface
                and word_index + 1 < len(word_tokens)
            ):
                apostrophe_index = surface.find("'") + 1
                prefix_token = dict(source_token)
                prefix_token["corrected"] = surface[:apostrophe_index]
                following_token = dict(word_tokens[word_index + 1])
                following_token["corrected"] = surface[apostrophe_index:]
                rebuilt.extend((prefix_token, following_token))
                word_index += 2
                cursor = match.end()
                continue
            if (
                source_token.get("article_phrase_corrected")
                and "-" in surface
                and word_index + 1 < len(word_tokens)
            ):
                prefix, following = surface.split("-", 1)
                prefix_token = dict(source_token)
                prefix_token["corrected"] = prefix + "-"
                following_token = dict(word_tokens[word_index + 1])
                following_token["corrected"] = following
                rebuilt.extend((prefix_token, following_token))
                word_index += 2
                cursor = match.end()
                continue
            token = dict(word_tokens[word_index])
            token["corrected"] = surface
            rebuilt.append(token)
        else:
            rebuilt.append(
                {
                    "type": "word",
                    "original": surface,
                    "corrected": surface,
                    "ambiguous": False,
                    "choices": [],
                    "crucial": False,
                    "unrecognized": False,
                }
            )
        word_index += 1
        cursor = match.end()

    if cursor < len(text):
        rebuilt.append({"type": "text", "text": text[cursor:]})
    return rebuilt


def _apply_final_english_suggestions(
    tokens: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Recognize exact English entries only after Maltese correction is complete."""
    if not EXACT_ENGLISH:
        return tokens

    english_entries = sorted(
        EXACT_ENGLISH.items(),
        key=lambda item: (-len(item[0].split()), -len(item[0]), item[0]),
    )
    rebuilt: list[dict[str, object]] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token.get("type") != "word":
            rebuilt.append(token)
            index += 1
            continue

        matched = False
        for source, target in english_entries:
            source_words = source.split()
            cursor = index
            matched_tokens: list[dict[str, object]] = []
            consumed_end = index
            actual_words: list[str] = []
            for source_index, source_word in enumerate(source_words):
                if cursor >= len(tokens) or tokens[cursor].get("type") != "word":
                    break
                source_token = tokens[cursor]
                original = str(source_token.get("original", ""))
                if normalize(original) != source_word:
                    break
                matched_tokens.append(source_token)
                actual_words.append(original)
                consumed_end = cursor + 1
                if source_index + 1 == len(source_words):
                    continue
                if consumed_end >= len(tokens) or tokens[consumed_end].get("type") != "text":
                    break
                separator = str(tokens[consumed_end].get("text", ""))
                if not separator or not separator.isspace():
                    break
                cursor = consumed_end + 1
            else:
                actual_source = " ".join(actual_words)
                suggestion = target if target and normalize(target) != normalize(actual_source) else ""
                rebuilt.append(
                    {
                        "type": "english_phrase",
                        "original": actual_source,
                        "inner_text": actual_source,
                        "corrected": actual_source,
                        "maltese_suggestion": [suggestion] if suggestion else [],
                        "choices": [],
                        "ambiguous": False,
                        "crucial": False,
                        "unrecognized": False,
                        "authority": "FINAL_ENGLISH_EXACT",
                    }
                )
                index = consumed_end
                matched = True
                break
        if not matched:
            rebuilt.append(token)
            index += 1
    return rebuilt


def correct_text(text: str) -> tuple[str, list[dict[str, object]]]:
    corrected_text, tokens = _correct_text_once(text)
    source_word_count = sum(
        token.get("type") == "word" and not token.get("source_continuation")
        for token in tokens
    )
    normalized_text = normalize_final_surface(
        corrected_text,
        source_word_count=source_word_count,
    )
    if normalized_text != corrected_text:
        tokens = _retokenize_preserving_word_metadata(normalized_text, tokens)
    tokens = _apply_final_english_suggestions(tokens)
    normalized_text = "".join(
        str(token.get("text", ""))
        if token.get("type") == "text"
        else str(token.get("corrected", ""))
        for token in tokens
    )
    return normalized_text, tokens


@app.get("/")
def index():
    return send_from_directory(UI_DIR, "index.html")


@app.get("/health")
def health():
    return jsonify(
        ok=True,
        words=len(BASE_DICTIONARY),
        paradigms=(
            CORRECTOR.suffix_engine.generator.verb_index.record_count()
            if CORRECTOR.suffix_engine is not None
            else 0
        ),
        suffixation_enabled=CORRECTOR.suffix_engine is not None,
        corpus=CORPUS_RANKER.status_payload(),
        morphology=MORPHOLOGY_RESOLVER.status_payload(),
        attributive_numerals=NUMERAL_RESOLVER.status_payload(),
        verb_context=VERB_CONTEXT_RESOLVER.status_payload(),
        context_families=CONTEXT_FAMILY_RESOLVER.status_payload(),
        manualdics_loaded=False,
    )


@app.post("/check-text")
def check_text_route():
    payload = request.get_json(silent=True) or {}
    text = payload.get("text", "")
    if not isinstance(text, str):
        return jsonify(error="text must be a string"), 400
    corrected_text, tokens = correct_text(text)
    return jsonify(corrected_text=corrected_text, tokens=tokens, log_id=None)


@app.post("/suggest-word")
@app.post("/debug-word")
def suggest_word_route():
    payload = request.get_json(silent=True) or {}
    word = payload.get("word", "")
    if not isinstance(word, str) or not word:
        return jsonify(error="word must be a non-empty string"), 400
    corrected, candidates, recognized = CORRECTOR.correct_word(word)
    return jsonify(
        original=word,
        corrected=corrected,
        recognized=recognized,
        candidates=[candidate_payload(candidate) for candidate in candidates],
        manualdics_loaded=False,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
