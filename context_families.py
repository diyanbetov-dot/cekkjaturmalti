from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from pathlib import Path


def normalize(value: str) -> str:
    return unicodedata.normalize("NFC", value or "").casefold()


@dataclass(frozen=True, slots=True)
class ContextFamilyChoice:
    word: str
    family: str
    reason: str


class ContextFamilyResolver:
    """Small, dictionary-backed ambiguity families for corpus ranking.

    These are not direct corrections. They simply expose plausible competing
    forms to the corpus ranker when the written word is already valid or close
    to several valid words.
    """

    BASE_FAMILIES = {
        "hadd_hadt_hatt": ("ħadd", "ħadt", "ħatt"),
        "tnejn_monday_number": ("tnejn", "Tnejn"),
    }

    STRONG_WEAK_PAIRS = (
        ("d", "t"),
        ("b", "p"),
        ("ġ", "k"),
        ("g", "k"),
        ("ż", "s"),
        ("z", "s"),
    )

    def __init__(self, base_dics: Path, *, enabled: bool = True) -> None:
        self.enabled = bool(enabled)
        self.available = False
        self.status = "disabled"
        self.surfaces: dict[str, tuple[str, ...]] = {}
        self.families_by_word: dict[str, tuple[ContextFamilyChoice, ...]] = {}
        if self.enabled:
            self._load(base_dics)

    def _load(self, base_dics: Path) -> None:
        try:
            surfaces: dict[str, set[str]] = {}
            for path in base_dics.glob("*.dic"):
                for line in path.read_text(encoding="utf-8-sig").splitlines():
                    stripped = line.strip()
                    if not stripped or stripped.startswith("#") or "/" not in stripped:
                        continue
                    surface = stripped.split("/", 1)[0].strip()
                    if surface:
                        surfaces.setdefault(normalize(surface), set()).add(surface)
            self.surfaces = {
                key: tuple(sorted(values, key=lambda item: (item.casefold(), item)))
                for key, values in surfaces.items()
            }

            families: dict[str, list[ContextFamilyChoice]] = {}
            for family, words in self.BASE_FAMILIES.items():
                valid = [word for word in words if normalize(word) in self.surfaces]
                for word in valid:
                    for other in valid:
                        if normalize(other) == normalize(word):
                            continue
                        families.setdefault(normalize(word), []).append(
                            ContextFamilyChoice(
                                word=self._surface_for(other),
                                family=family,
                                reason="listed_context_family",
                            )
                        )

            self.families_by_word = {
                key: tuple(values)
                for key, values in families.items()
            }
            self.available = True
            self.status = "ready"
        except OSError as exc:
            self.status = f"unavailable: {exc}"

    def _surface_for(self, word: str) -> str:
        surfaces = self.surfaces.get(normalize(word), ())
        if not surfaces:
            return word
        if word[:1].isupper():
            upper = [surface for surface in surfaces if surface[:1].isupper()]
            if upper:
                return upper[0]
        lower = [surface for surface in surfaces if surface.islower()]
        return lower[0] if lower else surfaces[0]

    def _strong_weak_variants(self, word: str) -> list[ContextFamilyChoice]:
        norm = normalize(word)
        variants: list[ContextFamilyChoice] = []
        for left, right in self.STRONG_WEAK_PAIRS:
            for source, target in ((left, right), (right, left)):
                start = 0
                while True:
                    index = norm.find(source, start)
                    if index < 0:
                        break
                    before = norm[index - 1] if index > 0 else ""
                    after = norm[index + len(source)] if index + len(source) < len(norm) else ""
                    adjacent_consonant = (
                        (before.isalpha() and before not in "aeiouàèìòù")
                        or (after.isalpha() and after not in "aeiouàèìòù")
                    )
                    if not adjacent_consonant:
                        start = index + len(source)
                        continue
                    candidate = norm[:index] + target + norm[index + len(source):]
                    if candidate != norm and candidate in self.surfaces:
                        variants.append(
                            ContextFamilyChoice(
                                word=self._surface_for(candidate),
                                family="strong_weak_consonant",
                                reason=f"{source}->{target}",
                            )
                        )
                    start = index + len(source)
        return variants

    def choices(self, word: str) -> list[ContextFamilyChoice]:
        if not (self.enabled and self.available):
            return []
        norm = normalize(word)
        out: list[ContextFamilyChoice] = list(self.families_by_word.get(norm, ()))
        if len(norm) >= 3:
            out.extend(self._strong_weak_variants(word))
        seen: set[str] = set()
        unique: list[ContextFamilyChoice] = []
        for choice in out:
            key = normalize(choice.word)
            if key == norm or key in seen:
                continue
            seen.add(key)
            unique.append(choice)
        return unique

    def status_payload(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "available": self.available,
            "status": self.status,
            "families": len(self.BASE_FAMILIES),
            "surfaces": len(self.surfaces),
        }
