from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from pathlib import Path

from morphology_agreement import parse_tagged_dictionary


def normalize(value: str) -> str:
    return unicodedata.normalize("NFC", value or "").casefold()


@dataclass(frozen=True, slots=True)
class VerbFeature:
    surface: str
    family: str
    root: str
    form: str
    tense: str
    person: str

    @property
    def key(self) -> tuple[str, str, str, str]:
        return (self.family, self.root, self.form, self.tense)


@dataclass(frozen=True, slots=True)
class VerbContextCandidate:
    word: str
    current_person: str
    target_person: str
    reason: str
    verb_key: tuple[str, str, str, str]


class VerbContextResolver:
    THIRD_PERSONS = {"3SM", "3SF", "3P"}
    NOUN_TO_PERSON = {
        "SINGNOUNM": "3SM",
        "SINGNOUNF": "3SF",
        "PLUNOUN": "3P",
        "COLLNOUN": "3P",
    }

    def __init__(self, base_dics: Path, *, enabled: bool = True) -> None:
        self.enabled = bool(enabled)
        self.available = False
        self.status = "disabled"
        self.noun_tags = parse_tagged_dictionary(base_dics / "fixednouns.dic")
        self.all_verb_surfaces: set[str] = set()
        self.persons_by_surface: dict[str, set[str]] = {}
        self.features_by_surface: dict[str, list[VerbFeature]] = {}
        self.forms_by_key_person: dict[tuple[str, str, str, str], dict[str, set[str]]] = {}
        if self.enabled:
            self._load([
                base_dics / "verbmt_semitic.dic",
                base_dics / "verbmt_nonsemitic.dic",
            ])

    def _parse_feature(self, surface: str, tag: str) -> VerbFeature | None:
        tag_head = tag.split("-", 1)[0] if "/" in tag else tag
        parts = tag_head.split("-")
        # The dictionaries use either T-root-FORM-TENSE-PERSON or
        # AS-lemma-TENSE-PERSON. Keep this strict so contextual verb correction
        # cannot drift into unrelated lexical edits.
        if len(parts) >= 5 and parts[0] == "T":
            family, root, form, tense, person = parts[:5]
        elif len(parts) >= 4:
            family, root, tense, person = parts[:4]
            form = ""
        else:
            return None
        if tense not in {"MPERF", "PERF"} or person not in self.THIRD_PERSONS:
            return None
        return VerbFeature(
            surface=normalize(surface),
            family=family,
            root=root,
            form=form,
            tense=tense,
            person=person,
        )

    def _load(self, paths: list[Path]) -> None:
        try:
            for path in paths:
                for line in path.read_text(encoding="utf-8-sig").splitlines():
                    stripped = line.strip()
                    if not stripped or stripped.startswith("#") or "/" not in stripped:
                        continue
                    surface, remainder = stripped.split("/", 1)
                    normalized_surface = normalize(surface.strip())
                    self.all_verb_surfaces.add(normalized_surface)
                    tag_parts = remainder.split("-")
                    if tag_parts and tag_parts[0] == "T" and len(tag_parts) >= 5:
                        self.persons_by_surface.setdefault(normalized_surface, set()).add(tag_parts[4])
                    elif len(tag_parts) >= 4:
                        self.persons_by_surface.setdefault(normalized_surface, set()).add(tag_parts[3])
                    feature = self._parse_feature(surface.strip(), remainder.strip())
                    if feature is None:
                        continue
                    self.features_by_surface.setdefault(feature.surface, []).append(feature)
                    self.forms_by_key_person.setdefault(feature.key, {}).setdefault(feature.person, set()).add(
                        feature.surface
                    )
            self.available = True
            self.status = "ready"
        except OSError as exc:
            self.status = f"unavailable: {exc}"

    def _noun_person(self, noun: str) -> str | None:
        tags = self.noun_tags.get(normalize(noun), set())
        possible = {self.NOUN_TO_PERSON[tag] for tag in tags if tag in self.NOUN_TO_PERSON}
        if len(possible) == 1:
            return next(iter(possible))
        return None

    def has_noun_reading(self, surface: str) -> bool:
        """Return whether a surface can be a noun in the local dictionaries."""
        return bool(self.noun_tags.get(normalize(surface)))

    def has_verb_reading(self, surface: str) -> bool:
        return normalize(surface) in self.all_verb_surfaces

    def allows_object_suffix_base(self, surface: str) -> bool:
        """Reject obvious non-base forms such as PERF-1P morna."""
        people = self.persons_by_surface.get(normalize(surface), set())
        return bool(people and any(person != "1P" for person in people))

    def _best_surface(self, feature: VerbFeature, target_person: str) -> str | None:
        if feature.person == target_person:
            return None
        forms = self.forms_by_key_person.get(feature.key, {}).get(target_person, set())
        if not forms:
            return None
        return sorted(forms)[0]

    def resolve_after_subject(self, subject: str, verb: str) -> VerbContextCandidate | None:
        if not (self.enabled and self.available):
            return None
        target_person = self._noun_person(subject)
        if target_person is None:
            return None
        features = [
            feature for feature in self.features_by_surface.get(normalize(verb), [])
            if feature.tense == "MPERF" and feature.person in self.THIRD_PERSONS
        ]
        if not features:
            return None
        for feature in features:
            surface = self._best_surface(feature, target_person)
            if surface:
                return VerbContextCandidate(
                    word=surface,
                    current_person=feature.person,
                    target_person=target_person,
                    reason="subject_mperfect_agreement",
                    verb_key=feature.key,
                )
        return None

    def resolve_after_third_person_verb(self, previous_verb: str, verb: str) -> VerbContextCandidate | None:
        if not (self.enabled and self.available):
            return None
        previous_features = [
            feature for feature in self.features_by_surface.get(normalize(previous_verb), [])
            if feature.person in self.THIRD_PERSONS
        ]
        if not previous_features:
            return None
        target_people = {feature.person for feature in previous_features}
        if len(target_people) != 1:
            return None
        target_person = next(iter(target_people))
        features = [
            feature for feature in self.features_by_surface.get(normalize(verb), [])
            if feature.tense == "MPERF" and feature.person in self.THIRD_PERSONS
        ]
        for feature in features:
            surface = self._best_surface(feature, target_person)
            if surface:
                return VerbContextCandidate(
                    word=surface,
                    current_person=feature.person,
                    target_person=target_person,
                    reason="previous_third_person_verb_agreement",
                    verb_key=feature.key,
                )
        return None

    def status_payload(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "available": self.available,
            "status": self.status,
            "verb_surfaces": len(self.features_by_surface),
            "verb_keys": len(self.forms_by_key_person),
        }
