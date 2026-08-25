from __future__ import annotations

import gzip
import json
import unicodedata
from dataclasses import dataclass
from pathlib import Path


def normalize(value: str) -> str:
    return unicodedata.normalize("NFC", value).casefold()


def parse_tagged_dictionary(path: Path) -> dict[str, set[str]]:
    tags: dict[str, set[str]] = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "/" not in stripped:
            continue
        surface, remainder = stripped.split("/", 1)
        tag = remainder.split("-", 1)[0].strip().upper()
        if surface.strip() and tag:
            tags.setdefault(normalize(surface.strip()), set()).add(tag)
    return tags


@dataclass(frozen=True, slots=True)
class AgreementCandidate:
    word: str
    noun_tag: str
    adjective_tag: str
    lemma: str
    corpus_frequency: float


class MorphologyAgreementResolver:
    EXPECTED_ADJECTIVE = {
        "SINGNOUNF": "SINGADJF",
        "SINGNOUNM": "SINGADJM",
        "PLUNOUN": "PLUADJ",
        "COLLNOUN": "PLUADJ",
    }

    def __init__(self, base_dics: Path, morphology_path: Path, *, enabled: bool = True) -> None:
        self.enabled = bool(enabled)
        self.available = False
        self.status = "disabled"
        self.noun_tags: dict[str, set[str]] = {}
        self.adjective_tags: dict[str, set[str]] = {}
        self.analyses: dict[str, list[dict[str, object]]] = {}
        self.lemma_forms: dict[str, dict[str, list[tuple[str, float]]]] = {}
        if self.enabled:
            self._load(base_dics, morphology_path)

    def _load(self, base_dics: Path, morphology_path: Path) -> None:
        try:
            self.noun_tags = parse_tagged_dictionary(base_dics / "fixednouns.dic")
            self.adjective_tags = parse_tagged_dictionary(base_dics / "maltese_adjectives.dic")
            with gzip.open(morphology_path, "rt", encoding="utf-8") as stream:
                self.analyses = json.load(stream)

            lemma_forms: dict[str, dict[str, dict[str, float]]] = {}
            for surface, analyses in self.analyses.items():
                dictionary_tags = self.adjective_tags.get(surface, set())
                if not dictionary_tags:
                    continue
                for analysis in analyses:
                    if analysis.get("pos") != "adj" or not analysis.get("lemma"):
                        continue
                    lemma = str(analysis["lemma"])
                    frequency = float(analysis.get("frequency", 0.0))
                    for tag in dictionary_tags:
                        if tag not in {"SINGADJM", "SINGADJF", "PLUADJ"}:
                            continue
                        lemma_forms.setdefault(lemma, {}).setdefault(tag, {})[surface] = max(
                            frequency,
                            lemma_forms.get(lemma, {}).get(tag, {}).get(surface, 0.0),
                        )
            self.lemma_forms = {
                lemma: {
                    tag: sorted(forms.items(), key=lambda item: (-item[1], item[0]))
                    for tag, forms in by_tag.items()
                }
                for lemma, by_tag in lemma_forms.items()
            }
            self.available = True
            self.status = "ready"
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            self.status = f"unavailable: {exc}"

    def adjective_lemmas(self, word: str) -> list[str]:
        return sorted({
            str(analysis["lemma"])
            for analysis in self.analyses.get(normalize(word), [])
            if analysis.get("pos") == "adj" and analysis.get("lemma")
        })

    def agreement_candidates(self, noun: str, adjective: str) -> list[AgreementCandidate]:
        if not (self.enabled and self.available):
            return []
        noun_key = normalize(noun)
        adjective_key = normalize(adjective)
        noun_tags = self.noun_tags.get(noun_key, set())
        adjective_tags = self.adjective_tags.get(adjective_key, set())
        expected_pairs = [
            (noun_tag, self.EXPECTED_ADJECTIVE[noun_tag])
            for noun_tag in noun_tags
            if noun_tag in self.EXPECTED_ADJECTIVE
        ]
        if not expected_pairs or any(expected in adjective_tags for _, expected in expected_pairs):
            return []

        candidates: dict[str, AgreementCandidate] = {}
        for lemma in self.adjective_lemmas(adjective_key):
            for noun_tag, expected_tag in expected_pairs:
                for surface, frequency in self.lemma_forms.get(lemma, {}).get(expected_tag, []):
                    if surface == adjective_key:
                        continue
                    candidate = AgreementCandidate(
                        word=surface,
                        noun_tag=noun_tag,
                        adjective_tag=expected_tag,
                        lemma=lemma,
                        corpus_frequency=frequency,
                    )
                    previous = candidates.get(surface)
                    if previous is None or candidate.corpus_frequency > previous.corpus_frequency:
                        candidates[surface] = candidate
        return sorted(candidates.values(), key=lambda item: (-item.corpus_frequency, item.word))

    def status_payload(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "available": self.available,
            "status": self.status,
            "noun_forms": len(self.noun_tags),
            "adjective_forms": len(self.adjective_tags),
            "adjective_lemmas": len(self.lemma_forms),
        }
