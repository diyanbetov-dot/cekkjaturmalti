# -*- coding: utf-8 -*-
"""Conservative, reversible final-stage selection using corpus evidence."""

from __future__ import annotations

import logging
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

_LEXICAL_TYPES = {"word", "phrase"}
_VALID_MODES = {"off", "shadow", "active"}
_KINSHIP_GLOSS_PATTERN = re.compile(
    r"\b(?:"
    r"mother|father|son|daughter|brother|sister|uncle|aunt|"
    r"grandmother|grandfather|grandson|granddaughter|"
    r"cousin|nephew|niece|husband|wife|spouse|parent|relative|"
    r"family member"
    r")\b",
    re.IGNORECASE,
)


class CorpusContextSelector:
    """
    Choose among candidates only after the normal correction pipeline.

    The selector is deliberately isolated from candidate generation. Set
    ``SPELLCHECK_CORPUS_MODE=off`` to remove every behavioral effect, or use
    ``shadow`` to retain diagnostics without changing output.
    """

    def __init__(
        self,
        scorer,
        *,
        mode: str = "active",
        minimum_margin: float = 0.012,
        max_candidates: int = 6,
        max_passes: int = 2,
    ) -> None:
        normalized_mode = str(mode or "off").strip().lower()
        self.mode = normalized_mode if normalized_mode in _VALID_MODES else "off"
        self.scorer = scorer
        self.minimum_margin = max(0.0, float(minimum_margin))
        self.max_candidates = max(2, int(max_candidates))
        self.max_passes = max(1, min(2, int(max_passes)))
        self._family_surface_to_base: Optional[Dict[str, str]] = None

    def is_available(self) -> bool:
        return (
            self.mode != "off"
            and self.scorer is not None
            and bool(self.scorer.is_available())
        )

    @staticmethod
    def _surface_words(surface: str) -> List[str]:
        return [
            part.lower()
            for part in re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿĊċĠġĦħŻż']+(?:-[^\s]+)?", str(surface))
            if part
        ]

    def _context_edge(self, surface: Optional[str], *, first: bool) -> Optional[str]:
        if not surface:
            return None
        corpus_tokens = self.scorer._surface_tokens(str(surface))
        if not corpus_tokens:
            return None
        return corpus_tokens[0] if first else corpus_tokens[-1]

    @staticmethod
    def _choice_word(choice) -> str:
        if isinstance(choice, dict):
            return str(choice.get("word", "")).strip()
        return str(choice or "").strip()

    def _candidate_surfaces(self, token: dict, spellchecker) -> List[str]:
        surfaces: List[str] = []

        def add(surface: str) -> None:
            surface = str(surface or "").strip()
            if not surface:
                return
            normalized = spellchecker._normalize_word(surface)
            if not normalized:
                return
            if any(spellchecker._normalize_word(item) == normalized for item in surfaces):
                return
            surfaces.append(surface)

        add(token.get("corrected", ""))
        for choice in token.get("choices", []) or []:
            if (
                isinstance(choice, dict)
                and choice.get("suggestion_kind") == "literal_article"
            ):
                # These are intentionally exposed as semantic alternatives,
                # not automatic rewrites. Their corpus frequency must not
                # displace the canonical contracted surface.
                continue
            add(self._choice_word(choice))

        original = spellchecker._normalize_word(str(token.get("original", "")))
        # Recover a bounded article candidate before a recognized noun that was
        # accidentally consumed as part of one word (for example lomm).
        if (
            " " not in original
            and original.startswith("l")
            and not original.startswith("l-")
            and len(original) > 2
        ):
            tail = original[1:]
            if spellchecker._is_recognized_surface(tail):
                add(f"l-{tail}")

        return surfaces[: self.max_candidates]

    @staticmethod
    def _generated_features(match) -> Tuple[str, str, str, str]:
        return (
            str(getattr(match, "base", "") or "").lower(),
            str(getattr(match, "tense", "") or "").upper(),
            str(getattr(match, "person", "") or "").upper(),
            str(getattr(match, "root", "") or "").lower(),
        )

    def _verb_features(self, surface: str, spellchecker) -> List[Tuple[str, str, str, str]]:
        normalized = spellchecker._normalize_word(surface)
        plain = normalized.rstrip("x")
        features: List[Tuple[str, str, str, str]] = []

        suffix_generator = getattr(spellchecker, "suffix_generator", None)
        verb_index = getattr(suffix_generator, "verb_index", None)
        if verb_index is not None:
            for record in verb_index.word_records(plain):
                features.append(
                    (
                        plain,
                        str(getattr(record, "tense", "") or "").upper(),
                        str(getattr(record, "person", "") or "").upper(),
                        str(getattr(record, "root", "") or "").lower(),
                    )
                )
        for match in spellchecker._exact_suffix_matches_cached(plain):
            features.append(self._generated_features(match))
        return features

    def _family_surfaces(self, spellchecker) -> Dict[str, str]:
        if self._family_surface_to_base is not None:
            return self._family_surface_to_base

        family_bases: set[str] = set()
        for word, tags in spellchecker.word_tags.items():
            if not any("NOUN" in str(tag).upper() for tag in tags):
                continue
            meanings = spellchecker._tag_meanings_for_word(word)
            if any(
                _KINSHIP_GLOSS_PATTERN.search(meaning)
                and "mother-of-pearl" not in meaning.lower()
                for meaning in meanings
            ):
                family_bases.add(spellchecker._normalize_word(word))

        surface_to_base: Dict[str, str] = {}
        for base in family_bases:
            surface_to_base[base] = base
            for surface in spellchecker._noun_possessive_surfaces_for_base(base):
                surface_to_base[spellchecker._normalize_word(surface)] = base
        self._family_surface_to_base = surface_to_base
        return surface_to_base

    def _family_l_details(
        self,
        token: dict,
        spellchecker,
    ) -> Optional[Tuple[str, str, bool, bool]]:
        original = spellchecker._normalize_word(str(token.get("original", "")))
        if "'" in original:
            return None

        source_prefix = ""
        tail = ""
        explicitly_hyphenated = "-" in original
        if original.startswith("il-"):
            source_prefix, tail = "il", original[3:].strip()
        elif original.startswith("l-"):
            source_prefix, tail = "l", original[2:].strip()
        elif original.startswith("il "):
            source_prefix, tail = "il", original[3:].strip()
        elif original.startswith("l "):
            source_prefix, tail = "l", original[2:].strip()
        elif original.startswith("il") and len(original) > 3:
            source_prefix, tail = "il", original[2:]
        elif original.startswith("l") and len(original) > 2:
            source_prefix, tail = "l", original[1:]
        if not source_prefix or not tail or " " in tail:
            return None

        family_surfaces = self._family_surfaces(spellchecker)
        if tail in family_surfaces:
            base = family_surfaces[tail]
            return source_prefix, tail, tail == base, explicitly_hyphenated

        corrected = spellchecker._normalize_word(str(token.get("corrected", "")))
        corrected_tail = re.sub(
            r"^(?:il-|l-|iċ-|id-|in-|ir-|is-|it-|ix-|iż-)",
            "",
            corrected,
        )
        if corrected_tail in family_surfaces:
            base = family_surfaces[corrected_tail]
            return (
                source_prefix,
                corrected_tail,
                corrected_tail == base,
                explicitly_hyphenated,
            )

        corrected_tail = spellchecker._normalize_word(
            spellchecker.correct_word(tail)
        )
        if corrected_tail in family_surfaces:
            base = family_surfaces[corrected_tail]
            return (
                source_prefix,
                corrected_tail,
                corrected_tail == base,
                explicitly_hyphenated,
            )
        return None

    def _apply_family_l_rule(
        self,
        token: dict,
        spellchecker,
        *,
        next_surface: Optional[str],
    ) -> bool:
        details = self._family_l_details(token, spellchecker)
        if details is None:
            return False

        source_prefix, tail, is_base_form, explicitly_hyphenated = details
        next_normalized = spellchecker._normalize_word(next_surface or "")
        followed_by_ta = next_normalized == "ta'" or next_normalized.startswith(
            ("ta' ", "ta'-", "t'")
        )
        if explicitly_hyphenated and is_base_form and not followed_by_ta:
            return False

        directional_prefix = "'il" if source_prefix == "il" else "'l"
        article_surface = f"{source_prefix}-{tail}"
        spaced_surface = f"{directional_prefix} {tail}"
        hyphenated_surface = f"{directional_prefix}-{tail}"
        old_surface = str(token.get("corrected", ""))
        preferred_surface = (
            hyphenated_surface
            if is_base_form and followed_by_ta
            else spaced_surface
        )
        winner = self._preserve_initial_case(old_surface, preferred_surface)

        choices = []
        for surface in (winner, article_surface, spaced_surface, hyphenated_surface):
            displayed = self._preserve_initial_case(old_surface, surface)
            if any(
                spellchecker._normalize_word(choice["word"])
                == spellchecker._normalize_word(displayed)
                for choice in choices
            ):
                continue
            choices.append(
                {
                    "word": displayed,
                    "meaning": "",
                    "suggestion_kind": "family_l",
                }
            )
            if len(choices) == 3:
                break

        token["corpus_context"] = {
            "mode": self.mode,
            "winner": winner,
            "accepted": True,
            "margin": 1.0,
            "hard_reason": "family_term_requires_lil",
            "scores": {},
        }
        if self.mode == "active":
            token["corrected"] = winner
            token["choices"] = choices
            token["ambiguous"] = True
            token["crucial"] = True
        return True

    def _filter_candidates(
        self,
        candidates: Sequence[str],
        *,
        previous_surface: Optional[str],
        spellchecker,
    ) -> Tuple[List[str], Optional[str], str]:
        filtered = [
            candidate
            for candidate in candidates
            if not spellchecker._normalize_word(candidate).startswith("l'")
        ]
        if not filtered:
            filtered = list(candidates)

        previous_features = self._verb_features(previous_surface or "", spellchecker)
        previous_is_mur_imperative = (
            spellchecker._normalize_word(previous_surface or "") in {"mur", "morru"}
            and any(tense == "IMP" for _base, tense, _person, _root in previous_features)
        )
        if previous_is_mur_imperative:
            imperative = [
                candidate
                for candidate in filtered
                if any(
                    tense == "IMP"
                    for _base, tense, _person, _root in self._verb_features(
                        candidate,
                        spellchecker,
                    )
                )
            ]
            if imperative:
                return imperative, imperative[0], "serial_imperative"

        without_directional = [
            candidate
            for candidate in filtered
            if not spellchecker._normalize_word(candidate).startswith(("'l", "'il"))
        ]
        if without_directional:
            filtered = without_directional

        return filtered, None, ""

    @staticmethod
    def _preserve_initial_case(source: str, winner: str) -> str:
        if source[:1].isupper() and winner[:1].islower():
            return winner[:1].upper() + winner[1:]
        return winner

    def _ordered_choices(self, token: dict, winner: str, candidates: Sequence[str], spellchecker) -> List[dict]:
        existing: Dict[str, dict] = {}
        for choice in token.get("choices", []) or []:
            if not isinstance(choice, dict):
                continue
            key = spellchecker._normalize_word(str(choice.get("word", "")))
            if key:
                existing[key] = dict(choice)

        ordered: List[dict] = []
        for surface in [winner, *candidates]:
            key = spellchecker._normalize_word(surface)
            if not key or any(
                spellchecker._normalize_word(str(item.get("word", ""))) == key
                for item in ordered
            ):
                continue
            item = existing.get(key, {"word": surface, "meaning": ""})
            item["word"] = self._preserve_initial_case(
                str(token.get("corrected", "")),
                str(item.get("word", surface)),
            )
            ordered.append(item)
        return ordered

    def apply(self, tokens: List[dict], spellchecker) -> Dict[str, object]:
        if not self.is_available():
            return {"mode": self.mode, "available": False, "changed": 0}

        lexical_indexes = [
            index
            for index, token in enumerate(tokens)
            if isinstance(token, dict) and token.get("type") in _LEXICAL_TYPES
        ]
        changed_count = 0

        for pass_index in range(self.max_passes):
            pass_changed = 0
            selected_surfaces = {
                index: str(tokens[index].get("corrected", ""))
                for index in lexical_indexes
            }
            for position, token_index in enumerate(lexical_indexes):
                token = tokens[token_index]
                previous_surface = (
                    selected_surfaces[lexical_indexes[position - 1]]
                    if position > 0
                    else None
                )
                previous_previous_surface = (
                    selected_surfaces[lexical_indexes[position - 2]]
                    if position > 1
                    else None
                )
                next_surface = (
                    selected_surfaces[lexical_indexes[position + 1]]
                    if position + 1 < len(lexical_indexes)
                    else None
                )
                if self._apply_family_l_rule(
                    token,
                    spellchecker,
                    next_surface=next_surface,
                ):
                    selected_surfaces[token_index] = str(token.get("corrected", ""))
                    if self.mode == "active":
                        pass_changed += 1
                    continue

                candidates = self._candidate_surfaces(token, spellchecker)
                if len(candidates) < 2:
                    continue

                candidates, forced_winner, hard_reason = self._filter_candidates(
                    candidates,
                    previous_surface=previous_surface,
                    spellchecker=spellchecker,
                )
                if not candidates:
                    continue

                previous_word = self._context_edge(previous_surface, first=False)
                previous_previous_word = self._context_edge(
                    previous_previous_surface,
                    first=False,
                )
                next_word = self._context_edge(next_surface, first=True)
                scored = []
                for candidate in candidates:
                    components = self.scorer.score_candidate_components(
                        candidate,
                        prev_word=previous_word,
                        next_word=next_word,
                        prev_prev_word=previous_previous_word,
                    )
                    scored.append((candidate, components))
                scored.sort(
                    key=lambda item: float(item[1]["total_before_cap"]),
                    reverse=True,
                )

                winner = forced_winner or scored[0][0]
                winner_components = next(
                    components
                    for candidate, components in scored
                    if spellchecker._normalize_word(candidate)
                    == spellchecker._normalize_word(winner)
                )
                runner_score = max(
                    (
                        float(components["total_before_cap"])
                        for candidate, components in scored
                        if spellchecker._normalize_word(candidate)
                        != spellchecker._normalize_word(winner)
                    ),
                    default=0.0,
                )
                margin = float(winner_components["total_before_cap"]) - runner_score
                has_context = any(
                    float(winner_components[name]) > 0.0
                    for name in (
                        "left_bigram",
                        "internal_bigram",
                        "right_bigram",
                        "trigram",
                    )
                )
                accepted = bool(forced_winner) or (
                    int(winner_components["evidence_count"]) > 0
                    and margin >= self.minimum_margin
                    and (
                        has_context
                        or margin >= max(0.025, self.minimum_margin * 2.0)
                    )
                )

                token["corpus_context"] = {
                    "mode": self.mode,
                    "pass": pass_index + 1,
                    "winner": winner,
                    "accepted": accepted,
                    "margin": round(margin, 6),
                    "hard_reason": hard_reason,
                    "previous": previous_word,
                    "next": next_word,
                    "scores": {
                        candidate: components
                        for candidate, components in scored
                    },
                }
                if self.mode != "active" or not accepted:
                    continue

                old_surface = str(token.get("corrected", ""))
                surface_winner = self._preserve_initial_case(old_surface, winner)
                token["corrected"] = surface_winner
                token["choices"] = self._ordered_choices(
                    token,
                    surface_winner,
                    candidates,
                    spellchecker,
                )
                token["ambiguous"] = len(token["choices"]) >= 2
                selected_surfaces[token_index] = surface_winner
                if spellchecker._normalize_word(old_surface) != spellchecker._normalize_word(
                    surface_winner
                ):
                    pass_changed += 1

            changed_count += pass_changed
            if not pass_changed:
                break

        if self.mode == "shadow":
            logger.info(
                "CORPUS_CONTEXT_SHADOW tokens=%s",
                sum("corpus_context" in token for token in tokens if isinstance(token, dict)),
            )
        return {
            "mode": self.mode,
            "available": True,
            "changed": changed_count,
        }
