from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from suffixation.suffix_generator import MalteseSuffixGenerator


@dataclass(frozen=True, slots=True)
class SuffixScore:
    candidate: str
    score: float
    edit_distance: int
    consonant_score: float
    vowel_slot_score: float
    vowel_count_score: float
    length_score: float
    stage: str
    matched_typo_form: str


class HopeSuffixSpellcheckerAdapter:
    """Minimal interface required by the established suffix modules."""

    VOWELS = frozenset("aeiouàèìòù")

    def __init__(self, dictionary_keys: Iterable[str], normalizer: Callable[[str], str]) -> None:
        self.dictionary_set = frozenset(dictionary_keys)
        self._normalizer = normalizer
        self.manual_suffix_stems: dict[str, tuple[str, ...]] = {}

    def _normalize_word(self, word: str) -> str:
        return self._normalizer(word)

    def _graphemes(self, word: str) -> list[str]:
        normalized = self._normalize_word(word)
        output: list[str] = []
        index = 0
        while index < len(normalized):
            if normalized.startswith("għ", index):
                output.append("għ")
                index += 2
            else:
                output.append(normalized[index])
                index += 1
        return output

    @staticmethod
    def _from_graphemes(graphemes: Iterable[str]) -> str:
        return "".join(graphemes)

    def _letter_tokens(self, word: str) -> tuple[str, ...]:
        return tuple(
            "ʕ" if grapheme == "għ" else grapheme
            for grapheme in self._graphemes(word)
            if grapheme == "għ" or grapheme.isalpha()
        )

    @staticmethod
    def _distance(first: tuple[str, ...], second: tuple[str, ...]) -> int:
        rows = len(first) + 1
        columns = len(second) + 1
        matrix = [[0] * columns for _ in range(rows)]
        for row in range(rows):
            matrix[row][0] = row
        for column in range(columns):
            matrix[0][column] = column
        for row in range(1, rows):
            for column in range(1, columns):
                cost = 0 if first[row - 1] == second[column - 1] else 1
                matrix[row][column] = min(
                    matrix[row - 1][column] + 1,
                    matrix[row][column - 1] + 1,
                    matrix[row - 1][column - 1] + cost,
                )
                if (
                    row > 1
                    and column > 1
                    and first[row - 1] == second[column - 2]
                    and first[row - 2] == second[column - 1]
                ):
                    matrix[row][column] = min(matrix[row][column], matrix[row - 2][column - 2] + 1)
        return matrix[-1][-1]

    def _word_distance(self, first: str, second: str) -> int:
        return self._distance(self._letter_tokens(first), self._letter_tokens(second))

    def _max_distance(self, word: str) -> int:
        length = len(self._letter_tokens(word))
        return 1 if length <= 4 else 2 if length <= 8 else 3

    def _strict_lookup_variants(self, word: str) -> list[str]:
        normalized = self._normalize_word(word)
        variants = [normalized]
        gh_variant = normalized.replace("gh", "għ")
        if gh_variant not in variants:
            variants.append(gh_variant)
        return variants

    def i_ie_variants(self, word: str) -> list[str]:
        normalized = self._normalize_word(word)
        variants: list[str] = []
        index = 0
        while index < len(normalized):
            if normalized.startswith("ie", index):
                candidate = normalized[:index] + "i" + normalized[index + 2:]
                if candidate not in variants:
                    variants.append(candidate)
                index += 2
                continue
            if normalized[index] == "i":
                candidate = normalized[:index] + "ie" + normalized[index + 1:]
                if candidate not in variants:
                    variants.append(candidate)
            index += 1
        return variants

    def missing_double_variants(self, word: str) -> list[str]:
        normalized = self._normalize_word(word)
        graphemes = self._graphemes(normalized)
        variants: list[str] = []
        for index, token in enumerate(graphemes):
            if len(token) != 1 or not token.isalpha() or token in self.VOWELS:
                continue
            previous_same = index > 0 and graphemes[index - 1] == token
            next_same = index + 1 < len(graphemes) and graphemes[index + 1] == token
            if previous_same or next_same:
                continue
            candidate = self._from_graphemes(
                graphemes[: index + 1] + [token] + graphemes[index + 1:]
            )
            if candidate not in variants:
                variants.append(candidate)
        return variants

    def single_double_variants(self, word: str) -> list[str]:
        variants = self.missing_double_variants(word)
        graphemes = self._graphemes(word)
        for index in range(len(graphemes) - 1):
            token = graphemes[index]
            if (
                token == graphemes[index + 1]
                and len(token) == 1
                and token.isalpha()
                and token not in self.VOWELS
            ):
                candidate = self._from_graphemes(
                    graphemes[:index] + graphemes[index + 1:]
                )
                if candidate not in variants:
                    variants.append(candidate)
        return variants

    def aj_ej_guttural_variants(self, word: str) -> list[str]:
        graphemes = self._graphemes(word)
        variants: list[str] = []
        for index in range(len(graphemes) - 1):
            pair = "".join(graphemes[index:index + 2])
            if pair not in {"aj", "ej"}:
                continue
            replacements = ("għi", "għej") if pair == "ej" else ("għi", "għaj")
            for replacement in replacements:
                repl_graphemes = self._graphemes(replacement)
                candidate_graphemes = graphemes[:index] + repl_graphemes + graphemes[index + 2:]
                candidate = self._from_graphemes(candidate_graphemes)
                if candidate not in variants:
                    variants.append(candidate)
                if index + 2 < len(graphemes):
                    next_token = graphemes[index + 2]
                    if (
                        len(next_token) == 1
                        and next_token.isalpha()
                        and next_token not in self.VOWELS
                    ):
                        doubled = self._from_graphemes(
                            candidate_graphemes[: index + len(repl_graphemes) + 1]
                            + [next_token]
                            + candidate_graphemes[index + len(repl_graphemes) + 1:]
                        )
                        if doubled not in variants:
                            variants.append(doubled)
                for j_index in range(len(candidate_graphemes) - 1):
                    token = candidate_graphemes[j_index]
                    if token == candidate_graphemes[j_index + 1] and token == "j":
                        single_j = self._from_graphemes(
                            candidate_graphemes[:j_index] + candidate_graphemes[j_index + 1:]
                        )
                        if single_j not in variants:
                            variants.append(single_j)
        return variants

    def suffix_stem_variants(self, word: str) -> list[str]:
        variants = self._strict_lookup_variants(word)
        for candidate in self.manual_suffix_stems.get(self._normalize_word(word), ()):
            if candidate not in variants:
                variants.append(candidate)
        for candidate in (
            self.i_ie_variants(word)
            + self.single_double_variants(word)
            + self.aj_ej_guttural_variants(word)
        ):
            if candidate not in variants:
                variants.append(candidate)
        return variants

    def _vowel_slots(self, word: str) -> list[tuple[int, str]]:
        return [
            (index, token)
            for index, token in enumerate(self._letter_tokens(word))
            if token in self.VOWELS
        ]

    @staticmethod
    def _vowel_slot_score(
        first: list[tuple[int, str]], second: list[tuple[int, str]]
    ) -> float:
        if not first and not second:
            return 1.0
        if not first or not second:
            return 0.0
        max_position = max(first[-1][0], second[-1][0], 1)
        used: set[int] = set()
        matched = 0.0
        for first_position, first_vowel in first:
            best = 0.0
            best_index = -1
            for index, (second_position, second_vowel) in enumerate(second):
                if index in used or first_vowel != second_vowel:
                    continue
                position_score = max(0.0, 1.0 - abs(first_position - second_position) / max_position)
                if position_score > best:
                    best = position_score
                    best_index = index
            if best_index >= 0:
                used.add(best_index)
                matched += best
        count_ratio = min(len(first), len(second)) / max(len(first), len(second))
        return (matched / len(first)) * count_ratio

    def _candidate_score(self, typo: str, candidate: str, stage: str) -> SuffixScore:
        typo = self._normalize_word(typo)
        candidate = self._normalize_word(candidate)
        typo_tokens = self._letter_tokens(typo)
        candidate_tokens = self._letter_tokens(candidate)
        max_length = max(1, len(typo_tokens), len(candidate_tokens))
        edit_distance = self._distance(typo_tokens, candidate_tokens)
        typo_consonants = tuple(token for token in typo_tokens if token not in self.VOWELS)
        candidate_consonants = tuple(token for token in candidate_tokens if token not in self.VOWELS)
        consonant_distance = self._distance(typo_consonants, candidate_consonants)
        consonant_score = consonant_distance / max(1, len(typo_consonants), len(candidate_consonants))
        vowel_slot_score = self._vowel_slot_score(self._vowel_slots(typo), self._vowel_slots(candidate))
        typo_vowels = sum(token in self.VOWELS for token in typo_tokens)
        candidate_vowels = sum(token in self.VOWELS for token in candidate_tokens)
        vowel_count_score = abs(typo_vowels - candidate_vowels) / max(1, typo_vowels, candidate_vowels)
        length_score = abs(len(typo_tokens) - len(candidate_tokens)) / max_length
        score = (
            (1.0 - vowel_slot_score) * 0.40
            + (edit_distance / max_length) * 0.25
            + consonant_score * 0.20
            + vowel_count_score * 0.10
            + length_score * 0.05
        )
        return SuffixScore(
            candidate=candidate,
            score=score,
            edit_distance=edit_distance,
            consonant_score=consonant_score,
            vowel_slot_score=vowel_slot_score,
            vowel_count_score=vowel_count_score,
            length_score=length_score,
            stage=stage,
            matched_typo_form=typo,
        )


class HopeSuffixEngine:
    def __init__(
        self,
        dictionary_keys: Iterable[str],
        verb_files: list[Path],
        normalizer: Callable[[str], str],
    ) -> None:
        self.adapter = HopeSuffixSpellcheckerAdapter(dictionary_keys, normalizer)
        self.generator = MalteseSuffixGenerator(
            spellchecker=self.adapter,
            verbs_file=verb_files,
        )

    def set_manual_suffix_stems(self, mapping: dict[str, tuple[str, ...]]) -> None:
        self.adapter.manual_suffix_stems = {
            self.adapter._normalize_word(source): tuple(
                self.adapter._normalize_word(target) for target in targets
            )
            for source, targets in mapping.items()
            if targets
        }
        self.generator._inverse_base_guesses_cached.cache_clear()
        self.generator.exact_suffix_matches.cache_clear()

    def exact(self, word: str) -> bool:
        return bool(self.generator.exact_suffix_matches(word))

    def exact_surface_variant(self, word: str) -> bool:
        """Accept reversible i-/i-ie spellings of exact generated suffix forms.

        This is intentionally narrower than suggestions(): it only validates a
        typed surface when a small reversible surface form lands on an exact
        generated suffix form. It does not invent a correction target.
        """
        normalized = self.adapter._normalize_word(word)
        variants: list[str] = [normalized]
        if normalized.startswith("i") and len(normalized) > 4:
            variants.append(normalized[1:])
        for base in list(variants):
            for candidate in self.adapter.i_ie_variants(base):
                if candidate not in variants:
                    variants.append(candidate)
        return any(self.exact(candidate) for candidate in variants)

    def ha_suffix_candidates(self, word: str) -> list[str]:
        """For a verb base ending in -a, return suffix-engine-validated -ha forms.

        Covers Maltese suffixation patterns:
          - Final-weak / Geminate (a → ie before -ha): semma → semmieha, bda → bdieha
          - Final-weak (VCa): aqra + ha → aqraha
          - Strong CVCCa: berika → berik + ha → berikha

        Only returns forms that pass the suffix engine's exact morphological
        match (exact_suffix_matches), so non-verb -a words (adjectives, nouns)
        are naturally excluded.
        """
        normalized = self.adapter._normalize_word(word)
        if not normalized.endswith("a") or len(normalized) < 2:
            return []
        results: list[str] = []
        # Pattern A – change final -a to -ie- then add -ha (semma → semmieha, bda → bdieha)
        candidate_a = normalized[:-1] + "ieha"
        if self.exact(candidate_a):
            results.append(candidate_a)
        # Pattern B – add -ha directly (aqra → aqraha)
        candidate_b = normalized + "ha"
        if candidate_b not in results and self.exact(candidate_b):
            results.append(candidate_b)
        # Pattern C – drop final -a then add -ha (berika → berikha)
        candidate_c = normalized[:-1] + "ha"
        if candidate_c not in results and self.exact(candidate_c):
            results.append(candidate_c)
        return results

    def suggestions(self, word: str, limit: int = 6) -> list[str]:
        return self.generator.suggest_suffixes(word, limit=limit)

    def preferred_f1_kaka_candidate(self, word: str) -> str | None:
        """Return the morphology-backed KKa+suffix reading, when present."""
        normalized = self.adapter._normalize_word(word)
        forms = [normalized]
        if normalized.endswith("ħ"):
            forms.append(normalized[:-1] + "h")
        for form in forms:
            for parsed in self.generator.parse_possible_suffixes(form):
                for candidate in self.generator._generated_candidates_for_parse(parsed):
                    if candidate.rule_id != "F1_FINAL_WEAK_KAKA_CONTRACTION":
                        continue
                    if candidate.base == parsed.typed_stem or candidate.surface == form:
                        return candidate.surface
        return None
