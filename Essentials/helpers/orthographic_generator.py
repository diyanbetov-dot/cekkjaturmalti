from typing import Iterable
from itertools import combinations

class MalteseOrthographicGenerator:
    """
    Sequential orthographic generator for Maltese għ/h repairs.

    This helper is intentionally strict and ordered:
    1. Generate one candidate.
    2. Check it immediately against the loaded dictionary.
    3. Return as soon as a valid match is found.

    It avoids unordered sets for priority-sensitive rules.
    """

    def __init__(self, spellchecker):
        self.spellchecker = spellchecker

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Unaccented keyboard shortcuts:
    #     h -> ħ
    #     c -> ċ
    #     z -> ż
    #     g -> ġ
    # ------------------------------------------------------------------

    SHORTCUT_TO_MALTESE = {
        "h": "ħ",
        "c": "ċ",
        "z": "ż",
        "g": "ġ",
    }


    def shortcut_letter_variants(
        self,
        word: str,
        max_changes: int = 3,
        max_variants: int = 128,
    ) -> list[str]:
        """
        Generate possible Maltese forms made by replacing ordinary keyboard
        letters with their Maltese counterparts.

        Examples:
            hazin  -> ħazin, hażin, ħażin
            cempel -> ċempel
            gdid   -> ġdid

        Literal 'gh' sequences are skipped because they are handled separately
        by the existing gh -> għ correction.
        """
        normalized = self._normalize(word)
        graphemes = self._graphemes(normalized)

        shortcut_positions: list[int] = []
        gh_positions: set[int] = set()

        # Do not interfere with the existing gh -> għ correction.
        for index in range(len(graphemes) - 1):
            if (
                graphemes[index] == "g"
                and graphemes[index + 1] == "h"
            ):
                gh_positions.add(index)
                gh_positions.add(index + 1)

        for index, letter in enumerate(graphemes):
            if (
                letter in self.SHORTCUT_TO_MALTESE
                and index not in gh_positions
            ):
                shortcut_positions.append(index)

        variants: list[str] = []

        maximum = min(
            max_changes,
            len(shortcut_positions),
        )

        # Generate forms with one change first, then two, then three.
        # This naturally ranks the smallest correction first.
        for number_of_changes in range(1, maximum + 1):
            for chosen_positions in combinations(
                shortcut_positions,
                number_of_changes,
            ):
                changed = graphemes[:]

                for position in chosen_positions:
                    changed[position] = (
                        self.SHORTCUT_TO_MALTESE[
                            changed[position]
                        ]
                    )

                candidate = self._from_graphemes(changed)
                self._add_unique(variants, candidate)

                if len(variants) >= max_variants:
                    return variants

        return variants


    def dictionary_shortcut_variants(
        self,
        word: str,
        max_changes: int = 3,
    ) -> list[str]:
        """
        Return only generated shortcut replacements that are real
        dictionary words.
        """
        matches: list[str] = []

        for candidate in self.shortcut_letter_variants(
            word,
            max_changes=max_changes,
        ):
            if self._is_dictionary_word(candidate):
                self._add_unique(matches, candidate)

        return matches


    def shortcut_change_count(
        self,
        original: str,
        candidate: str,
    ) -> int:
        """
        Count how many shortcut letters were replaced.
        """
        original_letters = self._graphemes(
            self._normalize(original)
        )
        candidate_letters = self._graphemes(
            self._normalize(candidate)
        )

        if len(original_letters) != len(candidate_letters):
            return 999

        return sum(
            first != second
            for first, second in zip(
                original_letters,
                candidate_letters,
            )
        )


    def correct_shortcut_letters(
        self,
        word: str,
    ) -> str | None:
        """
        Automatically correct an invalid shortcut spelling only when there
        is one uniquely best dictionary result.

        If several candidates require the same minimum number of changes,
        return None so that the spellchecker can show suggestions instead.
        """
        matches = self.dictionary_shortcut_variants(word)

        if not matches:
            return None

        ranked = sorted(
            matches,
            key=lambda candidate: (
                self.shortcut_change_count(
                    word,
                    candidate,
                ),
                candidate,
            ),
        )

        minimum_changes = self.shortcut_change_count(
            word,
            ranked[0],
        )

        best = [
            candidate
            for candidate in ranked
            if self.shortcut_change_count(
                word,
                candidate,
            ) == minimum_changes
        ]

        if len(best) == 1:
            return best[0]

        return None

    def _normalize(self, word: str) -> str:
        return self.spellchecker._normalize_word(word)

    def _graphemes(self, word: str) -> list[str]:
        return self.spellchecker._graphemes(word)

    def _from_graphemes(self, graphemes: Iterable[str]) -> str:
        return self.spellchecker._from_graphemes(graphemes)

    def _match_capitalisation(self, original: str, corrected: str) -> str:
        return self.spellchecker._match_capitalisation(original, corrected)

    def _add_unique(self, variants: list[str], candidate: str) -> None:
        candidate = self._normalize(candidate)
        if candidate and candidate not in variants:
            variants.append(candidate)

    def _is_dictionary_word(self, word: str) -> bool:
        return self._normalize(word) in self.spellchecker.dictionary_set

    # ------------------------------------------------------------------
    # Strict lookup variants
    # ------------------------------------------------------------------

    def move_gh_left_across_adjacent_vowel(self, word: str) -> list[str]:
        """
        Safe one-directional għ movement.

        Allowed:
            vowel + għ -> għ + vowel

        Example:
            ibagħt -> ibgħat

        Not allowed:
            għ + vowel -> vowel + għ

        The reverse movement is unsafe because it can make real words like
        għamel and agħmel compete as if they were spelling variants.
        """
        variants: list[str] = []
        g = self._graphemes(word)

        for i, token in enumerate(g):
            if token != "għ":
                continue

            if i > 0 and g[i - 1] in self.spellchecker.VOWELS:
                moved = g[:]
                moved[i - 1], moved[i] = moved[i], moved[i - 1]
                self._add_unique(variants, self._from_graphemes(moved))

        return variants

    def strict_lookup_variants(self, word: str) -> list[str]:
        """
        Ordered strict variants for lookup/scoring.

        Order matters:
            1. original normalized word
            2. ASCII gh -> Maltese għ
            3. shallow one-directional għ-left movement

        Example:
            ibaght -> ibagħt -> ibgħat
        """
        normalized = self._normalize(word)
        variants: list[str] = []

        self._add_unique(variants, normalized)
        self._add_unique(variants, normalized.replace("gh", "għ"))

        # Shallow recursion allows ibaght -> ibagħt -> ibgħat.
        for _ in range(2):
            current = list(variants)
            for variant in current:
                for moved in self.move_gh_left_across_adjacent_vowel(variant):
                    self._add_unique(variants, moved)

        return variants

    def first_dictionary_match(
        self,
        original_word: str,
        variants: Iterable[str],
        *,
        expand_strict: bool = True,
    ) -> str | None:
        """
        Checks generated variants in order and returns immediately
        when a valid dictionary word is found.
        """
        for variant in variants:
            normalized_variant = self._normalize(variant)

            # First check the exact candidate produced by the current rule.
            if normalized_variant in self.spellchecker.dictionary_set:
                return self._match_capitalisation(original_word, normalized_variant)

            # Then, optionally allow safe strict lookup variants.
            if expand_strict:
                for lookup in self.strict_lookup_variants(normalized_variant):
                    if lookup in self.spellchecker.dictionary_set:
                        return self._match_capitalisation(original_word, lookup)

        return None

    def correct_strict(self, word: str) -> str | None:
        """
        Tries exact strict orthographic variants.

        Example:
            ibaght -> ibgħat
            ibghatu -> ibgħatu
        """
        return self.first_dictionary_match(
            word,
            self.strict_lookup_variants(word),
            expand_strict=False,
        )

    def replace_g_with_ghajn(self, word: str) -> list[str]:
        """
        Generate variants with one plain g replaced by għ.

        This catches common missing-h spellings such as gandi -> għandi
        without letting the broader shortcut rule prefer ġ first.
        """
        normalized = self._normalize(word)
        graphemes = self._graphemes(normalized)
        variants: list[str] = []

        for index, letter in enumerate(graphemes):
            if letter != "g":
                continue

            changed = graphemes[:]
            changed[index] = "għ"
            self._add_unique(variants, self._from_graphemes(changed))

        return variants

    def move_gh_right_across_adjacent_vowel(self, word: str) -> list[str]:
        """
        Suggestion-only għ movement.

        Allowed for alternatives:
            għ + vowel -> vowel + għ

        Example:
            għamlu -> agħmlu

        This is intentionally not part of strict correction because the reverse
        direction can turn one valid word into another valid word.
        """
        variants: list[str] = []
        g = self._graphemes(word)

        for i, token in enumerate(g[:-1]):
            if token != "għ":
                continue

            if g[i + 1] in self.spellchecker.VOWELS:
                moved = g[:]
                moved[i], moved[i + 1] = moved[i + 1], moved[i]
                self._add_unique(variants, self._from_graphemes(moved))

        return variants

    def dictionary_gh_suggestion_variants(self, word: str) -> list[str]:
        matches = self.dictionary_gh_priority_variants(word)

        movement_sources = [word]
        movement_sources.extend(self.strict_lookup_variants(word))
        movement_sources.extend(matches)

        for source in movement_sources:
            for candidate in self.move_gh_right_across_adjacent_vowel(source):
                if self._is_dictionary_word(candidate):
                    self._add_unique(matches, candidate)

        return matches

    def dictionary_gh_priority_variants(self, word: str) -> list[str]:
        """
        Return dictionary matches from the high-priority għ path.

        Order:
            1. strict gh -> għ / movement variants
            2. one g -> għ replacement
            3. one inserted għ next to a vowel
        """
        matches: list[str] = []

        for group in (
            self.strict_lookup_variants(word),
            self.replace_g_with_ghajn(word),
            self.insert_token_next_to_vowels(word, "għ"),
            self.insert_h_after_gh_priority_variants(word),
        ):
            for candidate in group:
                normalized_candidate = self._normalize(candidate)

                if self._is_dictionary_word(normalized_candidate):
                    self._add_unique(matches, normalized_candidate)

                for lookup in self.strict_lookup_variants(normalized_candidate):
                    if self._is_dictionary_word(lookup):
                        self._add_unique(matches, lookup)

        return matches

    def correct_gh_priority(self, word: str) -> str | None:
        """
        Correct through the high-priority għ path before shortcut letters
        and broad dictionary scoring are allowed to compete.
        """
        match = self.correct_strict(word)
        if match:
            return match

        matches = self.dictionary_gh_priority_variants(word)
        if matches:
            original = self._graphemes(self._normalize(word))

            def shared_prefix(candidate: str) -> int:
                candidate_g = self._graphemes(self._normalize(candidate))
                count = 0
                for left, right in zip(original, candidate_g):
                    if left != right:
                        break
                    count += 1
                return count

            ranked = sorted(
                matches,
                key=lambda candidate: (
                    -shared_prefix(candidate),
                    self.spellchecker._word_distance(
                        self._normalize(word),
                        self._normalize(candidate),
                    ),
                    candidate,
                ),
            )
            return self._match_capitalisation(word, ranked[0])

        return None

    def insert_h_after_gh_priority_variants(self, word: str) -> list[str]:
        """
        Combine one high-priority għ repair with one missing h repair.

        This catches compact function-word spellings such as:
            adom -> għadom -> għadhom

        It is deliberately tied to the għ-priority path, so ordinary final-h
        insertion does not become a broad suggestion source.
        """
        variants: list[str] = []

        for gh_variant in self.insert_token_next_to_vowels(word, "għ"):
            for h_variant in self.insert_token_next_to_vowels(gh_variant, "h"):
                self._add_unique(variants, h_variant)

        return variants

    # ------------------------------------------------------------------
    # Insertion generation
    # ------------------------------------------------------------------

    def insert_token_next_to_vowels(self, word: str, token: str) -> list[str]:
        """
        Generates insertion candidates in strict order.

        For each vowel, try:
            1. before the vowel
            2. after the vowel

        Example with għ:
            amel -> għamel -> agħmel -> amgħel -> amegħl

        Since each candidate is tested immediately by the caller,
        għamel can win before agħmel is even tested.
        """
        normalized = self._normalize(word)
        g = self._graphemes(normalized)
        variants: list[str] = []

        for i, ch in enumerate(g):
            if ch not in self.spellchecker.VOWELS:
                continue

            before = self._from_graphemes(g[:i] + [token] + g[i:])
            after = self._from_graphemes(g[: i + 1] + [token] + g[i + 1:])

            if i == 0 or g[i - 1] != token:
                self._add_unique(variants, before)

            if i + 1 >= len(g) or g[i + 1] != token:
                self._add_unique(variants, after)

        return variants

    def correct_by_inserting_token(self, word: str, token: str) -> str | None:
        return self.first_dictionary_match(
            word,
            self.insert_token_next_to_vowels(word, token),
            expand_strict=True,
        )

    # ------------------------------------------------------------------
    # Removal generation
    # ------------------------------------------------------------------

    def remove_token(self, word: str, token: str) -> list[str]:
        """
        Generates removal candidates from left to right.

        Example:
            bagħħat -> bagħat, bagħat depending on position
        """
        normalized = self._normalize(word)
        g = self._graphemes(normalized)
        variants: list[str] = []

        for i, ch in enumerate(g):
            if ch == token:
                self._add_unique(variants, self._from_graphemes(g[:i] + g[i + 1:]))

        return variants

    def correct_by_removing_token(self, word: str, token: str) -> str | None:
        return self.first_dictionary_match(
            word,
            self.remove_token(word, token),
            expand_strict=True,
        )

    def remove_extra_double_variants(self, word: str) -> list[str]:
        normalized = self._normalize(word)
        graphemes = self._graphemes(normalized)
        variants: list[str] = []

        for index in range(len(graphemes) - 1):
            if graphemes[index] != graphemes[index + 1]:
                continue

            token = graphemes[index]
            if len(token) != 1 or not token.isalpha() or token in self.spellchecker.VOWELS:
                continue

            self._add_unique(
                variants,
                self._from_graphemes(graphemes[:index] + graphemes[index + 1:]),
            )

        return variants

    def dictionary_remove_extra_double_variants(self, word: str) -> list[str]:
        matches: list[str] = []

        for candidate in self.remove_extra_double_variants(word):
            if self._is_dictionary_word(candidate):
                self._add_unique(matches, candidate)

        return matches

    def correct_extra_double(self, word: str) -> str | None:
        return self.first_dictionary_match(
            word,
            self.remove_extra_double_variants(word),
            expand_strict=False,
        )

    def insert_h_after_d_variants(self, word: str) -> list[str]:
        normalized = self._normalize(word)
        graphemes = self._graphemes(normalized)
        variants: list[str] = []

        for index, token in enumerate(graphemes[:-1]):
            if token != "d" or graphemes[index + 1] == "h":
                continue

            self._add_unique(
                variants,
                self._from_graphemes(graphemes[:index + 1] + ["h"] + graphemes[index + 1:]),
            )

        return variants

    def dictionary_insert_h_after_d_variants(self, word: str) -> list[str]:
        matches: list[str] = []

        for candidate in self.insert_h_after_d_variants(word):
            if self._is_dictionary_word(candidate):
                self._add_unique(matches, candidate)

        return matches

    def correct_missing_h_after_d(self, word: str) -> str | None:
        return self.first_dictionary_match(
            word,
            self.insert_h_after_d_variants(word),
            expand_strict=False,
        )

# ------------------------------------------------------------------
# d/t confusion
# ------------------------------------------------------------------

    def _substitute_one_letter_pair(
        self,
        word: str,
        first: str,
        second: str,
    ) -> list[str]:
        """
        Generate variants containing exactly one substitution between a
        two-letter confusion pair.
        """
        normalized = self._normalize(word)
        graphemes = self._graphemes(normalized)
        variants: list[str] = []

        for index, letter in enumerate(graphemes):
            if letter not in {first, second}:
                continue

            replacement = second if letter == first else first

            changed = graphemes[:]
            changed[index] = replacement

            candidate = self._from_graphemes(changed)
            self._add_unique(variants, candidate)

        return variants

    def substitute_d_t(self, word: str) -> list[str]:
        """
        Generate variants containing exactly one d/t substitution.

        Examples:
            rqatt -> rqadt
            rmiet -> rmied

        Only one letter is changed at a time to avoid excessive candidates.
        """
        return self._substitute_one_letter_pair(word, "d", "t")


    def dictionary_d_t_variants(self, word: str) -> list[str]:
        """
        Return only d/t variants that are real dictionary words.
        """
        matches: list[str] = []

        for candidate in self.substitute_d_t(word):
            if self._is_dictionary_word(candidate):
                self._add_unique(matches, candidate)

        return matches


    def correct_d_t_confusion(self, word: str) -> str | None:
        """
        Correct an invalid word through one d/t substitution.

        This should only be used after confirming that the original word
        is not already in the dictionary.
        """
        return self.first_dictionary_match(
            word,
            self.substitute_d_t(word),
            expand_strict=False,
        )

    # ------------------------------------------------------------------
    # b/p confusion
    # ------------------------------------------------------------------

    def substitute_b_p(self, word: str) -> list[str]:
        """
        Generate variants containing exactly one b/p substitution.

        This mirrors d/t confusion and is useful for forms such as
        bitghada -> pitgħada.
        """
        return self._substitute_one_letter_pair(word, "b", "p")


    def dictionary_b_p_variants(self, word: str) -> list[str]:
        """
        Return only b/p variants that resolve to real dictionary words.
        """
        matches: list[str] = []

        for candidate in self.substitute_b_p(word):
            if self._is_dictionary_word(candidate):
                self._add_unique(matches, candidate)

            for lookup in self.strict_lookup_variants(candidate):
                if self._is_dictionary_word(lookup):
                    self._add_unique(matches, lookup)

        return matches


    def correct_b_p_confusion(self, word: str) -> str | None:
        """
        Correct an invalid word through one b/p substitution.

        Strict expansion is allowed so b/p can cooperate with gh -> għ.
        """
        return self.first_dictionary_match(
            word,
            self.substitute_b_p(word),
            expand_strict=True,
        )

    # ------------------------------------------------------------------
    # i/ie confusion
    # ------------------------------------------------------------------

    def substitute_i_ie(self, word: str) -> list[str]:
        """
        Generate variants containing exactly one i/ie substitution.

        Examples:
            jin  -> jien
            jien -> jin

        Only one vowel sequence is changed at a time to keep this as narrow
        as the d/t repair.
        """
        normalized = self._normalize(word)
        variants: list[str] = []
        index = 0

        while index < len(normalized):
            if normalized.startswith("ie", index):
                candidate = normalized[:index] + "i" + normalized[index + 2:]
                self._add_unique(variants, candidate)
                index += 2
                continue

            if normalized[index] == "i":
                candidate = normalized[:index] + "ie" + normalized[index + 1:]
                self._add_unique(variants, candidate)

            index += 1

        return variants


    def dictionary_i_ie_variants(self, word: str) -> list[str]:
        """
        Return only i/ie variants that are real dictionary words.
        """
        matches: list[str] = []

        for candidate in self.substitute_i_ie(word):
            if self._is_dictionary_word(candidate):
                self._add_unique(matches, candidate)
            for lookup in self.strict_lookup_variants(candidate):
                if self._is_dictionary_word(lookup):
                    self._add_unique(matches, lookup)

        return matches


    def correct_i_ie_confusion(self, word: str) -> str | None:
        """
        Correct an invalid word through one i/ie substitution.

        This should only be used after confirming that the original word
        is not already in the dictionary.
        """
        return self.first_dictionary_match(
            word,
            self.substitute_i_ie(word),
            expand_strict=True,
        )

    # ------------------------------------------------------------------
    # aw / għu confusion
    # ------------------------------------------------------------------

    def final_aw_to_ghu_variants(self, word: str) -> list[str]:
        normalized = self._normalize(word)
        variants: list[str] = []

        if normalized.endswith("aw") and len(normalized) > 2:
            self._add_unique(variants, normalized[:-2] + "għu")

        return variants

    def dictionary_final_aw_to_ghu_variants(self, word: str) -> list[str]:
        matches: list[str] = []

        for candidate in self.final_aw_to_ghu_variants(word):
            if self._is_dictionary_word(candidate):
                self._add_unique(matches, candidate)

        return matches

    def correct_final_aw_to_ghu(self, word: str) -> str | None:
        return self.first_dictionary_match(
            word,
            self.final_aw_to_ghu_variants(word),
            expand_strict=False,
        )

    # ------------------------------------------------------------------
    # final j/h suggestion pairs
    # ------------------------------------------------------------------

    def final_j_h_variants(self, word: str) -> list[str]:
        normalized = self._normalize(word)
        variants: list[str] = []

        if normalized.endswith("ja"):
            self._add_unique(variants, normalized[:-2] + "ha")
        elif normalized.endswith("ha"):
            self._add_unique(variants, normalized[:-2] + "ja")

        return variants

    def dictionary_final_j_h_variants(self, word: str) -> list[str]:
        matches: list[str] = []

        for candidate in self.final_j_h_variants(word):
            if self._is_dictionary_word(candidate):
                self._add_unique(matches, candidate)

        return matches

    # ------------------------------------------------------------------
    # final għ/h/ħ confusion
    # ------------------------------------------------------------------

    def substitute_final_gh_h_hbar(self, word: str) -> list[str]:
        """
        Generate variants by replacing only final għ/h/ħ.

        This is intentionally final-only because these letters are much less
        safely interchangeable inside a word.
        """
        normalized = self._normalize(word)
        variants: list[str] = []

        if normalized.endswith("gh"):
            base = normalized[:-2]
            self._add_unique(variants, base + "h")
            self._add_unique(variants, base + "ħ")
            self._add_unique(variants, base + "għ")
            return variants

        graphemes = self._graphemes(normalized)

        if not graphemes or graphemes[-1] not in {"għ", "h", "ħ"}:
            return variants

        for replacement in ("għ", "h", "ħ"):
            if replacement == graphemes[-1]:
                continue

            changed = graphemes[:]
            changed[-1] = replacement
            self._add_unique(variants, self._from_graphemes(changed))

        return variants


    def dictionary_final_gh_h_hbar_variants(self, word: str) -> list[str]:
        """
        Return only final għ/h/ħ variants that are real dictionary words.
        """
        matches: list[str] = []

        for candidate in self.substitute_final_gh_h_hbar(word):
            if self._is_dictionary_word(candidate):
                self._add_unique(matches, candidate)

        return matches


    def correct_final_gh_h_hbar_confusion(self, word: str) -> str | None:
        """
        Correct an invalid word through one final għ/h/ħ substitution.

        This should only be used after confirming that the original word
        is not already in the dictionary.
        """
        return self.first_dictionary_match(
            word,
            self.substitute_final_gh_h_hbar(word),
            expand_strict=False,
        )

    # ------------------------------------------------------------------
    # Debug
    # ------------------------------------------------------------------

    def debug(self, word: str) -> dict:
        return {
            "word": word,
            "strict_lookup_variants": self.strict_lookup_variants(word),
            "replace_g_with_ghajn_variants": self.replace_g_with_ghajn(word),
            "gh_priority_dictionary_matches": self.dictionary_gh_priority_variants(word),
            "insert_gh_variants": self.insert_token_next_to_vowels(word, "għ"),
            "insert_h_variants": self.insert_token_next_to_vowels(word, "h"),
            "remove_gh_variants": self.remove_token(word, "għ"),
            "remove_h_variants": self.remove_token(word, "h"),
            "remove_q_variants": self.remove_token(word, "q"),
            "b_p_variants": self.substitute_b_p(word),
            "i_ie_variants": self.substitute_i_ie(word),
            "final_gh_h_hbar_variants": self.substitute_final_gh_h_hbar(word),
            "gh_priority_match": self.correct_gh_priority(word),
            "strict_match": self.correct_strict(word),
            "insert_gh_match": self.correct_by_inserting_token(word, "għ"),
            "insert_h_match": self.correct_by_inserting_token(word, "h"),
            "remove_gh_match": self.correct_by_removing_token(word, "għ"),
            "remove_h_match": self.correct_by_removing_token(word, "h"),
            "remove_q_match": self.correct_by_removing_token(word, "q"),
            "b_p_match": self.correct_b_p_confusion(word),
            "i_ie_match": self.correct_i_ie_confusion(word),
            "final_gh_h_hbar_match": self.correct_final_gh_h_hbar_confusion(word),
        }
