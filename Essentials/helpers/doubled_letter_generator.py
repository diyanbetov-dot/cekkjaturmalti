class MalteseDoubledLetterGenerator:
    """
    Strict missing-doubled-consonant generator.

    It generates candidates in a fixed left-to-right order and returns
    immediately when a generated candidate is found in the dictionary.

    Example:
        basejt -> bassejt
    """

    def __init__(self, spellchecker):
        self.spellchecker = spellchecker

    def _normalize(self, word: str) -> str:
        return self.spellchecker._normalize_word(word)

    def _graphemes(self, word: str) -> list[str]:
        return self.spellchecker._graphemes(word)

    def _from_graphemes(self, graphemes) -> str:
        return self.spellchecker._from_graphemes(graphemes)

    def _is_vowel(self, grapheme: str) -> bool:
        return grapheme in self.spellchecker.VOWELS

    def _is_doublable_consonant(self, grapheme: str) -> bool:
        return (
            len(grapheme) == 1
            and grapheme.isalpha()
            and not self._is_vowel(grapheme)
        )

    def missing_double_variants(self, word: str) -> list[str]:
        normalized = self._normalize(word)
        g = self._graphemes(normalized)
        variants: list[str] = []

        def add(candidate: str) -> None:
            if candidate and candidate not in variants:
                variants.append(candidate)

        for i, token in enumerate(g):
            if not self._is_doublable_consonant(token):
                continue

            previous_same = i > 0 and g[i - 1] == token
            next_same = i + 1 < len(g) and g[i + 1] == token

            if previous_same or next_same:
                continue

            candidate = self._from_graphemes(g[: i + 1] + [token] + g[i + 1:])
            add(candidate)

        return variants

    def j_priority_variants(self, word: str) -> list[str]:
        normalized = self._normalize(word)
        g = self._graphemes(normalized)
        variants: list[str] = []

        def add(candidate: str) -> None:
            if candidate and candidate not in variants:
                variants.append(candidate)

        for i, token in enumerate(g):
            if token != "j":
                continue

            previous_same = i > 0 and g[i - 1] == "j"
            next_same = i + 1 < len(g) and g[i + 1] == "j"

            if previous_same or next_same:
                if next_same:
                    add(self._from_graphemes(g[:i] + g[i + 1:]))
                continue

            add(self._from_graphemes(g[: i + 1] + ["j"] + g[i + 1:]))

        return variants

    def correct_j_priority(self, word: str) -> str | None:
        for candidate in self.j_priority_variants(word):
            if candidate in self.spellchecker.dictionary_set:
                return self.spellchecker._match_capitalisation(word, candidate)

            for lookup in self.spellchecker._strict_lookup_variants(candidate):
                if lookup in self.spellchecker.dictionary_set:
                    return self.spellchecker._match_capitalisation(word, lookup)

        return None

    def correct_missing_double(self, word: str) -> str | None:
        for candidate in self.missing_double_variants(word):
            if candidate in self.spellchecker.dictionary_set:
                return self.spellchecker._match_capitalisation(word, candidate)

            for lookup in self.spellchecker._strict_lookup_variants(candidate):
                if lookup in self.spellchecker.dictionary_set:
                    return self.spellchecker._match_capitalisation(word, lookup)

        return None

    def debug(self, word: str) -> dict:
        variants = self.missing_double_variants(word)
        checks = []

        for candidate in variants:
            checks.append({
                "candidate": candidate,
                "exact_dictionary_hit": candidate in self.spellchecker.dictionary_set,
                "strict_lookup_hits": [
                    lookup
                    for lookup in self.spellchecker._strict_lookup_variants(candidate)
                    if lookup in self.spellchecker.dictionary_set
                ],
            })

        return {
            "word": word,
            "normalized": self._normalize(word),
            "corrected": self.correct_missing_double(word),
            "variants_in_order": variants,
            "checks": checks,
        }
