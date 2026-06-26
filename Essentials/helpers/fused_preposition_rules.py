from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FusedPrepositionSuggestion:
    corrected: str
    choices: list[dict[str, str]]


class MalteseFusedPrepositionRules:
    """
    Split fused short function words from their following word.

    This is deliberately deterministic. The remainder is recovered only
    through exact dictionary lookup and strict orthographic helpers, never
    through broad fuzzy scoring.
    """

    APOSTROPHE_PREFIXES = {"x", "b", "f", "t"}
    SPACED_PREFIXES = {"xi", "bi", "fi"}

    def __init__(
        self,
        *,
        spellchecker,
        article_rules,
        meaning_index,
    ) -> None:
        self.spellchecker = spellchecker
        self.article_rules = article_rules
        self.meaning_index = meaning_index

    def normalize(self, word: str) -> str:
        return self.spellchecker._normalize_word(word)

    def add_unique(self, items: list[str], word: str | None) -> None:
        if not word:
            return

        normalized = self.normalize(word)
        if normalized and normalized not in items:
            items.append(normalized)

    def strict_remainder_candidates(self, remainder: str) -> list[str]:
        normalized = self.normalize(remainder)
        candidates: list[str] = []
        orthographic = getattr(self.spellchecker, "orthographic_generator", None)

        def add_if_dictionary(candidate: str) -> None:
            candidate = self.normalize(candidate)
            if candidate in self.spellchecker.dictionary_set:
                self.add_unique(candidates, candidate)

        add_if_dictionary(normalized)

        for candidate in self.spellchecker._strict_lookup_variants(normalized):
            add_if_dictionary(candidate)

        if normalized.startswith("h") and len(normalized) >= 2:
            h_as_gh = "għ" + normalized[1:]
            add_if_dictionary(h_as_gh)
            for candidate in self.spellchecker._strict_lookup_variants(h_as_gh):
                add_if_dictionary(candidate)

        if orthographic is not None:
            helper_groups = [
                getattr(orthographic, "dictionary_gh_priority_variants", None),
                getattr(orthographic, "dictionary_shortcut_variants", None),
                getattr(orthographic, "dictionary_final_gh_h_hbar_variants", None),
                getattr(orthographic, "dictionary_i_ie_variants", None),
            ]

            for helper in helper_groups:
                if helper is None:
                    continue

                for candidate in helper(normalized):
                    add_if_dictionary(candidate)

        return candidates

    def is_verb(self, word: str) -> bool:
        normalized = self.normalize(word)
        suffix_generator = getattr(self.spellchecker, "suffix_generator", None)

        if suffix_generator is not None:
            if suffix_generator.verb_index.word_records(normalized):
                return True
            if suffix_generator.exact_suffix_matches(normalized):
                return True

        for tag in self.spellchecker.word_tags.get(normalized, set()):
            if tag.startswith("T-"):
                return True

        return False

    def noun_choice_meaning(self, prefix: str, noun: str) -> str:
        noun_meaning = self.meaning_index.meaning_for(noun)
        return f"{prefix} {noun_meaning}" if noun_meaning else prefix

    def short_article_choices(self, noun: str) -> list[dict[str, str]]:
        article = self.article_rules.assimilate("l-", noun)
        definite = f"{article}{noun}"

        return [
            {
                "word": definite,
                "meaning": self.noun_choice_meaning("the", noun),
            },
            {
                "word": f"'l {noun}",
                "meaning": self.noun_choice_meaning("to a", noun),
            },
            {
                "word": f"'l-{noun}",
                "meaning": self.noun_choice_meaning("to the", noun),
            },
        ]

    def match(self, word: str) -> FusedPrepositionSuggestion | None:
        normalized = self.normalize(word)

        if (
            not normalized
            or len(normalized) < 3
            or normalized.startswith(("'", "għ"))
        ):
            return None

        prefix = normalized[0]
        if normalized in self.spellchecker.dictionary_set:
            return None

        if len(normalized) >= 3 and normalized[1] == "'":
            remainder = normalized[2:]
            if self.strict_remainder_candidates(remainder):
                return None

        orthographic = getattr(self.spellchecker, "orthographic_generator", None)
        if orthographic is not None and hasattr(orthographic, "correct_shortcut_letters"):
            shortcut_match = orthographic.correct_shortcut_letters(normalized)
            if shortcut_match and self.normalize(shortcut_match) in self.spellchecker.dictionary_set:
                return None

        if normalized in self.spellchecker.dictionary_set and prefix != "t":
            return None

        for spaced_prefix in self.SPACED_PREFIXES:
            if not normalized.startswith(spaced_prefix) or len(normalized) <= len(spaced_prefix):
                continue

            remainder = normalized[len(spaced_prefix):]
            if len(remainder) < 2:
                continue

            candidates = self.strict_remainder_candidates(remainder)
            if not candidates:
                continue

            corrected_remainder = candidates[0]
            corrected = f"{spaced_prefix} {corrected_remainder}"
            return FusedPrepositionSuggestion(
                corrected=corrected,
                choices=[
                    {
                        "word": corrected,
                        "meaning": self.meaning_index.meaning_for(corrected_remainder),
                    }
                ],
            )

        if len(normalized) >= 3 and normalized[1] == "'":
            remainder = normalized[2:]
        else:
            if prefix == "t":
                return None
            remainder = normalized[1:]

        if prefix == "x" and remainder.startswith("h"):
            h_candidate = "ħ" + remainder[1:]
            if h_candidate in self.spellchecker.dictionary_set:
                x_apostrophe = f"x'{h_candidate}"
                choices = [
                    {
                        "word": x_apostrophe,
                        "meaning": self.meaning_index.meaning_for(x_apostrophe),
                    }
                ]
                
                x_combined = f"x{h_candidate}"
                corrected = x_apostrophe
                
                if x_combined in self.spellchecker.dictionary_set:
                    if "'" not in word:
                        # If user didn't type an apostrophe, prioritize the combined word
                        choices.insert(0,
                            {
                                "word": x_combined,
                                "meaning": self.meaning_index.meaning_for(x_combined),
                            }
                        )
                        corrected = x_combined
                    else:
                        # If user typed an apostrophe, prioritize the split word
                        choices.append(
                            {
                                "word": x_combined,
                                "meaning": self.meaning_index.meaning_for(x_combined),
                            }
                        )
                
                return FusedPrepositionSuggestion(
                    corrected=corrected,
                    choices=choices,
                )

        if len(remainder) < 3:
            return None

        candidates = self.strict_remainder_candidates(remainder)

        if not candidates and remainder.startswith("l"):
            potential_noun = remainder[1:]
            if self.article_rules.is_noun(potential_noun):
                if prefix in self.APOSTROPHE_PREFIXES and prefix in {"f", "b", "t", "s", "m"}:
                    corrected = f"{prefix}l-{potential_noun}"
                    return FusedPrepositionSuggestion(
                        corrected=corrected,
                        choices=[
                            {
                                "word": corrected,
                                "meaning": self.meaning_index.meaning_for(potential_noun),
                            }
                        ],
                    )

        if not candidates:
            return None

        if prefix in self.APOSTROPHE_PREFIXES:
            if prefix in {"b", "f", "t"}:
                candidates = [
                    candidate
                    for candidate in candidates
                    if self.article_rules.is_noun(candidate)
                ]

            if not candidates:
                return None

            corrected_remainder = candidates[0]
            corrected = f"{prefix}'{corrected_remainder}"
            return FusedPrepositionSuggestion(
                corrected=corrected,
                choices=[
                    {
                        "word": corrected,
                        "meaning": self.meaning_index.meaning_for(corrected_remainder),
                    }
                ],
            )

        if prefix == "m":
            corrected_remainder = candidates[0]
            if self.is_verb(corrected_remainder):
                corrected = f"ma {corrected_remainder}"
            else:
                corrected = f"m'{corrected_remainder}"

            return FusedPrepositionSuggestion(
                corrected=corrected,
                choices=[
                    {
                        "word": corrected,
                        "meaning": self.meaning_index.meaning_for(corrected_remainder),
                    }
                ],
            )

        if prefix == "l":
            noun_candidates = [
                candidate
                for candidate in candidates
                if self.article_rules.is_noun(candidate)
            ]

            if noun_candidates:
                noun = noun_candidates[0]
                choices = self.short_article_choices(noun)
                return FusedPrepositionSuggestion(
                    corrected=choices[0]["word"],
                    choices=choices,
                )

            corrected_remainder = candidates[0]
            corrected = f"li {corrected_remainder}"
            return FusedPrepositionSuggestion(
                corrected=corrected,
                choices=[
                    {
                        "word": corrected,
                        "meaning": self.meaning_index.meaning_for(corrected_remainder),
                    }
                ],
            )

        return None
