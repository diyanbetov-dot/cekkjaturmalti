from __future__ import annotations

from collections import OrderedDict
from functools import lru_cache
from pathlib import Path

try:
    from .suffix_rules import GeneratedSuffixCandidate, MalteseSuffixRules, ParsedSuffix
    from .verb_form_index import MalteseVerbFormIndex, VerbFormRecord
except ImportError:  # pragma: no cover
    from suffix_rules import GeneratedSuffixCandidate, MalteseSuffixRules, ParsedSuffix
    from verb_form_index import MalteseVerbFormIndex, VerbFormRecord


class MalteseSuffixGenerator:
    def __init__(self, *, spellchecker, verbs_file: Path | list[Path]) -> None:
        self.spellchecker = spellchecker
        if isinstance(verbs_file, (list, tuple)):
            self.verbs_files = [Path(path) for path in verbs_file]
        else:
            self.verbs_files = [Path(verbs_file)]
        self.verbs_file = self.verbs_files[0]
        self.verb_index = MalteseVerbFormIndex(
            verbs_file=self.verbs_files,
            grapheme_splitter=spellchecker._graphemes,
            normalizer=spellchecker._normalize_word,
        )
        self.rules = MalteseSuffixRules(spellchecker=spellchecker, verb_index=self.verb_index)
        self._record_cache: OrderedDict[tuple[str, str, str, int], tuple[VerbFormRecord, ...]] = OrderedDict()
        self._candidate_cache: OrderedDict[tuple[str, str, str, int], tuple[GeneratedSuffixCandidate, ...]] = OrderedDict()
        self._cache_limit = 2048
        self.generated_lhom_forms: dict[str, str] = {}
        self.generated_suffix_forms: dict[str, list[GeneratedSuffixCandidate]] = {}

    def _normalize(self, word: str) -> str:
        return self.spellchecker._normalize_word(word)

    def _graphemes(self, word: str) -> list[str]:
        return self.spellchecker._graphemes(word)

    def _from_graphemes(self, graphemes) -> str:
        return self.spellchecker._from_graphemes(graphemes)

    def _cache_get(self, cache: OrderedDict, key):
        if key not in cache:
            return None
        value = cache.pop(key)
        cache[key] = value
        return value

    def _cache_set(self, cache: OrderedDict, key, value) -> None:
        cache[key] = value
        if len(cache) > self._cache_limit:
            cache.popitem(last=False)

    def parse_possible_suffixes(self, word: str) -> list[ParsedSuffix]:
        return self.rules.parse_suffixes(self._normalize(word))

    def has_suffix_parse(self, word: str) -> bool:
        normalized = self._normalize(word)
        if normalized.endswith("x") and len(normalized) > 1:
            normalized = normalized[:-1]
        if normalized.endswith("gh"):
            return False
        return bool(self.parse_possible_suffixes(normalized))

    from functools import lru_cache
    @lru_cache(maxsize=32768)
    def exact_suffix_matches(self, word: str) -> list[GeneratedSuffixCandidate]:
        normalized = self._normalize(word)
        if not normalized:
            return []

        has_negative_x = normalized.endswith("x") and len(normalized) > 1
        suffix_body = normalized[:-1] if has_negative_x else normalized

        if suffix_body.endswith("gh"):
            return []

        matches: list[GeneratedSuffixCandidate] = []
        seen: set[tuple[str, str, str]] = set()

        for parsed in self.parse_possible_suffixes(suffix_body):
            for candidate in self._generated_candidates_for_parse(parsed):
                if candidate.surface != suffix_body:
                    continue

                key = (candidate.surface, candidate.base, candidate.raw_tag)
                if key in seen:
                    continue

                matches.append(candidate)
                seen.add(key)

        return matches

    def _looks_suffix_like(self, word: str) -> bool:
        return self.has_suffix_parse(word)

    def _add_unique_word(self, words: list[str], word: str | None) -> None:
        if not word:
            return
        word = self._normalize(word)
        if word and word not in words:
            words.append(word)

    def _insert_vowel_before_final_consonant(self, word: str) -> list[str]:
        g = self._graphemes(word)
        if len(g) < 2 or not g[-1].isalpha() or g[-1] in self.spellchecker.VOWELS:
            return []
        out: list[str] = []
        for vowel in ("e", "a", "i", "o", "u"):
            self._add_unique_word(out, self._from_graphemes(g[:-1] + [vowel, g[-1]]))
        return out

    def _inverse_base_guesses(self, typed_stem: str) -> list[str]:
        stem = self._normalize(typed_stem)
        return list(self._inverse_base_guesses_cached(stem))

    @lru_cache(maxsize=2048)
    def _inverse_base_guesses_cached(self, stem: str) -> tuple[str, ...]:
        guesses: list[str] = []
        self._add_unique_word(guesses, stem)
        g = self._graphemes(stem)
        orthographic = getattr(self.spellchecker, "orthographic_generator", None)
        doubled = getattr(self.spellchecker, "doubled_letter_generator", None)

        if orthographic is not None:
            for variant in orthographic.strict_lookup_variants(stem):
                self._add_unique_word(guesses, variant)
            for variant in orthographic.insert_token_next_to_vowels(stem, "għ"):
                self._add_unique_word(guesses, variant)
            for variant in orthographic.dictionary_shortcut_variants(stem):
                self._add_unique_word(guesses, variant)
            for variant in orthographic.shortcut_letter_variants(stem, max_changes=2, max_variants=24):
                self._add_unique_word(guesses, variant)

        for candidate in self._insert_vowel_before_final_consonant(stem):
            self._add_unique_word(guesses, candidate)

        if g and g[-1] == "j":
            self._add_unique_word(guesses, self._from_graphemes(g[:-1] + ["i"]))
        if g and g[-1] == "w":
            self._add_unique_word(guesses, self._from_graphemes(g[:-1]))
        if len(g) >= 2 and g[-2:] == ["a", "w"]:
            self._add_unique_word(guesses, self._from_graphemes(g[:-2] + ["għ", "u"]))

        if "għ" not in stem:
            if g and g[0] in self.spellchecker.VOWELS:
                self._add_unique_word(guesses, "għ" + stem)
            if len(g) >= 2 and g[0].isalpha() and g[1] in self.spellchecker.VOWELS:
                self._add_unique_word(guesses, self._from_graphemes(g[:2] + ["għ"] + g[2:]))
            if g and g[0].isalpha():
                self._add_unique_word(guesses, self._from_graphemes(g[:1] + ["għ"] + g[1:]))

        if g and g[-1] == "i":
            without_i = self._from_graphemes(g[:-1])
            self._add_unique_word(guesses, without_i)
            for candidate in self._insert_vowel_before_final_consonant(without_i):
                self._add_unique_word(guesses, candidate)
        if len(g) >= 2 and g[-2] == "i" and g[-1].isalpha():
            self._add_unique_word(guesses, self._from_graphemes(g[:-2] + ["e", g[-1]]))
        if len(g) >= 2 and g[-2:] == ["i", "e"]:
            self._add_unique_word(guesses, self._from_graphemes(g[:-2] + ["a"]))
            if len(g) >= 3:
                self._add_unique_word(guesses, self._from_graphemes(g[:-3] + ["e", g[-3], "a"]))
        if stem.endswith("għ"):
            self._add_unique_word(guesses, stem[:-2] + "'")
        if stem.endswith("għa"):
            self._add_unique_word(guesses, stem[:-3] + "'")

        for guess in list(guesses):
            if orthographic is not None:
                for variant in orthographic.strict_lookup_variants(guess):
                    self._add_unique_word(guesses, variant)
                for variant in orthographic.insert_token_next_to_vowels(guess, "għ"):
                    self._add_unique_word(guesses, variant)
                for variant in orthographic.shortcut_letter_variants(guess, max_changes=2, max_variants=24):
                    self._add_unique_word(guesses, variant)
            for variant in self._insert_vowel_before_final_consonant(guess):
                self._add_unique_word(guesses, variant)
            if doubled is not None:
                for variant in doubled.missing_double_variants(guess):
                    self._add_unique_word(guesses, variant)
            gg = self._graphemes(guess)
            if len(gg) >= 2 and gg[-2] == "i" and gg[-1].isalpha():
                self._add_unique_word(guesses, self._from_graphemes(gg[:-2] + ["e", gg[-1]]))
            if len(gg) >= 3 and gg[:3] == ["w", "e", "ġ"]:
                self._add_unique_word(guesses, self._from_graphemes(["w", "i", "e", *gg[2:]]))

        return tuple(guesses)

    def _parse_specific_guesses(self, parsed: ParsedSuffix) -> list[str]:
        guesses = self._inverse_base_guesses(parsed.typed_stem)
        typed_stem_tokens = self.spellchecker._letter_tokens(parsed.typed_stem)
        typed_stem_starts_cc = (
            len(typed_stem_tokens) >= 2
            and typed_stem_tokens[0] not in self.spellchecker.VOWELS
            and typed_stem_tokens[1] not in self.spellchecker.VOWELS
        )
        if (
            parsed.spec.label.startswith("COMBINED_1P_")
            and (
                parsed.spec.label != "COMBINED_1P_DO3P"
                or typed_stem_starts_cc
            )
        ):
            self._add_unique_word(guesses, parsed.typed_stem + "na")
            for guess in self._inverse_base_guesses(parsed.typed_stem + "na"):
                self._add_unique_word(guesses, guess)
        if parsed.spec.label.startswith("COMBINED_3P_") and parsed.typed_stem.endswith("j"):
            self._add_unique_word(guesses, parsed.typed_stem[:-1])
            for guess in self._inverse_base_guesses(parsed.typed_stem[:-1] + "i"):
                self._add_unique_word(guesses, guess)
        return guesses

    def _deduplicate_records(self, records: list[VerbFormRecord]) -> list[VerbFormRecord]:
        out: list[VerbFormRecord] = []
        seen: set[tuple[str, str]] = set()
        for record in records:
            key = (record.word, record.raw_tag)
            if key not in seen:
                seen.add(key)
                out.append(record)
        return out
    
    def _candidate_records_for_parse(self, parsed: ParsedSuffix, *, max_records: int = 90) -> list[VerbFormRecord]:
        cache_key = (parsed.spec.label, parsed.typed_ending, parsed.typed_stem, max_records)
        cached = self._cache_get(self._record_cache, cache_key)
        if cached is not None:
            return list(cached)
        records: list[VerbFormRecord] = []
        guesses = self._parse_specific_guesses(parsed)
        for guess in guesses:
            records.extend(self.verb_index.word_records(guess))
        anchors: list[str] = []

        def add_anchor(word: str) -> None:
            anchor = self.verb_index.consonant_anchor(word)
            if anchor and anchor not in anchors:
                anchors.append(anchor)

        add_anchor(parsed.typed_stem)
        for guess in guesses:
            add_anchor(guess)
        for anchor in anchors:
            records.extend(self.verb_index.by_anchor.get(anchor, []))
            if len(records) >= max_records:
                break
        if not records and anchors:
            records.extend(self.verb_index.near_anchor_records(anchors[0], max_distance=1, max_records=max_records))
        result = self._deduplicate_records(records)[:max_records]
        self._cache_set(self._record_cache, cache_key, tuple(result))
        return result

    def _generated_candidates_for_parse(self, parsed: ParsedSuffix, *, max_candidates: int = 250) -> list[GeneratedSuffixCandidate]:
        cache_key = (parsed.spec.label, parsed.typed_ending, parsed.typed_stem, max_candidates)
        cached = self._cache_get(self._candidate_cache, cache_key)
        if cached is not None:
            return list(cached)
        candidates: list[GeneratedSuffixCandidate] = []
        seen: set[tuple[str, str, str]] = set()
        direct_record_words = set(self._parse_specific_guesses(parsed))
        for record in self._candidate_records_for_parse(parsed):
            allow_1p_record = (
                parsed.spec.label.startswith("COMBINED_1P_")
                and record.is_perf
                and record.person == "1P"
                and (
                    parsed.spec.label != "COMBINED_1P_DO3P"
                    or self.verb_index.root_class(record).startswith("F1_")
                )
                and (
                    parsed.spec.label != "COMBINED_1P_DO3P"
                    or (
                        len(self.spellchecker._letter_tokens(parsed.typed_stem)) >= 2
                        and self.spellchecker._letter_tokens(parsed.typed_stem)[0] not in self.spellchecker.VOWELS
                        and self.spellchecker._letter_tokens(parsed.typed_stem)[1] not in self.spellchecker.VOWELS
                    )
                )
            )
            if not (
                record.word in direct_record_words
                or allow_1p_record
                or (record.is_perf and record.person in {"3SM", "3P"})
                or (record.is_imp and record.person in {"2S", "2P"})
            ):
                continue
            for candidate in self.rules.generate_for_record_and_spec(record, parsed.spec):
                key = (candidate.surface, candidate.raw_tag, candidate.rule_id)
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(candidate)
                if len(candidates) >= max_candidates:
                    self._cache_set(self._candidate_cache, cache_key, tuple(candidates))
                    return candidates
        self._cache_set(self._candidate_cache, cache_key, tuple(candidates))
        return candidates
    
    def _score_candidate(self, typo: str, candidate: str, stage: str):
        return self._score_candidate_cached(typo, candidate, stage)

    @lru_cache(maxsize=4096)
    def _score_candidate_cached(self, typo: str, candidate: str, stage: str):
        return self.spellchecker._candidate_score(typo, candidate, stage)

    def _direct_guess_rank(self, candidate: GeneratedSuffixCandidate, parsed: ParsedSuffix) -> int | None:
        for index, guess in enumerate(self._parse_specific_guesses(parsed)):
            if candidate.base == guess:
                return index
        return None

    def _person_penalty(self, candidate: GeneratedSuffixCandidate) -> float:
        label = candidate.suffix_label

        if label.startswith(("COMBINED_3SM_", "COMBINED_3SF_")):
            if candidate.person == "3SM":
                return -0.12
            if candidate.tense == "IMP":
                return 0.10

        if label.startswith("COMBINED_1P_"):
            if candidate.person == "3P":
                return 0.20
            if candidate.tense == "IMP":
                return 0.08

        return 0.0

    def _ranking_penalty(self, candidate: GeneratedSuffixCandidate, parsed: ParsedSuffix) -> float:
        penalty = 0.0
        penalty += self._person_penalty(candidate)
        direct_rank = self._direct_guess_rank(candidate, parsed)
        typed_surface = parsed.typed_stem + parsed.typed_ending
        exact_typed_match = candidate.surface in self.spellchecker._strict_lookup_variants(typed_surface)
        if parsed.spec.kind == "IDO" and len(self.spellchecker._letter_tokens(parsed.typed_stem)) < 4:
            penalty += 0.20
        if (
            parsed.spec.kind == "IDO"
            and candidate.tense == "IMP"
            and candidate.rule_id == "DEFAULT_ADD"
        ):
            penalty += 0.18
        if direct_rank is not None:
            penalty -= 0.18
            penalty += direct_rank * 0.003
        if (
            parsed.typed_stem
            and parsed.typed_stem[0] in self.spellchecker.VOWELS
            and candidate.surface.startswith("għ")
            and direct_rank is not None
        ):
            penalty -= 0.08
        if candidate.tense == "IMP" and direct_rank is None:
            penalty += 0.12
        if candidate.rule_id == "PERF_1P_DROP_FINAL_A":
            penalty -= 0.30
        if (
            parsed.spec.kind == "DO_IDO"
            and candidate.person == "3P"
            and candidate.base not in self._parse_specific_guesses(parsed)
        ):
            penalty += 0.22
        if (
            parsed.spec.label == "IDO_3P"
            and parsed.typed_ending in {"ilhom", "ilom"}
            and parsed.typed_stem.endswith("il")
        ):
            penalty += 0.22
        if (
            parsed.spec.label == "DO_3P"
            and parsed.typed_ending in {"hom", "om", "wom"}
            and parsed.typed_stem.endswith(("ul", "hul"))
        ):
            penalty += 0.32
        if (
            parsed.spec.label == "DO_3P"
            and not exact_typed_match
            and (
                (
                    parsed.typed_ending == "hom"
                    and parsed.typed_stem.endswith("il")
                )
                or (
                    parsed.typed_ending in {"om", "wom"}
                    and parsed.typed_stem.endswith("ilh")
                )
            )
        ):
            penalty += 0.72
        if (
            parsed.spec.label == "IDO_3P"
            and parsed.typed_ending == "lhom"
            and parsed.typed_stem.endswith("i")
        ):
            penalty += 0.10
        if (
            parsed.spec.label == "IDO_3P"
            and parsed.typed_ending in {"lom", "lhom"}
            and parsed.typed_stem.endswith(("u", "hu"))
            and direct_rank is None
        ):
            penalty += 0.32
        if (
            parsed.spec.label == "COMBINED_3SF_3P"
            and parsed.typed_ending in {"ilhom", "ilom"}
            and parsed.typed_stem.endswith("il")
        ):
            penalty -= 0.20
        if (
            parsed.spec.label == "COMBINED_3SF_3P"
            and parsed.typed_ending in {"ilhom", "ilom"}
            and not parsed.typed_stem.endswith("il")
        ):
            penalty += 0.48
            if len(self.spellchecker._letter_tokens(parsed.typed_stem)) < 3:
                penalty += 0.30
        if parsed.spec.label == "COMBINED_3SM_3SM" and parsed.typed_ending == "ulu":
            penalty -= 0.25
            if (
                candidate.root_class == "HOLLOW"
                and candidate.rule_id in {"DEFAULT_ADD", "COMBINED_3SM_EC_TO_IC"}
            ):
                penalty += 0.18
        if candidate.rule_id == "IDO_L_FORMS_HOLLOW_ADD_I":
            penalty -= 0.16
            if parsed.typed_stem.endswith("i") or parsed.typed_ending.startswith("il"):
                penalty -= 0.08
        if parsed.spec.label == "DO_3SM" and candidate.rule_id.endswith("DROP_V"):
            penalty += 0.18
        if parsed.spec.label.startswith("COMBINED_3P_") and candidate.rule_id == "DEFAULT_ADD":
            penalty -= 0.06
            base_graphemes = self.spellchecker._graphemes(candidate.base)
            if (
                len(base_graphemes) >= 2
                and base_graphemes[-2] == "e"
                and base_graphemes[-1].isalpha()
                and base_graphemes[-1] not in self.spellchecker.VOWELS
            ):
                penalty += 0.16
        if (
            parsed.typed_stem.endswith("ij")
            and candidate.root == "għtj"
            and candidate.form_class == "F2"
        ):
            penalty += 0.12
        if parsed.spec.label.startswith("COMBINED_3P_") and candidate.rule_id == "COMBINED_3P_EC_TO_IC":
            penalty -= 0.22
            if candidate.root_class == "HOLLOW":
                penalty += 0.34
        if parsed.spec.label.startswith("COMBINED_3P_") and candidate.base in self._parse_specific_guesses(parsed):
            penalty -= 0.26
        if candidate.rule_id == "PERF_3P_WIE_TO_WI":
            penalty -= 0.28
        if candidate.rule_id == "IECEC_TO_ECIC" and candidate.tense == "PERF" and candidate.person == "3P":
            penalty += 0.22
        if candidate.rule_id == "IDO_L_FORMS_DOUBLE_C_ADD_I":
            typed_stem_graphemes = self.spellchecker._graphemes(parsed.typed_stem)
            typed_has_final_double = (
                len(typed_stem_graphemes) >= 2
                and typed_stem_graphemes[-1] == typed_stem_graphemes[-2]
            )
            penalty -= 0.08
            if not typed_has_final_double:
                penalty += 0.16
        if candidate.suffix_label in {"COMBINED_1P_3SM_3P", "COMBINED_1P_3SF_3P"}:
            if "għ" in candidate.surface or "ġ" in candidate.surface:
                penalty -= 0.08
            if candidate.rule_id == "COMBINED_1P_DO_EC_TO_IC":
                penalty -= 0.06
            if "ff" in candidate.base or "ff" in candidate.surface:
                penalty -= 0.16
        if candidate.surface == typed_surface:
            penalty += 0.35
            if all(ord(ch) < 128 for ch in candidate.surface):
                penalty += 0.18
        return penalty
    
    def correct_suffix(self, word: str, *, score_limit: float = 0.46) -> str | None:
        normalized = self._normalize(word)
        if not normalized:
            return None
        has_negative_x = normalized.endswith("x") and len(normalized) > 1
        suffix_body = normalized[:-1] if has_negative_x else normalized
        suffix_tail = "x" if has_negative_x else ""
        if suffix_body.endswith("gh"):
            return None
        parses = self.parse_possible_suffixes(suffix_body)
        if not parses:
            return None
        rows: list[tuple[object, GeneratedSuffixCandidate, ParsedSuffix, float]] = []
        for parsed in parses:
            for candidate in self._generated_candidates_for_parse(parsed):
                for lookup in self.spellchecker._strict_lookup_variants(candidate.surface):
                    if suffix_body == lookup and lookup != candidate.surface:
                        return lookup + suffix_tail
                row = self._score_candidate(suffix_body, candidate.surface, "generated_suffix_parser")
                rows.append((row, candidate, parsed, self._ranking_penalty(candidate, parsed)))
        if not rows:
            return None
        rows.sort(key=lambda item: (item[0].score + item[3], item[0].edit_distance, item[2].priority, item[1].surface, item[1].rule_id))
        max_distance = max(2, self.spellchecker._max_distance(suffix_body))
        for row, candidate, parsed, penalty in rows:
            if candidate.surface == suffix_body and candidate.surface not in self.spellchecker.dictionary_set:
                continue
            if row.edit_distance <= max_distance + 1 and row.score <= score_limit:
                return candidate.surface + suffix_tail
            if (
                (candidate.suffix_label == "COMBINED_1P_3SM_3P" or candidate.suffix_label.startswith("COMBINED_3P_"))
                and row.score <= 0.34
                and row.edit_distance <= max_distance + 3
            ):
                return candidate.surface + suffix_tail
        return None

    def suggest_suffixes(
        self,
        word: str,
        *,
        limit: int = 5,
        score_limit: float = 0.50,
    ) -> list[str]:
        normalized = self._normalize(word)
        if not normalized:
            return []

        has_negative_x = normalized.endswith("x") and len(normalized) > 1
        suffix_body = normalized[:-1] if has_negative_x else normalized
        suffix_tail = "x" if has_negative_x else ""
        if suffix_body.endswith("gh"):
            return []
        parses = self.parse_possible_suffixes(suffix_body)

        if not parses:
            return []

        rows: list[tuple[object, GeneratedSuffixCandidate, ParsedSuffix, float]] = []
        for parsed in parses:
            for candidate in self._generated_candidates_for_parse(parsed):
                row = self._score_candidate(
                    suffix_body,
                    candidate.surface,
                    "generated_suffix_suggest",
                )
                rows.append((row, candidate, parsed, self._ranking_penalty(candidate, parsed)))

        rows.sort(key=lambda item: (
            item[0].score + item[3],
            item[0].edit_distance,
            item[2].priority,
            item[1].surface,
            item[1].rule_id,
        ))

        max_distance = max(2, self.spellchecker._max_distance(suffix_body))
        suggestions: list[str] = []

        def add(candidate: str) -> None:
            candidate = self._normalize(candidate)
            if candidate and candidate not in suggestions:
                suggestions.append(candidate)

        for row, candidate, parsed, _penalty in rows:
            if row.score > score_limit or row.edit_distance > max_distance + 3:
                continue

            if parsed.spec.kind == "DO_IDO" or row.edit_distance <= max_distance + 1:
                add(candidate.surface + suffix_tail)

            if len(suggestions) >= limit:
                break

        return suggestions

    def candidates_for_surface(
        self,
        word: str,
        *,
        limit: int = 8,
    ) -> list[GeneratedSuffixCandidate]:
        normalized = self._normalize(word)
        if not normalized:
            return []

        has_negative_x = normalized.endswith("x") and len(normalized) > 1
        suffix_body = normalized[:-1] if has_negative_x else normalized
        if suffix_body.endswith("gh"):
            return []
        parses = self.parse_possible_suffixes(suffix_body)
        if not parses:
            return []

        matches: list[GeneratedSuffixCandidate] = []
        seen: set[tuple[str, str, str]] = set()

        for parsed in parses:
            for candidate in self._generated_candidates_for_parse(parsed):
                if candidate.surface != suffix_body:
                    continue

                key = (candidate.surface, candidate.base, candidate.raw_tag)
                if key in seen:
                    continue

                matches.append(candidate)
                seen.add(key)

                if len(matches) >= limit:
                    return matches

        return matches

    def correct_lhom(self, word: str) -> str | None:
        if not self._looks_suffix_like(word):
            return None
        return self.correct_suffix(word)

    def debug_lhom(self, word: str, limit: int = 10) -> dict:
        return self.debug_suffix(word, limit=limit)

    def debug_suffix(self, word: str, limit: int = 20) -> dict:
        normalized = self._normalize(word)
        has_negative_x = normalized.endswith("x") and len(normalized) > 1
        suffix_body = normalized[:-1] if has_negative_x else normalized
        suffix_tail = "x" if has_negative_x else ""
        parses = self.parse_possible_suffixes(suffix_body)
        rows: list[tuple[object, GeneratedSuffixCandidate, ParsedSuffix, float]] = []
        parse_debug = []
        for parsed in parses:
            records = self._candidate_records_for_parse(parsed)
            candidates = self._generated_candidates_for_parse(parsed)
            parse_debug.append({
                "suffix_label": parsed.spec.label,
                "suffix_kind": parsed.spec.kind,
                "suffix_display": parsed.spec.display,
                "typed_ending": parsed.typed_ending,
                "typed_stem": parsed.typed_stem,
                "candidate_record_count": len(records),
                "generated_candidate_count": len(candidates),
                "inverse_base_guesses": self._parse_specific_guesses(parsed)[:20],
            })
            for candidate in candidates:
                row = self._score_candidate(suffix_body, candidate.surface, "generated_suffix_parser")
                rows.append((row, candidate, parsed, self._ranking_penalty(candidate, parsed)))
        rows.sort(key=lambda item: (item[0].score + item[3], item[0].edit_distance, item[2].priority, item[1].surface, item[1].rule_id))
        top = []
        for row, candidate, parsed, penalty in rows[:limit]:
            item = candidate.to_debug_dict()
            item.update({
                "score": row.score,
                "edit_distance": row.edit_distance,
                "vowel_slot_score": row.vowel_slot_score,
                "consonant_score": row.consonant_score,
                "ranking_penalty": penalty,
                "matched_parse_ending": parsed.typed_ending,
                "matched_parse_stem": parsed.typed_stem,
            })
            top.append(item)
        return {
            "word": word,
            "normalized": normalized,
            "suffix_body": suffix_body,
            "negative_marker": suffix_tail,
            "corrected": self.correct_suffix(word),
            "parser_mode": "ending-first",
            "verb_files": [str(path) for path in self.verbs_files],
            "verb_record_count": self.verb_index.record_count(),
            "parse_count": len(parses),
            "parses": parse_debug,
            "top_candidates": top,
        }
