from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


NUMERAL_PREFIXES = {
    "wieħed",
    "waħda",
    "żewġ",
    "tnejn",
    "tlieta",
    "erbgħa",
    "ħamsa",
    "ħams",
    "sitta",
    "sitt",
    "sebgħa",
    "seba",
    "tmienja",
    "tmien",
    "disgħa",
    "disgħ",
    "għaxra",
    "għoxrin",
}


@dataclass(frozen=True, slots=True)
class GrammarFinding:
    rule_id: str
    family: str
    action: str
    description_en: str
    production_enabled: bool
    evidence_status: str
    span: tuple[int, int] | None = None
    surface: str | None = None
    suggestion: str | None = None
    suggestions: tuple[str, ...] = ()
    note: str | None = None

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        if payload["span"] is None:
            payload.pop("span")
        if payload["surface"] is None:
            payload.pop("surface")
        if payload["suggestion"] is None:
            payload.pop("suggestion")
        if not payload["suggestions"]:
            payload.pop("suggestions")
        if payload["note"] is None:
            payload.pop("note")
        return payload


class MalteseGrammarRuleEngine:
    def __init__(
        self,
        *,
        rules_path: Path,
        spellchecker,
        meaning_index=None,
        article_rules=None,
    ) -> None:
        self.rules_path = Path(rules_path)
        self.spellchecker = spellchecker
        self.meaning_index = meaning_index
        self.article_rules = article_rules
        self.catalogue = self._load_catalogue()
        self.rules = self.catalogue.get("rules", [])
        self.rules_by_id = {rule["id"]: rule for rule in self.rules}

    def _load_catalogue(self) -> dict:
        return json.loads(self.rules_path.read_text(encoding="utf-8"))

    def rule_ids(self) -> list[str]:
        return [rule["id"] for rule in self.rules]

    def analyze(
        self,
        *,
        text: str,
        request_words: list[str],
        tokens: list[dict],
    ) -> list[dict[str, object]]:
        findings: list[GrammarFinding] = []
        findings.extend(self._subject_verb_person_number(request_words))
        findings.extend(self._verb_chain_person_number(request_words))
        findings.extend(self._definite_np_article_propagation(request_words))
        findings.extend(self._amod_order(request_words))
        findings.extend(self._amod_gender_agreement(request_words))
        findings.extend(self._amod_number_agreement(request_words))
        findings.extend(self._min_minn_context(request_words))
        findings.extend(self._hu_u_hu_confusable(request_words))
        findings.extend(self._negation_ma_x(request_words))
        findings.extend(self._numeral_noun_number(request_words))
        findings.extend(self._verb_adjective_compatibility(request_words))
        findings.extend(self._li_adjective_compatibility(request_words))
        findings.extend(self._preposition_article_contraction(tokens))
        return [finding.as_dict() for finding in findings]

    def apply_safe_rewrites(
        self,
        *,
        original_text: str,
        corrected_text: str,
        tokens: list[dict],
    ) -> tuple[str, list[dict], list[dict[str, object]]]:
        request_words = [
            match.group(0)
            for match in self.spellchecker.WORD_PATTERN.finditer(original_text)
        ]
        findings = [
            finding
            for finding in self.analyze(
                text=original_text,
                request_words=request_words,
                tokens=tokens,
            )
            # Grammar diagnostics are not permission to rewrite text.  The
            # subject scan is deliberately broad and remains suggestion-only;
            # only the explicitly production-enabled verb-chain rule may
            # alter a surface automatically.
            if finding.get("rule_id") == "VERB_VERB_PERSON_NUMBER"
            and finding.get("production_enabled") is True
            and finding.get("suggestion")
            and finding.get("span")
        ]

        rewritten_text = corrected_text
        rewritten_tokens = tokens
        applied: list[dict[str, object]] = []

        for finding in findings:
            span = finding.get("span")
            if not isinstance(span, (list, tuple)) or len(span) != 2:
                continue
            start, end = int(span[0]), int(span[1])
            if end <= start or end > len(request_words):
                continue

            source_word = request_words[end - 1]
            target_word = str(finding["suggestion"]).split()[-1]
            if self._normalized(source_word) == self._normalized(target_word):
                continue

            occurrence = sum(
                1
                for word in request_words[:end]
                if self._normalized(word) == self._normalized(source_word)
            )
            new_text, replaced = self._replace_nth_word(
                rewritten_text,
                source_word=source_word,
                target_word=target_word,
                occurrence=occurrence,
            )
            if not replaced:
                continue

            rewritten_text = new_text
            rewritten_tokens = self._rewrite_token_occurrence(
                rewritten_tokens,
                source_word=source_word,
                target_word=target_word,
                occurrence=occurrence,
            )
            applied.append(finding)

        corrected_words = [
            match.group(0)
            for match in self.spellchecker.WORD_PATTERN.finditer(rewritten_text)
        ]
        agreement_findings = [
            finding
            for finding in self.analyze(
                text=original_text,
                request_words=request_words,
                tokens=rewritten_tokens,
            )
            if finding.get("rule_id") in {
                "AMOD_GENDER_AGREEMENT",
                "AMOD_NUMBER_AGREEMENT",
            }
            and finding.get("production_enabled") is True
            and finding.get("suggestion")
            and finding.get("span")
        ]
        for finding in agreement_findings:
            span = finding.get("span")
            if not isinstance(span, (list, tuple)) or len(span) != 2:
                continue
            start, end = int(span[0]), int(span[1])
            if end <= start or end > len(request_words):
                continue
            suggestion_words = str(finding["suggestion"]).split()
            if len(suggestion_words) != end - start:
                continue
            current_words = [
                match.group(0)
                for match in self.spellchecker.WORD_PATTERN.finditer(rewritten_text)
            ]

            if start < len(current_words):
                original_head = request_words[start]
                current_head = current_words[start]
                if (
                    self._normalized(current_head)
                    != self._normalized(original_head)
                    and original_head
                ):
                    head_occurrence = sum(
                        1
                        for word in current_words[: start + 1]
                        if self._normalized(word) == self._normalized(current_head)
                    )
                    new_text, replaced = self._replace_nth_word(
                        rewritten_text,
                        source_word=current_head,
                        target_word=original_head,
                        occurrence=head_occurrence,
                    )
                    if replaced:
                        rewritten_text = new_text
                        rewritten_tokens = self._rewrite_token_occurrence(
                            rewritten_tokens,
                            source_word=current_head,
                            target_word=original_head,
                            occurrence=head_occurrence,
                        )
                        applied.append(finding)
                        current_words = [
                            match.group(0)
                            for match in self.spellchecker.WORD_PATTERN.finditer(
                                rewritten_text
                            )
                        ]

            source_word = current_words[end - 1]
            target_word = suggestion_words[-1]
            if self._normalized(source_word) == self._normalized(target_word):
                continue

            occurrence = sum(
                1
                for word in current_words[:end]
                if self._normalized(word) == self._normalized(source_word)
            )
            new_text, replaced = self._replace_nth_word(
                rewritten_text,
                source_word=source_word,
                target_word=target_word,
                occurrence=occurrence,
            )
            if not replaced:
                continue

            rewritten_text = new_text
            rewritten_tokens = self._rewrite_token_occurrence(
                rewritten_tokens,
                source_word=source_word,
                target_word=target_word,
                occurrence=occurrence,
            )
            applied.append(finding)
            corrected_words = [
                match.group(0)
                for match in self.spellchecker.WORD_PATTERN.finditer(rewritten_text)
            ]

        return rewritten_text, rewritten_tokens, applied

    def _replace_nth_word(
        self,
        text: str,
        *,
        source_word: str,
        target_word: str,
        occurrence: int,
    ) -> tuple[str, bool]:
        source_norm = self._normalized(source_word)
        seen = 0
        parts: list[str] = []
        cursor = 0
        for match in self.spellchecker.WORD_PATTERN.finditer(text):
            if self._normalized(match.group(0)) != source_norm:
                continue
            seen += 1
            if seen != occurrence:
                continue
            replacement = self.spellchecker._match_capitalisation(
                match.group(0),
                target_word,
            )
            parts.append(text[cursor : match.start()])
            parts.append(replacement)
            parts.append(text[match.end() :])
            return "".join(parts), True
        return text, False

    def _rewrite_token_occurrence(
        self,
        tokens: list[dict],
        *,
        source_word: str,
        target_word: str,
        occurrence: int,
    ) -> list[dict]:
        source_norm = self._normalized(source_word)
        seen = 0
        rewritten: list[dict] = []
        for token in tokens:
            if token.get("type") != "word":
                rewritten.append(token)
                continue
            corrected = str(token.get("corrected", token.get("original", "")))
            if self._normalized(corrected) != source_norm:
                rewritten.append(token)
                continue
            seen += 1
            if seen != occurrence:
                rewritten.append(token)
                continue
            replacement = self.spellchecker._match_capitalisation(
                corrected,
                target_word,
            )
            updated = dict(token)
            updated["corrected"] = replacement
            updated["meaning"] = self.spellchecker.meaning_for(replacement)
            updated["ambiguous"] = False
            updated["choices"] = []
            updated["unrecognized"] = False
            rewritten.append(updated)
        return rewritten

    def _normalized(self, word: str) -> str:
        return self.spellchecker._normalize_word(word)

    def _rule(self, rule_id: str) -> dict[str, object]:
        return self.rules_by_id[rule_id]

    def _base_finding(
        self,
        rule_id: str,
        *,
        span: tuple[int, int] | None = None,
        surface: str | None = None,
        suggestion: str | None = None,
        suggestions: Iterable[str] = (),
        note: str | None = None,
    ) -> GrammarFinding:
        rule = self._rule(rule_id)
        return GrammarFinding(
            rule_id=rule["id"],
            family=rule["family"],
            action=rule["action"],
            description_en=rule["description_en"],
            production_enabled=bool(rule.get("production_enabled", False)),
            evidence_status=rule.get("evidence_status", "seed_only"),
            span=span,
            surface=surface,
            suggestion=suggestion,
            suggestions=tuple(suggestions),
            note=note,
        )

    def _noun_person_number(
        self,
        word: str,
    ) -> tuple[str | None, str | None, str | None]:
        normalized = self._normalized(word)
        if self.spellchecker._capitalized_name_kind(normalized):
            return None, None, None

        prefixes = self._tag_prefixes(normalized)
        if "SINGNOUNF" in prefixes:
            return "F", "SING", None
        if "SINGNOUNM" in prefixes:
            return "M", "SING", None
        if "PLUNOUN" in prefixes or "PAUCNOUN" in prefixes or "COLLNOUN" in prefixes:
            return None, "PLU", None
        return None, None, None

    def _pronoun_person_number(
        self,
        word: str,
    ) -> tuple[str | None, str | None, str | None]:
        normalized = self._normalized(word)
        return {
            "jien": (None, "SING", "1S"),
            "int": (None, "SING", "2S"),
            "inti": (None, "SING", "2S"),
            "hu": ("M", "SING", "3SM"),
            "hi": ("F", "SING", "3SF"),
            "aħna": (None, "PLU", "1P"),
            "ahna": (None, "PLU", "1P"),
            "intom": (None, "PLU", "2P"),
            "huma": (None, "PLU", "3P"),
        }.get(normalized, (None, None, None))

    def _subject_features(
        self,
        word: str,
    ) -> tuple[str | None, str | None, str | None]:
        pronoun = self._pronoun_person_number(word)
        if pronoun != (None, None, None):
            return pronoun

        noun_gender, noun_number, _ = self._noun_person_number(word)
        if noun_number == "SING":
            if noun_gender == "F":
                return "F", "SING", "3SF"
            if noun_gender == "M":
                return "M", "SING", "3SM"
        if noun_number == "PLU":
            return None, "PLU", "3P"
        return None, None, None

    def _tag_prefixes(self, word: str) -> set[str]:
        normalized = self._normalized(word)
        return {
            tag.split("-", 1)[0].upper()
            for tag in self.spellchecker.word_tags.get(normalized, set())
            if tag
        }

    def _is_noun_like(self, word: str) -> bool:
        if self.article_rules is not None and self.article_rules.is_noun(word):
            return True
        return self.spellchecker._is_noun_tagged_word(word)

    def _is_adjective_like(self, word: str) -> bool:
        normalized = self._normalized(word)
        if self.article_rules is not None and self.article_rules.is_adjective(word):
            return True
        if self.spellchecker._is_adjective_tagged_word(word):
            return True
        return normalized in {"hafna", "naqra", "iktar", "inqas", "anqas"}

    def _is_exception_adjective(self, word: str) -> bool:
        normalized = self._normalized(word)
        if normalized in {"hafna", "naqra", "iktar", "inqas", "anqas"}:
            return True
        return any(
            prefix.startswith("EXC") and "ADJ" in prefix
            for prefix in self._tag_prefixes(normalized)
        )

    def _is_comparative_or_superlative_adjective(self, word: str) -> bool:
        normalized = self._normalized(word)
        if not self._is_adjective_like(normalized):
            return False
        if self._is_exception_adjective(normalized):
            return True
        article_rules = self.article_rules
        if article_rules is None:
            return False
        meaning = self.spellchecker.meaning_for(normalized)
        return bool(article_rules._superlative_meaning(meaning))

    def _noun_gender_marker(self, word: str) -> str | None:
        prefixes = self._tag_prefixes(word)
        if any(prefix.startswith("SINGNOUNF") for prefix in prefixes):
            return "F"
        if any(prefix.startswith("SINGNOUNM") for prefix in prefixes):
            return "M"
        return None

    def _adjective_gender_marker(self, word: str) -> str | None:
        prefixes = self._tag_prefixes(word)
        if any(prefix.startswith(("SINGADJF", "PLUADJF")) for prefix in prefixes):
            return "F"
        if any(prefix.startswith(("SINGADJM", "PLUADJM")) for prefix in prefixes):
            return "M"
        return None

    def _adjective_number_marker(self, word: str) -> str | None:
        prefixes = self._tag_prefixes(word)
        if any(prefix.startswith("SINGADJ") for prefix in prefixes):
            return "SING"
        if any(prefix.startswith("PLUADJ") for prefix in prefixes):
            return "PLU"
        return None

    def _adjective_variant_for_noun(
        self,
        adjective: str,
        *,
        noun_gender: str | None = None,
        noun_number: str | None = None,
    ) -> str | None:
        normalized = self._normalized(adjective)
        anchor = self.spellchecker.word_anchors.get(normalized) or self.spellchecker._extract_consonant_anchor(normalized)
        if not anchor:
            return None

        source_meanings = set()
        if self.meaning_index is not None:
            for meaning in self.meaning_index.meanings_for(normalized):
                key = self.spellchecker._english_meaning_key(meaning)
                if key:
                    source_meanings.add(key)

        candidate_pool: list[tuple[int, int, int, str]] = []
        for candidate in self.spellchecker.anchor_map.get(anchor, ()):
            if candidate == normalized:
                continue
            if not self.spellchecker._is_adjective_tagged_word(candidate):
                continue
            if noun_gender is not None and self._adjective_gender_marker(candidate) not in {None, noun_gender}:
                continue
            if noun_number is not None and self._adjective_number_marker(candidate) not in {None, noun_number}:
                continue
            candidate_meanings = set()
            if self.meaning_index is not None:
                for meaning in self.meaning_index.meanings_for(candidate):
                    key = self.spellchecker._english_meaning_key(meaning)
                    if key:
                        candidate_meanings.add(key)
            shares_meaning = bool(source_meanings and candidate_meanings & source_meanings)
            number_bonus = 0 if self._adjective_number_marker(candidate) == noun_number else 1
            gender_bonus = 0 if self._adjective_gender_marker(candidate) == noun_gender else 1
            candidate_pool.append((
                0 if shares_meaning else 1,
                gender_bonus,
                number_bonus,
                candidate,
            ))

        if not candidate_pool:
            return None

        candidate_pool.sort()
        return candidate_pool[0][3]

    def _verb_variant_for_person(self, word: str, target_person: str) -> str | None:
        records = [
            record
            for record in self.spellchecker._verb_records_for_surface(word)
            if record.tense != "IMP"
        ]
        if not records:
            return None

        suffix_generator = getattr(self.spellchecker, "suffix_generator", None)
        verb_index = getattr(suffix_generator, "verb_index", None)
        if verb_index is None:
            return None

        normalized = self._normalized(word)
        candidates: list[tuple[int, int, str]] = []
        for record in records:
            for related in verb_index.by_short_tag.get(record.short_tag, ()):
                if related.tense == "IMP" or related.person != target_person:
                    continue
                if self._normalized(related.word) == normalized:
                    continue
                candidates.append(
                    (
                        0 if related.tense == record.tense else 1,
                        related.line_number,
                        related.word,
                    )
                )

        if not candidates:
            return None

        candidates.sort()
        return candidates[0][2]

    def _is_definite_noun_context(self, request_words: list[str], index: int) -> bool:
        if index <= 0:
            return False
        previous = self._normalized(request_words[index - 1])
        return previous in {"il", "l", "id", "din", "dan", "ic", "iċ", "in", "ir", "is", "it", "ix", "iz", "iż"}

    def _definite_adjective_form(self, noun: str, adjective: str) -> str | None:
        if self.article_rules is None:
            return None
        article = self.article_rules.article_from_previous(noun)
        if not article:
            return None
        normalized = self._normalized(adjective)
        if normalized.startswith(("l-", "il-", "l'")):
            return None
        return f"{article}{adjective}"

    def _subject_verb_person_number(self, request_words: list[str]) -> list[GrammarFinding]:
        if self.article_rules is None:
            return []

        findings: list[GrammarFinding] = []
        article_like = {"il", "l", "id", "din", "dan", "ic", "iċ", "in", "ir", "is", "it", "ix", "iz", "iż"}
        clause_boundaries = {
            "ax",
            "għax",
            "ghax",
            "għaliex",
            "ghaliex",
            "li",
            "jekk",
            "meta",
            "imma",
            "pero",
            "però",
            "u",
            "jew",
            "biex",
            "mela",
        }
        for index, word in enumerate(request_words):
            if not self.spellchecker._is_verb_tagged_word(word):
                continue
            if (
                self.article_rules.is_noun(word)
                or self.spellchecker._is_adverb_tagged_word(word)
                or self.spellchecker._is_preposition_tagged_word(word)
            ):
                continue
            verb_records = [
                record
                for record in self.spellchecker._verb_records_for_surface(word)
                if record.tense != "IMP"
            ]
            if not verb_records:
                continue

            target_persons = {record.person for record in verb_records if record.person}
            if not target_persons:
                continue

            subject_index = None
            subject_features: tuple[str | None, str | None, str | None] | None = None
            for back in range(index - 1, max(-1, index - 6), -1):
                candidate = request_words[back]
                normalized_candidate = self._normalized(candidate)
                if normalized_candidate in clause_boundaries:
                    subject_index = None
                    subject_features = None
                    break
                if normalized_candidate in article_like:
                    continue
                if not (
                    self.article_rules.is_noun(candidate)
                    or self.spellchecker._is_pronoun_tagged_word(candidate)
                    or self.spellchecker._capitalized_name_kind(candidate)
                ):
                    continue
                subject_features = self._subject_features(candidate)
                if subject_features == (None, None, None):
                    continue
                subject_index = back
                break

            if subject_index is None or subject_features is None:
                continue

            _, _, expected_person = subject_features
            if expected_person is None or expected_person in target_persons:
                continue

            suggestion_tail = self._verb_variant_for_person(word, expected_person)
            if suggestion_tail is None:
                continue

            surface = " ".join(request_words[subject_index : index + 1])
            suggestion = " ".join(request_words[subject_index:index] + [suggestion_tail])
            findings.append(
                self._base_finding(
                    "SUBJECT_VERB_PERSON_NUMBER",
                    span=(subject_index, index + 1),
                    surface=surface,
                    suggestion=suggestion,
                    suggestions=(suggestion,),
                    note="A nearby overt subject suggests a different finite-verb person.",
                )
            )
        return findings

    def _verb_chain_person_number(self, request_words: list[str]) -> list[GrammarFinding]:
        """Keep adjacent finite verbs aligned with the established subject form."""
        findings: list[GrammarFinding] = []
        for index, head in enumerate(request_words[:-1]):
            tail = request_words[index + 1]
            head_records = [
                record
                for record in self.spellchecker._verb_records_for_surface(head)
                if record.tense != "IMP" and record.person
            ]
            tail_records = [
                record
                for record in self.spellchecker._verb_records_for_surface(tail)
                if record.tense != "IMP" and record.person
            ]
            if not head_records or not tail_records:
                continue

            # The production rewrite is reserved for the clear auxiliary-like
            # pattern used by ``marret jixtri``: a perfect form followed by an
            # imperfect form.  Other adjacent verbs need syntactic context and
            # are reported only as diagnostics.
            if not any(record.is_perf for record in head_records):
                continue
            if not any(record.is_mperf for record in tail_records):
                continue

            head_persons = {record.person for record in head_records}
            tail_persons = {record.person for record in tail_records}
            if len(head_persons) != 1:
                continue
            target_person = next(iter(head_persons))
            if target_person in tail_persons:
                continue

            suggestion_tail = self._verb_variant_for_person(tail, target_person)
            if suggestion_tail is None:
                continue
            suggestion = f"{head} {suggestion_tail}"
            findings.append(
                self._base_finding(
                    "VERB_VERB_PERSON_NUMBER",
                    span=(index, index + 2),
                    surface=f"{head} {tail}",
                    suggestion=suggestion,
                    suggestions=(suggestion,),
                    note="Adjacent finite verbs normally share the same subject features.",
                )
            )
        return findings

    def _definite_np_article_propagation(self, request_words: list[str]) -> list[GrammarFinding]:
        if self.article_rules is None:
            return []

        findings: list[GrammarFinding] = []
        for index, word in enumerate(request_words[:-1]):
            next_word = request_words[index + 1]
            if not self._is_noun_like(word):
                continue
            if not self._is_definite_noun_context(request_words, index):
                continue
            if not self._is_adjective_like(next_word):
                continue
            suggestion_tail = self._definite_adjective_form(word, next_word)
            if suggestion_tail is None:
                continue
            article_word = request_words[index - 1]
            article_noun = self.article_rules.corrected_article_phrase(
                article_word,
                word,
                request_words[index - 2] if index >= 2 else None,
            )
            suggestion = f"{article_noun} {suggestion_tail}"
            findings.append(
                self._base_finding(
                    "DEF_NP_ARTICLE_PROPAGATION",
                    span=(index - 1, index + 2),
                    surface=f"{article_word} {word} {next_word}",
                    suggestion=suggestion,
                    suggestions=(suggestion,),
                    note="A definite noun phrase propagates definiteness to the following adjective.",
                )
            )
        return findings

    def _amod_gender_agreement(self, request_words: list[str]) -> list[GrammarFinding]:
        if self.article_rules is None:
            return []
        findings: list[GrammarFinding] = []
        for index, word in enumerate(request_words[:-1]):
            next_word = request_words[index + 1]
            if not self._is_noun_like(word):
                continue
            if self.spellchecker._capitalized_name_kind(word):
                continue
            if not self._is_adjective_like(next_word):
                continue
            noun_gender = self._noun_gender_marker(word)
            adj_gender = self._adjective_gender_marker(next_word)
            if not noun_gender or not adj_gender or noun_gender == adj_gender:
                continue
            suggestion_tail = self._adjective_variant_for_noun(next_word, noun_gender=noun_gender)
            suggestion = f"{word} {suggestion_tail}" if suggestion_tail else None
            findings.append(
                self._base_finding(
                    "AMOD_GENDER_AGREEMENT",
                    span=(index, index + 2),
                    surface=f"{word} {next_word}",
                    suggestion=suggestion,
                    suggestions=(suggestion,) if suggestion else (),
                    note="Both noun and adjective expose gender; the adjective should match the noun.",
                )
            )
        return findings

    def _amod_number_agreement(self, request_words: list[str]) -> list[GrammarFinding]:
        if self.article_rules is None:
            return []
        findings: list[GrammarFinding] = []
        for index, word in enumerate(request_words[:-1]):
            next_word = request_words[index + 1]
            if not self._is_noun_like(word):
                continue
            if self.spellchecker._capitalized_name_kind(word):
                continue
            if not self._is_adjective_like(next_word):
                continue
            noun_number = "PLU" if self.spellchecker._is_plural_like_noun(word) else "SING" if "SINGNOUN" in self.spellchecker._noun_number_markers(word) else None
            adj_number = self._adjective_number_marker(next_word)
            if not noun_number or not adj_number or noun_number == adj_number:
                continue
            suggestion_tail = self._adjective_variant_for_noun(next_word, noun_number=noun_number)
            suggestion = f"{word} {suggestion_tail}" if suggestion_tail else None
            findings.append(
                self._base_finding(
                    "AMOD_NUMBER_AGREEMENT",
                    span=(index, index + 2),
                    surface=f"{word} {next_word}",
                    suggestion=suggestion,
                    suggestions=(suggestion,) if suggestion else (),
                    note="Both noun and adjective expose number; the adjective should match the noun.",
                )
            )
        return findings

    def _min_minn_context(self, request_words: list[str]) -> list[GrammarFinding]:
        findings: list[GrammarFinding] = []
        for index, word in enumerate(request_words):
            if self._normalized(word) != "min":
                continue
            next_word = request_words[index + 1] if index + 1 < len(request_words) else ""
            next_kind = self.spellchecker._capitalized_name_kind(next_word)
            next_is_place = bool(
                next_word
                and self.spellchecker._normalize_word(next_word)
                in getattr(self.spellchecker, "place_word_set", set())
            )
            if next_kind or next_is_place or self.spellchecker._is_initial_capitalized(next_word):
                ranked = ("minn", "min")
                note = "Capitalized or place-like context prefers minn."
            else:
                ranked = ("min", "minn")
                note = "Default context keeps min first."
            findings.append(
                self._base_finding(
                    "MIN_MINN_CONTEXT",
                    span=(index, index + 1),
                    surface=word,
                    suggestion=ranked[0],
                    suggestions=ranked,
                    note=note,
                )
            )
        return findings

    def _amod_order(self, request_words: list[str]) -> list[GrammarFinding]:
        if self.article_rules is None:
            return []
        findings: list[GrammarFinding] = []
        for index, word in enumerate(request_words[:-1]):
            next_word = request_words[index + 1]
            if not self._is_adjective_like(word):
                continue
            if not (
                self._is_noun_like(next_word)
                or self.spellchecker._capitalized_name_kind(next_word)
            ):
                continue
            findings.append(
                self._base_finding(
                    "AMOD_ORDER_NOUN_BEFORE_ADJ",
                    span=(index, index + 2),
                    surface=f"{word} {next_word}",
                    suggestion=f"{next_word} {word}",
                    suggestions=(f"{next_word} {word}", f"{word} {next_word}"),
                    note="Attributive adjective follows the noun in this rule family.",
                )
            )
        return findings

    def _hu_u_hu_confusable(self, request_words: list[str]) -> list[GrammarFinding]:
        findings: list[GrammarFinding] = []
        for index, word in enumerate(request_words):
            normalized = self._normalized(word)
            if normalized not in {"hu", "u", "ħu"}:
                continue
            if normalized == "hu":
                ranked = ("hu", "u", "ħu")
            elif normalized == "u":
                ranked = ("u", "hu", "ħu")
            else:
                ranked = ("ħu", "hu", "u")
            findings.append(
                self._base_finding(
                    "HU_U_HU_CONFUSABLE",
                    span=(index, index + 1),
                    surface=word,
                    suggestion=ranked[0],
                    suggestions=ranked,
                    note="Contextual ranking is heuristic in the current grammar layer.",
                )
            )
        return findings

    def _negation_ma_x(self, request_words: list[str]) -> list[GrammarFinding]:
        findings: list[GrammarFinding] = []
        normalized_words = [self._normalized(word) for word in request_words]
        for index, normalized in enumerate(normalized_words):
            if normalized != "ma":
                continue
            window = normalized_words[index + 1 : index + 5]
            if any(word.endswith("x") for word in window):
                findings.append(
                    self._base_finding(
                        "NEGATION_MA_X",
                        span=(index, min(len(request_words), index + 5)),
                        surface=request_words[index],
                        suggestion="ma ... x",
                        suggestions=("ma", "x"),
                        note="Discontinuous negation marker pair detected.",
                    )
                )
        return findings

    def _numeral_noun_number(self, request_words: list[str]) -> list[GrammarFinding]:
        findings: list[GrammarFinding] = []
        if self.article_rules is None:
            return findings
        for index, word in enumerate(request_words[:-1]):
            next_word = request_words[index + 1]
            normalized = self._normalized(word)
            if not self.article_rules.is_num(word):
                corrected_word = self._normalized(self.spellchecker.correct_word(word))
                if corrected_word and self.article_rules.is_num(corrected_word):
                    normalized = corrected_word
                else:
                    continue
            if not self._is_noun_like(next_word):
                continue
            if normalized.endswith("t") or normalized in {"ħamest", "għoxrin"}:
                expected = "singular"
            elif normalized in {"ewġ", "żewġ"}:
                expected = "plural_or_paucal"
            else:
                expected = "plural_or_paucal"
            findings.append(
                self._base_finding(
                    "NUMERAL_NOUN_NUMBER",
                    span=(index, index + 2),
                    surface=f"{word} {next_word}",
                    suggestion=expected,
                    suggestions=(expected,),
                    note="Numeral-noun number agreement heuristic.",
                )
            )
        return findings

    def _verb_records_for_grammar_surface(self, word: str):
        normalized = self._normalized(word)
        records = self.spellchecker._verb_records_for_surface(normalized)
        if records:
            return records
        if len(normalized) > 2 and normalized.startswith("i"):
            rest = normalized[1:]
            for candidate in (rest, f"j{rest}"):
                records = self.spellchecker._verb_records_for_surface(candidate)
                if records:
                    return records
        if len(normalized) > 2 and normalized.startswith("u"):
            records = self.spellchecker._verb_records_for_surface(f"w{normalized[1:]}")
            if records:
                return records
        return []

    def _verb_allows_predicative_adjective(self, verb: str, adjective: str) -> bool:
        records = self._verb_records_for_grammar_surface(verb)
        if not records:
            return False

        normalized_adj = self._normalized(adjective)
        if self._is_exception_adjective(normalized_adj):
            return True

        for record in records:
            root = getattr(record, "root", "")
            form_class = getattr(record, "form_class", "")
            if root == "dwm":
                return normalized_adj in {"itwal", "iqsar"}
            if form_class == "F1" and root in {"kbr", "sjr", "qwm", "kwn", "rqd", "mwt"}:
                return True
            if form_class == "F1" and len(root) == 3 and root not in {"rjq"}:
                return True
            if root == "wld" and form_class.startswith("F6"):
                return True
        return False

    def _verb_adjective_compatibility(self, request_words: list[str]) -> list[GrammarFinding]:
        findings: list[GrammarFinding] = []
        for index, verb in enumerate(request_words[:-1]):
            adjective = request_words[index + 1]
            if not self._is_adjective_like(adjective):
                continue
            if not self._verb_records_for_grammar_surface(verb):
                continue
            if self._is_noun_like(verb):
                continue
            if self._verb_allows_predicative_adjective(verb, adjective):
                continue
            suggestion = f"{verb} u jsir {adjective}"
            findings.append(
                self._base_finding(
                    "VERB_ADJECTIVE_COMPATIBILITY",
                    span=(index, index + 2),
                    surface=f"{verb} {adjective}",
                    suggestion=suggestion,
                    suggestions=(suggestion,),
                    note="Forsi ridt tfisser: Verb u jsir ADJ.",
                )
            )
        return findings

    def _li_adjective_compatibility(self, request_words: list[str]) -> list[GrammarFinding]:
        """Flag li + ordinary adjective before it is mistaken for a relative."""
        findings: list[GrammarFinding] = []
        for index, word in enumerate(request_words[:-1]):
            if self._normalized(word) != "li":
                continue
            adjective = request_words[index + 1]
            if not self._is_adjective_like(adjective):
                continue
            if self._is_comparative_or_superlative_adjective(adjective):
                continue
            findings.append(
                self._base_finding(
                    "LI_ADJECTIVE_COMPATIBILITY",
                    span=(index, index + 2),
                    surface=f"{word} {adjective}",
                    suggestion=f"l-{adjective}",
                    suggestions=(f"l-{adjective}",),
                    note="li cannot directly introduce an ordinary adjective.",
                )
            )
        return findings

    def _preposition_article_contraction(self, tokens: list[dict]) -> list[GrammarFinding]:
        findings: list[GrammarFinding] = []
        for index, token in enumerate(tokens):
            if token.get("type") != "phrase":
                continue
            original = str(token.get("original", ""))
            corrected = str(token.get("corrected", ""))
            if " " not in original:
                continue
            if "-" not in corrected and "'" not in corrected:
                continue
            findings.append(
                self._base_finding(
                    "PREP_ARTICLE_CONTRACTION",
                    span=(index, index + 1),
                    surface=original,
                    suggestion=corrected,
                    suggestions=(corrected,),
                    note="Phrase contraction recognized from spellchecker output.",
                )
            )
        return findings
