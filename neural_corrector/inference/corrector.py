from __future__ import annotations

import json
import re
import time
from difflib import SequenceMatcher
from itertools import combinations, product
from pathlib import Path

import torch

from neural_corrector.inference.dictionary_index import (
    TAG_LONG_ATTRIBUTIVE_NUMBER,
    TAG_NOUN,
    TAG_NUMBER,
    TAG_PLURAL_NOUN,
    TAG_SHORT_ATTRIBUTIVE_NUMBER,
    WORD_RE,
    DictionaryIndex,
    fuzzy_key,
    normalize_key,
)
from neural_corrector.inference.edits import structured_edits
from neural_corrector.inference.suffix_bloom import SuffixBloomIndex
from neural_corrector.models.alignment import (
    COPY_ACTION,
    apply_actions,
    derive_actions,
    render_action,
)
from neural_corrector.models.char_edit_tagger import CharEditTagger
from neural_corrector.models.vocab import UNK_CHAR, Vocabularies


class NeuralCorrector:
    INDIRECT_OBJECT_ENDINGS = (
        "lhom",
        "lkom",
        "lek",
        "lha",
        "lna",
        "li",
        "lu",
    )

    def __init__(
        self,
        artifact_dir: Path,
        threshold: float | None = None,
        device: str | None = None,
        use_dictionary_validation: bool | None = None,
        dictionary_index_path: Path | None = None,
        use_suffix_validation: bool | None = None,
        suffix_index_path: Path | None = None,
        _allow_suffix_specialist: bool = True,
    ) -> None:
        self.artifact_dir = Path(artifact_dir)
        self.vocab = Vocabularies.load(self.artifact_dir / "vocab.json")
        self.inverse_actions = self.vocab.inverse_actions
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        checkpoint = torch.load(
            self.artifact_dir / "model.pt",
            map_location=self.device,
            weights_only=False,
        )
        config = checkpoint["config"]
        inference_config_path = self.artifact_dir / "inference_config.json"
        inference_config = (
            json.loads(inference_config_path.read_text(encoding="utf-8"))
            if inference_config_path.exists()
            else {}
        )
        self.threshold = float(
            threshold
            if threshold is not None
            else inference_config.get(
                "action_threshold", config["inference_action_threshold"]
            )
        )
        self.dictionary_validation_enabled = bool(
            use_dictionary_validation
            if use_dictionary_validation is not None
            else inference_config.get(
                "use_dictionary_validation",
                config.get("use_dictionary_validation", False),
            )
        )
        self.dictionary_rescue_min_confidence = float(
            inference_config.get("dictionary_rescue_min_confidence", 0.85)
        )
        configured_index = inference_config.get(
            "dictionary_index",
            "neural_corrector/data/indexes/maltese_dictionary.sqlite3",
        )
        resolved_index = Path(dictionary_index_path or configured_index)
        if not resolved_index.is_absolute():
            resolved_index = Path(__file__).resolve().parents[2] / resolved_index
        self.dictionary_index = (
            DictionaryIndex(resolved_index)
            if self.dictionary_validation_enabled
            else None
        )
        self.suffix_validation_enabled = bool(
            use_suffix_validation
            if use_suffix_validation is not None
            else inference_config.get("use_suffix_validation", False)
        )
        configured_suffix_index = inference_config.get(
            "suffix_index",
            "neural_corrector/data/indexes/maltese_suffix_forms.bloom",
        )
        resolved_suffix_index = Path(
            suffix_index_path or configured_suffix_index
        )
        if not resolved_suffix_index.is_absolute():
            resolved_suffix_index = (
                Path(__file__).resolve().parents[2] / resolved_suffix_index
            )
        self.suffix_index = (
            SuffixBloomIndex(resolved_suffix_index)
            if self.suffix_validation_enabled
            else None
        )
        self.suffix_specialist_min_margin = float(
            inference_config.get("suffix_specialist_min_margin", 0.05)
        )
        specialist_artifact = inference_config.get(
            "suffix_specialist_artifact"
        )
        self.suffix_specialist = None
        if _allow_suffix_specialist and specialist_artifact:
            specialist_path = Path(specialist_artifact)
            if not specialist_path.is_absolute():
                specialist_path = (
                    Path(__file__).resolve().parents[2] / specialist_path
                )
            self.suffix_specialist = NeuralCorrector(
                specialist_path,
                dictionary_index_path=resolved_index,
                suffix_index_path=resolved_suffix_index,
                _allow_suffix_specialist=False,
            )
        self.max_length = int(config["max_sequence_length"])
        self.model_version = checkpoint["model_version"]
        self.model = CharEditTagger(
            len(self.vocab.characters),
            len(self.vocab.actions),
            config["embedding_dim"],
            config["hidden_dim"],
            config["layers"],
            config["dropout"],
        ).to(self.device)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()

    def _dictionary_rescue(
        self,
        original: str,
        corrected: str,
        position_candidates: list[list[tuple[str, float]]],
    ) -> tuple[str, list[dict]]:
        if self.dictionary_index is None:
            return corrected, []
        original_words = list(WORD_RE.finditer(original))
        corrected_words = list(WORD_RE.finditer(corrected))
        if len(original_words) == len(corrected_words):
            aligned_words = list(zip(original_words, corrected_words))
        else:
            aligned_words = []
            corrected_cursor = 0
            for source_match in original_words:
                source = source_match.group(0)
                corrected_match = re.search(
                    rf"(?<!\w){re.escape(source)}(?!\w)",
                    corrected[corrected_cursor:],
                    flags=re.IGNORECASE,
                )
                if corrected_match is None:
                    continue
                absolute_start = corrected_cursor + corrected_match.start()
                absolute_end = corrected_cursor + corrected_match.end()
                absolute_match = re.search(
                    re.escape(corrected[absolute_start:absolute_end]),
                    corrected[absolute_start:absolute_end],
                )
                if absolute_match is None:
                    continue
                # re.Match offsets are relative, so retain the absolute span
                # separately while keeping the source match for neural scores.
                aligned_words.append(
                    (
                        source_match,
                        (absolute_start, absolute_end, source),
                    )
                )
                corrected_cursor = absolute_end

        replacements: list[tuple[int, int, str]] = []
        decisions: list[dict] = []
        for source_match, corrected_word in aligned_words:
            source = source_match.group(0)
            if isinstance(corrected_word, tuple):
                corrected_start, corrected_end, existing = corrected_word
            else:
                corrected_start = corrected_word.start()
                corrected_end = corrected_word.end()
                existing = corrected_word.group(0)
            if (
                normalize_key(existing) != normalize_key(source)
                and self.dictionary_index.contains_surface_form(existing)
            ):
                continue

            local_options: list[tuple[int, str, float]] = []
            for local_position, global_position in enumerate(
                range(source_match.start(), source_match.end())
            ):
                action, probability = position_candidates[global_position][0]
                if (
                    action != COPY_ACTION
                    and probability >= self.dictionary_rescue_min_confidence
                ):
                    local_options.append(
                        (local_position, action, probability)
                    )
            if not local_options:
                continue

            valid_candidates: dict[str, tuple[float, int]] = {}
            for change_count in (1, 2):
                for selected in combinations(local_options, change_count):
                    local_actions = [COPY_ACTION] * len(source)
                    probabilities = []
                    for position, action, probability in selected:
                        local_actions[position] = action
                        probabilities.append(probability)
                    candidate = apply_actions(source, local_actions)
                    if (
                        normalize_key(candidate) == normalize_key(source)
                        or not WORD_RE.fullmatch(candidate)
                        or not (
                            self.dictionary_index.contains_surface_form(
                                candidate
                            )
                            or (
                                self.suffix_index is not None
                                and self._validate_generated_suffix(
                                    candidate, source
                                )
                            )
                        )
                    ):
                        continue
                    score = sum(probabilities) / len(probabilities)
                    previous = valid_candidates.get(candidate)
                    if previous is None or (score, -change_count) > (
                        previous[0],
                        -previous[1],
                    ):
                        valid_candidates[candidate] = (score, change_count)

            if not valid_candidates:
                continue
            candidate, (score, change_count) = max(
                valid_candidates.items(),
                key=lambda item: (item[1][0], -item[1][1], item[0]),
            )
            replacements.append(
                (
                    corrected_start,
                    corrected_end,
                    candidate,
                )
            )
            decisions.append(
                {
                    "source": source,
                    "previous_candidate": existing,
                    "candidate": candidate,
                    "confidence": round(score, 4),
                    "changed_positions": change_count,
                    "decision": "accept_dictionary_rescue",
                }
            )

        rescued = corrected
        for start, end, replacement in reversed(replacements):
            rescued = rescued[:start] + replacement + rescued[end:]
        return rescued, decisions

    @classmethod
    def _suffix_repair_has_evidence(
        cls, source: str, candidate: str
    ) -> bool:
        return (
            any(character in candidate for character in "ċġħżĊĠĦŻ'")
            or candidate.casefold().count("h")
            > source.casefold().count("h")
        )

    def _validate_generated_suffix(
        self, candidate: str, source: str
    ) -> bool:
        if not self._suffix_form_exists(candidate, source):
            return False
        source_key = normalize_key(source)
        candidate_key = normalize_key(candidate)
        if (
            "ie" in source_key
            and source_key.replace("ie", "i") == candidate_key
            and candidate_key.endswith(self.INDIRECT_OBJECT_ENDINGS)
        ):
            return False
        return True

    def _suffix_form_exists(self, candidate: str, source: str) -> bool:
        if self.suffix_index is None:
            return False
        if self.suffix_index.contains(candidate):
            return True
        candidate_key = normalize_key(candidate)
        source_key = normalize_key(source)
        if (
            not source_key.startswith("i")
            or not candidate_key.startswith("i")
            or len(candidate_key) < 4
            or not self.suffix_index.contains(candidate_key[1:])
        ):
            return False
        first, second = candidate_key[1], candidate_key[2]
        vowels = set("aeiou")
        return (
            first not in vowels
            and second not in vowels
            and (first == second or first in "nlmr")
        )

    def _phonological_prefix_rescue(
        self,
        original: str,
        corrected: str,
        position_candidates: list[list[tuple[str, float]]],
    ) -> tuple[str, list[dict]]:
        """Apply general Maltese phonological expansion hypotheses to each
        unrecognised word and validate candidates against the suffix bloom
        filter.

        Rules (all bloom-validated, no hardcoded word lists):
          P1 - Insert 'għ' after a leading vowel when the word looks like an
               epenthetically-shortened verb prefix (e.g. 'amilulu' →
               'agħmilhulu' via 'a' + 'għ' + stem).
          P2 - Prepend 'agħ' to words starting with consonant clusters that
               resemble a stripped agħ- prefix.
          P3 - Insert 'h' before 'om'/'ul'/'a' suffix sequences that typically
               carry an 'h' in formal Maltese spelling.
          P4 - Expand 'i' to 'ie' before indirect-object endings.

        Each accepted candidate must pass ``_suffix_form_exists`` AND show
        non-trivial neural support (> 0.05) to be admitted.
        """
        if self.suffix_index is None:
            return corrected, []

        original_words = list(WORD_RE.finditer(original))
        corrected_words = list(WORD_RE.finditer(corrected))
        if len(original_words) != len(corrected_words):
            return corrected, []

        replacements: list[tuple[int, int, str]] = []
        decisions: list[dict] = []

        for source_match, corrected_match in zip(original_words, corrected_words):
            source = source_match.group(0)
            existing = corrected_match.group(0)

            # Skip already-recognised words and very short inputs.
            if (
                len(source) < 4
                or self.suffix_index.contains(source)
                or self.suffix_index.contains(existing)
                or (self.dictionary_index is not None
                    and self.dictionary_index.contains_surface_form(existing))
            ):
                continue

            hypotheses: list[str] = []
            src_lower = source.casefold()

            # P1: leading vowel + consonant → try inserting 'għ' after vowel or replacing leading 'a' with 'agħ'
            if len(src_lower) > 3 and src_lower[0] in "aeiou" and src_lower[1] not in "aeiou":
                hypotheses.append(source[0] + "għ" + source[1:])
                if src_lower[0] == "a":
                    hypotheses.append("agħ" + source[1:])

            # P2: starts with consonant cluster, no leading vowel →
            #     try prepending 'agħ' (for agħ- prefix collapse)
            if (
                len(src_lower) > 4
                and src_lower[0] not in "aeiou"
                and src_lower[:2] not in ("il", "it", "is", "in")
            ):
                hypotheses.append("agħ" + source)

            # P3: 'om' / 'omlu' / 'oma' without preceding 'h' →
            #     insert 'h' before each 'om' occurrence
            for marker in ("om",):
                pos = src_lower.find(marker)
                while pos > 0 and pos < len(source) - 1:
                    if src_lower[pos - 1] != "h":
                        hypotheses.append(source[:pos] + "h" + source[pos:])
                    pos = src_lower.find(marker, pos + 1)

            # P4: 'i' before indirect-object ending → try expanding to 'ie'
            for ending in ("lu", "lha", "lna", "lkom", "lhom", "lom"):
                if src_lower.endswith("i" + ending):
                    prefix_end = len(source) - len(ending) - 1
                    hypotheses.append(source[:prefix_end] + "ie" + source[prefix_end + 1:])

            # Chain generated hypotheses through suffix 'h' restoration:
            # - For 'om': insert 'h' before 'om' (om → hom)
            # - For 'lu'/'la'/'na': insert 'h' after 'l' (lu → lhu/lhulu, la → lha)
            chained: list[str] = list(hypotheses)
            for hyp in hypotheses:
                hyp_lower = hyp.casefold()
                for marker in ("om",):
                    pos = hyp_lower.find(marker)
                    while pos > 0 and pos < len(hyp) - 1:
                        if hyp_lower[pos - 1] != "h":
                            chained.append(hyp[:pos] + "h" + hyp[pos:])
                        pos = hyp_lower.find(marker, pos + 1)
                for marker in ("lu", "la", "na"):
                    pos = hyp_lower.find(marker)
                    while pos > 0 and pos < len(hyp) - 1:
                        if hyp_lower[pos - 1] == "l" and (pos < 2 or hyp_lower[pos - 2] != "h"):
                            chained.append(hyp[:pos] + "h" + hyp[pos:])
                        elif hyp_lower[pos] == "l" and pos + 1 < len(hyp) and hyp_lower[pos + 1] in ("u", "a", "i"):
                            if hyp_lower[pos - 1] != "h":
                                chained.append(hyp[: pos + 1] + "h" + hyp[pos + 1 :])
                        pos = hyp_lower.find(marker, pos + 1)
            hypotheses = chained

            if not hypotheses:
                continue

            # Validate and score each hypothesis
            best_candidate: str | None = None
            best_support: float = 0.05  # minimum bar

            for hyp in dict.fromkeys(hypotheses):  # deduplicate, preserve order
                if not WORD_RE.fullmatch(hyp):
                    continue
                if not self._suffix_form_exists(hyp, source):
                    continue
                if not self._suffix_repair_has_evidence(source, hyp):
                    continue
                support = self._neural_action_support(
                    source, hyp, source_match.start(), position_candidates
                )
                if support > best_support:
                    best_support = support
                    best_candidate = hyp

            if best_candidate is None:
                continue

            replacements.append(
                (corrected_match.start(), corrected_match.end(), best_candidate)
            )
            decisions.append(
                {
                    "source": source,
                    "previous_candidate": existing,
                    "candidate": best_candidate,
                    "neural_support": round(best_support, 4),
                    "decision": "accept_phonological_prefix_rescue",
                }
            )

        rescued = corrected
        for start, end, replacement in reversed(replacements):
            rescued = rescued[:start] + replacement + rescued[end:]
        return rescued, decisions

    @staticmethod
    def _uses_long_attributive_number(noun: str) -> bool:
        letters = fuzzy_key(noun)
        vowels = set("aeiou")
        return (
            len(letters) >= 3
            and letters[0] == "i"
            and letters[1] not in vowels
            and letters[2] not in vowels
        )

    def _dictionary_context_rescue(
        self, original: str, corrected: str
    ) -> tuple[str, list[dict]]:
        if self.dictionary_index is None:
            return corrected, []
        original_words = list(WORD_RE.finditer(original))
        corrected_words = list(WORD_RE.finditer(corrected))
        if len(original_words) != len(corrected_words):
            return corrected, []

        replacements: list[tuple[int, int, str]] = []
        decisions: list[dict] = []
        for index, (source_match, corrected_match) in enumerate(
            zip(original_words, corrected_words)
        ):
            if index + 1 >= len(original_words):
                break
            source = source_match.group(0)
            existing = corrected_match.group(0)
            if self.dictionary_index.contains_surface_form(source):
                continue

            next_word = corrected_words[index + 1].group(0)
            next_entry = self.dictionary_index.lookup(next_word)
            if next_entry is None or not (next_entry.tag_bits & TAG_NOUN):
                continue

            required_tag = 0
            if next_entry.tag_bits & TAG_PLURAL_NOUN:
                required_tag = (
                    TAG_LONG_ATTRIBUTIVE_NUMBER
                    if self._uses_long_attributive_number(next_word)
                    else TAG_SHORT_ATTRIBUTIVE_NUMBER
                )
            if not required_tag:
                continue

            candidates = [
                (entry, distance)
                for entry, distance in self.dictionary_index.nearby_with_tags(
                    source, TAG_NUMBER, 2
                )
                if entry.tag_bits & required_tag
            ]
            if not candidates:
                continue
            candidate, distance = candidates[0]
            replacement_end = corrected_match.end()
            if (
                candidate.canonical.endswith("'")
                and corrected[replacement_end : replacement_end + 1]
                in {"'", "’"}
            ):
                replacement_end += 1
            replacements.append(
                (
                    corrected_match.start(),
                    replacement_end,
                    candidate.canonical,
                )
            )
            decisions.append(
                {
                    "source": source,
                    "previous_candidate": existing,
                    "candidate": candidate.canonical,
                    "following_word": next_word,
                    "distance": distance,
                    "decision": "accept_contextual_number_candidate",
                }
            )

        rescued = corrected
        for start, end, replacement in reversed(replacements):
            rescued = rescued[:start] + replacement + rescued[end:]
        return rescued, decisions

    def _suffix_candidate_rescue(
        self,
        original: str,
        corrected: str,
        position_candidates: list[list[tuple[str, float]]],
    ) -> tuple[str, list[dict]]:
        if self.suffix_index is None:
            return corrected, []
        original_words = list(WORD_RE.finditer(original))
        corrected_words = list(WORD_RE.finditer(corrected))
        if len(original_words) != len(corrected_words):
            return corrected, []

        replacements: list[tuple[int, int, str]] = []
        decisions: list[dict] = []
        for source_match, corrected_match in zip(
            original_words, corrected_words
        ):
            source = source_match.group(0)
            existing = corrected_match.group(0)
            if (
                len(source) < 4
                or len(source) > 20
                or self.dictionary_index.contains_surface_form(existing)
                or self.suffix_index.contains(source)
                or self.suffix_index.contains(existing)
            ):
                continue

            source_key = normalize_key(source)
            looks_suffixed = source_key.endswith(
                (
                    "ni",
                    "ek",
                    "ha",
                    "na",
                    "kom",
                    "hom",
                    "om",
                    "li",
                    "lek",
                    "lu",
                    "lha",
                    "lna",
                    "lkom",
                    "lhom",
                    "lom",
                    "x",
                )
            )
            local_options = []
            for local_position, global_position in enumerate(
                range(source_match.start(), source_match.end())
            ):
                options = []
                for action, probability in position_candidates[global_position]:
                    if action == COPY_ACTION or probability < 0.05:
                        continue
                    rendered = apply_actions(
                        source[local_position],
                        [action],
                    )
                    if rendered.casefold() == source[local_position].casefold():
                        continue
                    options.append((action, probability))
                if looks_suffixed:
                    character = source[local_position]
                    structural_actions = []
                    if character.casefold() == "e":
                        structural_actions.append("i")
                    special = {
                        "c": "ċ",
                        "g": "ġ",
                        "h": "ħ",
                        "z": "ż",
                    }.get(character.casefold())
                    if special:
                        structural_actions.append(
                            special.upper()
                            if character.isupper()
                            else special
                        )
                    if (
                        character.isalpha()
                        and character.casefold() not in "aeiou"
                        and (
                            local_position + 1 >= len(source)
                            or source[local_position + 1].casefold()
                            != character.casefold()
                        )
                    ):
                        structural_actions.append(
                            COPY_ACTION + character
                        )
                    if (
                        character.casefold() == "o"
                        and local_position + 1 < len(source)
                        and source[local_position + 1].casefold() == "m"
                    ):
                        structural_actions.append("h" + COPY_ACTION)
                    for action in structural_actions:
                        if all(existing_action != action for existing_action, _ in options):
                            options.append((action, 0.08))
                if options:
                    local_options.append((local_position, options))
            local_options.sort(
                key=lambda item: max(
                    probability for _, probability in item[1]
                ),
                reverse=True,
            )
            local_options = local_options[:12]

            valid_candidates: dict[str, tuple[float, int]] = {}
            for change_count in range(
                1, min(4, len(local_options)) + 1
            ):
                for selected_positions in combinations(
                    local_options, change_count
                ):
                    for selected_actions in product(
                        *(options for _, options in selected_positions)
                    ):
                        actions = [COPY_ACTION] * len(source)
                        probabilities = []
                        for (
                            position,
                            _,
                        ), (action, probability) in zip(
                            selected_positions, selected_actions
                        ):
                            actions[position] = action
                            probabilities.append(probability)
                        candidate = apply_actions(source, actions)
                        if (
                            normalize_key(candidate)
                            == normalize_key(source)
                            or not WORD_RE.fullmatch(candidate)
                            or not self._suffix_form_exists(
                                candidate, source
                            )
                            or not self._suffix_repair_has_evidence(
                                source, candidate
                            )
                        ):
                            continue
                        score = sum(probabilities) / len(probabilities)
                        previous = valid_candidates.get(candidate)
                        if previous is None or (score, -change_count) > (
                            previous[0],
                            -previous[1],
                        ):
                            valid_candidates[candidate] = (
                                score,
                                change_count,
                            )

            if not valid_candidates:
                continue
            candidate, (score, change_count) = max(
                valid_candidates.items(),
                key=lambda item: (
                    -item[1][1],
                    item[1][0],
                    item[0],
                ),
            )
            replacements.append(
                (
                    corrected_match.start(),
                    corrected_match.end(),
                    candidate,
                )
            )
            decisions.append(
                {
                    "source": source,
                    "previous_candidate": existing,
                    "candidate": candidate,
                    "confidence": round(score, 4),
                    "changed_positions": change_count,
                    "decision": "accept_suffix_candidate_rescue",
                }
            )

        rescued = corrected
        for start, end, replacement in reversed(replacements):
            rescued = rescued[:start] + replacement + rescued[end:]
        return rescued, decisions

    def _suffix_specialist_rescue(
        self,
        original: str,
        corrected: str,
        existing_decisions: list[dict],
    ) -> tuple[str, list[dict]]:
        if self.suffix_specialist is None or self.suffix_index is None:
            return corrected, []
        specialist_result = self.suffix_specialist.correct(original)
        original_words = list(WORD_RE.finditer(original))
        corrected_words = list(WORD_RE.finditer(corrected))
        specialist_words = list(
            WORD_RE.finditer(specialist_result["corrected_text"])
        )
        if not (
            len(original_words)
            == len(corrected_words)
            == len(specialist_words)
        ):
            return corrected, []

        replacements: list[tuple[int, int, str]] = []
        decisions: list[dict] = []
        specialist_edits = specialist_result.get("edits", [])
        for source_match, primary_match, specialist_match in zip(
            original_words, corrected_words, specialist_words
        ):
            source = source_match.group(0)
            primary = primary_match.group(0)
            candidate = specialist_match.group(0)
            if (
                normalize_key(candidate) == normalize_key(source)
                or normalize_key(candidate) == normalize_key(primary)
                or not self._suffix_form_exists(candidate, source)
                or not self._suffix_repair_has_evidence(
                    source, candidate
                )
                or self.dictionary_index.contains_surface_form(primary)
            ):
                continue

            specialist_confidences = [
                float(edit.get("confidence", 0.0))
                for edit in specialist_edits
                if int(edit.get("start", -1)) < source_match.end()
                and int(edit.get("end", -1)) > source_match.start()
            ]
            if not specialist_confidences:
                continue
            specialist_confidence = max(specialist_confidences)
            primary_confidence = 0.0
            for decision in reversed(existing_decisions):
                if (
                    decision.get("source") == source
                    and normalize_key(
                        str(decision.get("candidate", ""))
                    )
                    == normalize_key(primary)
                ):
                    primary_confidence = float(
                        decision.get("confidence", 0.0)
                    )
                    break
            if (
                specialist_confidence
                < primary_confidence + self.suffix_specialist_min_margin
            ):
                continue
            replacements.append(
                (
                    primary_match.start(),
                    primary_match.end(),
                    candidate,
                )
            )
            decisions.append(
                {
                    "source": source,
                    "previous_candidate": primary,
                    "candidate": candidate,
                    "primary_confidence": round(primary_confidence, 4),
                    "specialist_confidence": round(
                        specialist_confidence, 4
                    ),
                    "decision": "accept_suffix_specialist_candidate",
                }
            )

        rescued = corrected
        for start, end, replacement in reversed(replacements):
            rescued = rescued[:start] + replacement + rescued[end:]
        return rescued, decisions

    def _neural_action_support(
        self,
        source: str,
        candidate: str,
        source_start: int,
        position_candidates: list[list[tuple[str, float]]],
    ) -> float:
        try:
            actions = derive_actions(source, candidate)
        except (AssertionError, ValueError):
            return 0.0
        probabilities = []
        for offset, action in enumerate(actions):
            rows = position_candidates[source_start + offset]
            probability = max(
                (
                    row_probability
                    for row_action, row_probability in rows
                    if row_action == action
                    or row_action.casefold() == action.casefold()
                ),
                default=0.005,
            )
            probabilities.append(probability)
        return sum(probabilities) / max(1, len(probabilities))

    def _fuzzy_dictionary_rescue(
        self,
        original: str,
        corrected: str,
        position_candidates: list[list[tuple[str, float]]],
    ) -> tuple[str, list[dict]]:
        if self.dictionary_index is None:
            return corrected, []
        original_words = list(WORD_RE.finditer(original))
        corrected_words = list(WORD_RE.finditer(corrected))
        if len(original_words) != len(corrected_words):
            return corrected, []

        replacements: list[tuple[int, int, str]] = []
        decisions: list[dict] = []
        for source_match, corrected_match in zip(
            original_words, corrected_words
        ):
            source = source_match.group(0)
            existing = corrected_match.group(0)
            if (
                self.dictionary_index.contains_surface_form(source)
                or self.dictionary_index.contains_surface_form(existing)
                or self._suffix_form_exists(existing, source)
            ):
                continue
            max_distance = 2 if len(source) <= 8 else 1
            scored = []
            for entry, distance in self.dictionary_index.nearby(
                source, max_distance=max_distance, limit=48
            ):
                candidate = entry.canonical
                if not WORD_RE.fullmatch(candidate):
                    continue
                support = self._neural_action_support(
                    source,
                    candidate,
                    source_match.start(),
                    position_candidates,
                )
                if distance > 0 and support < (
                    0.16 if distance == 1 else 0.20
                ):
                    continue
                score = support + (0.35 if distance == 0 else 0.0)
                score -= distance * 0.06
                scored.append((score, -distance, candidate, support))
            if not scored:
                continue
            score, negative_distance, candidate, support = max(scored)
            replacements.append(
                (
                    corrected_match.start(),
                    corrected_match.end(),
                    candidate,
                )
            )
            decisions.append(
                {
                    "source": source,
                    "previous_candidate": existing,
                    "candidate": candidate,
                    "distance": -negative_distance,
                    "neural_support": round(support, 4),
                    "confidence": round(min(1.0, score), 4),
                    "decision": "accept_neural_fuzzy_dictionary_candidate",
                }
            )

        rescued = corrected
        for start, end, replacement in reversed(replacements):
            rescued = rescued[:start] + replacement + rescued[end:]
        return rescued, decisions

    @staticmethod
    def _finalize_surface(text: str, original: str) -> str:
        sun_articles = {
            "ċ": "iċ",
            "d": "id",
            "n": "in",
            "r": "ir",
            "s": "is",
            "t": "it",
            "x": "ix",
            "z": "iż",
            "ż": "iż",
        }

        def article_replacement(match: re.Match) -> str:
            article = match.group(1)
            word = match.group(2)
            first = word[:1].casefold()
            if first in "aeiou":
                resolved = "l"
            else:
                resolved = sun_articles.get(first, "il")
            if article[:1].isupper():
                resolved = resolved[:1].upper() + resolved[1:]
            return f"{resolved}-{word}"

        finalized = re.sub(
            r"(?<!\w)(il|l)[\s-]+([^\W\d_]+(?:['’][^\W\d_]+)*)",
            article_replacement,
            text,
            flags=re.IGNORECASE,
        )
        finalized = re.sub(r"\s+([,.?!])", r"\1", finalized)
        if (
            len(list(WORD_RE.finditer(original))) > 1
            and not re.search(r"[.?!]\s*$", finalized)
        ):
            finalized = finalized.rstrip() + "."
        return finalized

    def _predict_chunk(
        self, text: str
    ) -> tuple[list[str], list[float], list[list[tuple[str, float]]]]:
        unknown_id = self.vocab.characters[UNK_CHAR]
        ids = [
            self.vocab.characters.get(character, unknown_id) for character in text
        ]
        inputs = torch.tensor([ids], dtype=torch.long, device=self.device)
        lengths = torch.tensor([len(ids)], dtype=torch.long, device=self.device)
        with torch.inference_mode():
            probabilities = self.model(inputs, lengths).softmax(dim=-1)[0]
        top_probabilities, top_indexes = probabilities.topk(
            k=min(3, probabilities.shape[-1]), dim=-1
        )
        actions: list[str] = []
        confidences: list[float] = []
        candidates: list[list[tuple[str, float]]] = []
        for position, character in enumerate(text):
            position_candidates = [
                (
                    self.inverse_actions[int(index.item())],
                    float(probability.item()),
                )
                for probability, index in zip(
                    top_probabilities[position], top_indexes[position]
                )
            ]
            action, confidence = position_candidates[0]
            if ids[position] == unknown_id:
                action, confidence = COPY_ACTION, 1.0
            elif action != COPY_ACTION and confidence < self.threshold:
                action = COPY_ACTION
            actions.append(action)
            confidences.append(confidence)
            candidates.append(position_candidates)
        return actions, confidences, candidates

    @staticmethod
    def _edit_distance(source: str, target: str) -> int:
        return sum(
            max(i2 - i1, j2 - j1)
            for tag, i1, i2, j1, j2 in SequenceMatcher(
                None, source, target, autojunk=False
            ).get_opcodes()
            if tag != "equal"
        )

    def _bare_word_alternatives(
        self,
        text: str,
        selected_actions: list[str],
        position_candidates: list[list[tuple[str, float]]],
        selected_text: str,
    ) -> list[str]:
        if (
            len(text) > 32
            or not re.fullmatch(r"[^\W\d_]+(?:['’][^\W\d_]+)?", text)
        ):
            return []
        scored: dict[str, float] = {selected_text: 1.0}
        positions = range(len(text))
        for change_count in (1, 2):
            for changed_positions in combinations(positions, change_count):
                options = []
                for position in changed_positions:
                    candidates = [
                        (action, probability)
                        for action, probability in position_candidates[position]
                        if action != selected_actions[position]
                    ]
                    if not candidates:
                        break
                    options.append(candidates)
                if len(options) != change_count:
                    continue
                for replacements in product(*options):
                    actions = list(selected_actions)
                    probabilities = []
                    for position, (action, probability) in zip(
                        changed_positions, replacements
                    ):
                        actions[position] = action
                        probabilities.append(probability)
                    candidate = apply_actions(text, actions)
                    if candidate == selected_text:
                        continue
                    if not re.fullmatch(
                        r"[^\W\d_]+(?:['’][^\W\d_]+)?", candidate
                    ):
                        continue
                    if text[:1].islower() and not candidate[:1].islower():
                        continue
                    if text[:1].isupper() and not candidate[:1].isupper():
                        continue
                    if self._edit_distance(text, candidate) > 2:
                        continue
                    score = sum(probabilities) / max(1, len(probabilities))
                    scored[candidate] = max(score, scored.get(candidate, 0.0))
        return [
            candidate
            for candidate, _ in sorted(
                scored.items(), key=lambda item: item[1], reverse=True
            )[:2]
        ]

    def correct(self, text: str) -> dict:
        started = time.perf_counter()
        if not text:
            return {
                "corrected_text": "",
                "edits": [],
                "confidence": 1.0,
                "processing_time": 0.0,
                "model_version": self.model_version,
            }
        actions: list[str] = []
        confidences: list[float] = []
        candidates: list[list[tuple[str, float]]] = []
        for start in range(0, len(text), self.max_length):
            chunk = text[start : start + self.max_length]
            chunk_actions, chunk_confidences, chunk_candidates = self._predict_chunk(
                chunk
            )
            actions.extend(chunk_actions)
            confidences.extend(chunk_confidences)
            candidates.extend(chunk_candidates)
        raw_neural_text = apply_actions(text, actions)
        dictionary_decisions: list[dict] = []
        corrected = raw_neural_text
        if self.dictionary_index is not None:
            corrected, dictionary_decisions = self.dictionary_index.guard_text(
                text,
                raw_neural_text,
                (
                    self._validate_generated_suffix
                    if self.suffix_index is not None
                    else None
                ),
            )
            corrected, rescue_decisions = self._dictionary_rescue(
                text, corrected, candidates
            )
            dictionary_decisions.extend(rescue_decisions)
            corrected, phonological_decisions = self._phonological_prefix_rescue(
                text, corrected, candidates
            )
            dictionary_decisions.extend(phonological_decisions)
            corrected, suffix_rescue_decisions = (
                self._suffix_candidate_rescue(
                    text, corrected, candidates
                )
            )
            dictionary_decisions.extend(suffix_rescue_decisions)
            corrected, context_decisions = self._dictionary_context_rescue(
                text, corrected
            )
            dictionary_decisions.extend(context_decisions)
            corrected, specialist_decisions = (
                self._suffix_specialist_rescue(
                    text, corrected, dictionary_decisions
                )
            )
            dictionary_decisions.extend(specialist_decisions)
            corrected, fuzzy_decisions = self._fuzzy_dictionary_rescue(
                text, corrected, candidates
            )
            dictionary_decisions.extend(fuzzy_decisions)
        corrected = self._finalize_surface(corrected, text)
        sequence_alternatives = self._bare_word_alternatives(
            text, actions, candidates, raw_neural_text
        )
        if self.dictionary_index is not None:
            validated_alternatives = [corrected]
            for candidate in sequence_alternatives:
                guarded_candidate, _ = self.dictionary_index.guard_text(
                    text,
                    candidate,
                    (
                        self._validate_generated_suffix
                        if self.suffix_index is not None
                        else None
                    ),
                )
                if guarded_candidate == candidate and candidate not in validated_alternatives:
                    validated_alternatives.append(candidate)
            sequence_alternatives = validated_alternatives[:2]
        if corrected == text and sequence_alternatives == [text]:
            sequence_alternatives = []

        def alternatives(
            start: int, end: int, replacement: str, original: str
        ) -> list[str]:
            values = [replacement, original]
            if end > start:
                for position in range(start, min(end, start + 6)):
                    for candidate_action, _ in candidates[position][1:]:
                        local_actions = list(actions[start:end])
                        local_actions[position - start] = candidate_action
                        candidate = apply_actions(text[start:end], local_actions)
                        if candidate not in values:
                            values.append(candidate)
                        if len(values) >= 4:
                            break
                    if len(values) >= 4:
                        break
            return [value for value in values if value != ""][:4]

        edits = structured_edits(
            text, corrected, confidences, alternatives
        )
        changed_confidences = [
            edit["confidence"] for edit in edits if edit["replacement"] != edit["original"]
        ]
        overall_confidence = (
            sum(changed_confidences) / len(changed_confidences)
            if changed_confidences
            else 1.0
        )
        return {
            "original_text": text,
            "corrected_text": corrected,
            "changed": corrected != text,
            "edits": edits,
            "confidence": round(overall_confidence, 4),
            "processing_time": round(time.perf_counter() - started, 6),
            "model_version": self.model_version,
            "action_threshold": self.threshold,
            "sequence_alternatives": sequence_alternatives,
            "dictionary_validation": {
                "enabled": self.dictionary_validation_enabled,
                "suffix_validation": self.suffix_validation_enabled,
                "raw_neural_text": raw_neural_text,
                "decisions": dictionary_decisions,
            },
        }
