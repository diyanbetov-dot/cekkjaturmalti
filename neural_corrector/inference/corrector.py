from __future__ import annotations

import json
import re
import time
from difflib import SequenceMatcher
from itertools import combinations, product
from pathlib import Path

import torch

from neural_corrector.inference.edits import structured_edits
from neural_corrector.models.alignment import COPY_ACTION, apply_actions, render_action
from neural_corrector.models.char_edit_tagger import CharEditTagger
from neural_corrector.models.vocab import UNK_CHAR, Vocabularies


class NeuralCorrector:
    def __init__(
        self,
        artifact_dir: Path,
        threshold: float | None = None,
        device: str | None = None,
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
        corrected = apply_actions(text, actions)
        sequence_alternatives = self._bare_word_alternatives(
            text, actions, candidates, corrected
        )

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
        }
