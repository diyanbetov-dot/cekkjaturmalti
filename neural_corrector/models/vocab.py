from __future__ import annotations

import json
import string
from dataclasses import dataclass
from pathlib import Path

from .alignment import COPY_ACTION, PAD_ACTION

PAD_CHAR = "<PAD>"
UNK_CHAR = "<UNK>"
FIXED_MALTESE = (
    "aàbcċdeèfġgħhħiìjklmnoòpqrstuvwxyzż"
    "AÀBCĊDEÈFĠGĦHĦIÌJKLMNOÒPQRSTUVWXYZŻ"
)
FIXED_TEXT = string.printable + FIXED_MALTESE + "’“”€…"


@dataclass(frozen=True)
class Vocabularies:
    characters: dict[str, int]
    actions: dict[str, int]

    @property
    def inverse_characters(self) -> list[str]:
        return _inverse(self.characters)

    @property
    def inverse_actions(self) -> list[str]:
        return _inverse(self.actions)

    def save(self, path: Path) -> None:
        path.write_text(
            json.dumps(
                {"characters": self.characters, "actions": self.actions},
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "Vocabularies":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(payload["characters"], payload["actions"])


def _inverse(mapping: dict[str, int]) -> list[str]:
    result = [""] * len(mapping)
    for value, index in mapping.items():
        result[index] = value
    return result


def build_vocabularies(
    sources: list[str], action_sequences: list[list[str]]
) -> Vocabularies:
    characters = {PAD_CHAR: 0, UNK_CHAR: 1}
    for character in sorted(set(FIXED_TEXT).union(*(set(text) for text in sources))):
        if character not in characters:
            characters[character] = len(characters)
    actions = {PAD_ACTION: 0, COPY_ACTION: 1}
    for action in sorted({action for sequence in action_sequences for action in sequence}):
        if action not in actions:
            actions[action] = len(actions)
    return Vocabularies(characters, actions)

