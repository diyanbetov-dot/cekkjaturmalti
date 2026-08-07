from __future__ import annotations

from difflib import SequenceMatcher

COPY_ACTION = "<COPY>"
PAD_ACTION = "<PAD>"


def render_action(action: str, source_character: str) -> str:
    return action.replace(COPY_ACTION, source_character)


def _parts(value: str, count: int) -> list[str]:
    if count <= 0:
        return []
    return [
        value[round(index * len(value) / count) : round((index + 1) * len(value) / count)]
        for index in range(count)
    ]


def derive_actions(source: str, target: str) -> list[str]:
    """Return one output action per source character.

    Literal text replaces the source character. ``<COPY>`` retains it. Insertions
    are attached to an adjacent source action, making decoding a single pass.
    """

    if not source:
        if target:
            raise ValueError("A non-empty target cannot be aligned to an empty source")
        return []
    base: list[str | None] = [None] * len(source)
    prefix = [""] * len(source)
    suffix = [""] * len(source)
    matcher = SequenceMatcher(None, source, target, autojunk=False)

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for index in range(i1, i2):
                base[index] = COPY_ACTION
        elif tag == "delete":
            for index in range(i1, i2):
                base[index] = ""
        elif tag == "replace":
            target_parts = _parts(target[j1:j2], i2 - i1)
            for offset, index in enumerate(range(i1, i2)):
                part = target_parts[offset]
                base[index] = COPY_ACTION if part == source[index] else part
        elif tag == "insert":
            inserted = target[j1:j2]
            if i1 < len(source):
                prefix[i1] += inserted
            else:
                suffix[-1] += inserted

    actions: list[str] = []
    for index, action in enumerate(base):
        resolved = COPY_ACTION if action is None else action
        actions.append(prefix[index] + resolved + suffix[index])
    if apply_actions(source, actions) != target:
        raise AssertionError("Character action alignment did not reconstruct the target")
    return actions


def apply_actions(source: str, actions: list[str]) -> str:
    if len(source) != len(actions):
        raise ValueError("Source and action lengths differ")
    return "".join(
        render_action(action, character)
        for character, action in zip(source, actions)
    )


def chunk_aligned(
    source: str, actions: list[str], max_length: int
) -> list[tuple[str, list[str]]]:
    if len(source) != len(actions):
        raise ValueError("Source and action lengths differ")
    return [
        (source[start : start + max_length], actions[start : start + max_length])
        for start in range(0, len(source), max_length)
    ]

