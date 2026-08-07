from __future__ import annotations

import random
from typing import NamedTuple


class CorruptionResult(NamedTuple):
    corrupted_s1_text: str
    corruption_family: str | None
    is_corrupted: bool


CORRUPTION_FAMILIES = (
    "false_lexical_substitution",
    "overcorrection_valid_text",
    "name_corruption",
    "incorrect_suffix",
    "incorrect_article_preposition",
    "incorrect_join_split",
    "apostrophe_hyphen_error",
    "deletion_valid_material",
    "incorrect_preservation_error",
)


def corrupt_stage1_output(
    s1_text: str,
    *,
    corruption_rate: float = 0.30,
    seed: int = 42,
) -> CorruptionResult:
    """
    Deliberately corrupt S1 text on ~30% of applicable training examples.
    RAW text is NEVER touched or modified.
    """
    rng = random.Random(seed)
    if rng.random() > corruption_rate or not s1_text.strip():
        return CorruptionResult(
            corrupted_s1_text=s1_text,
            corruption_family=None,
            is_corrupted=False,
        )

    family = rng.choice(CORRUPTION_FAMILIES)
    words = s1_text.split()
    if not words:
        return CorruptionResult(s1_text, None, False)

    corrupted_words = list(words)
    idx = rng.randint(0, len(words) - 1)
    word = words[idx]

    if family == "false_lexical_substitution":
        if "vann" in s1_text:
            corrupted_s1 = s1_text.replace("vann", "sann")
        else:
            corrupted_words[idx] = word + "x"
            corrupted_s1 = " ".join(corrupted_words)

    elif family == "overcorrection_valid_text":
        if "iwassalhom" in s1_text:
            corrupted_s1 = s1_text.replace("iwassalhom", "wasalhom")
        else:
            corrupted_words[idx] = word.rstrip("a") if len(word) > 3 else word
            corrupted_s1 = " ".join(corrupted_words)

    elif family == "name_corruption":
        if "Ċensu" in s1_text:
            corrupted_s1 = s1_text.replace("Ċensu", "Censu")
        else:
            corrupted_words[0] = corrupted_words[0].lower()
            corrupted_s1 = " ".join(corrupted_words)

    elif family == "incorrect_suffix":
        if "tefgħu" in s1_text:
            corrupted_s1 = s1_text.replace("tefgħu", "tefgħuh")
        else:
            corrupted_words[idx] = word + "h"
            corrupted_s1 = " ".join(corrupted_words)

    elif family == "incorrect_article_preposition":
        if "ġol-vann" in s1_text:
            corrupted_s1 = s1_text.replace("ġol-vann", "ġol, vann")
        else:
            corrupted_words[idx] = "il-" + word
            corrupted_s1 = " ".join(corrupted_words)

    elif family == "incorrect_join_split":
        if "daqs likieku" in s1_text:
            corrupted_s1 = s1_text.replace("daqs likieku", "daqsli kieku")
        elif len(words) >= 2:
            mid = len(words) // 2
            corrupted_s1 = " ".join(words[:mid]) + words[mid] + " " + " ".join(words[mid+1:])
        else:
            corrupted_s1 = s1_text

    elif family == "apostrophe_hyphen_error":
        if "ma'riedx" not in s1_text and "ma riedx" in s1_text:
            corrupted_s1 = s1_text.replace("ma riedx", "ma'riedx")
        else:
            corrupted_words[idx] = word.replace("-", "")
            corrupted_s1 = " ".join(corrupted_words)

    elif family == "deletion_valid_material":
        if len(corrupted_words) > 2:
            del corrupted_words[idx]
            corrupted_s1 = " ".join(corrupted_words)
        else:
            corrupted_s1 = s1_text

    else:  # incorrect_preservation_error
        corrupted_s1 = s1_text

    return CorruptionResult(
        corrupted_s1_text=corrupted_s1,
        corruption_family=family,
        is_corrupted=True,
    )
