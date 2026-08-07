# -*- coding: utf-8 -*-
"""Training-only context augmentation for the neural corrector.

For each (noisy, clean) pair, generates N additional views by placing the
correction at different positions within short filler contexts.  All derived
views share the parent's ``source_group`` so that ``build_groups()`` always
keeps them in the same split partition — preventing any augmented view from
leaking into validation or test.

Usage (from train.py):
    from neural_corrector.dataset.context_augmentation import augment_for_training
    extra = augment_for_training(train_rows, views_per_pair=2)
    train_rows = train_rows + extra   # augmented rows go to training only
"""

from __future__ import annotations

import random
from typing import Sequence

# A small pool of neutral Maltese filler sentences that add context without
# introducing corrections of their own.  These are already-correct sentences
# so no target-side changes are needed when they appear around the correction.
_FILLER_SENTENCES: list[str] = [
    "Il-ħajja hija sabiħa.",
    "Il-kelb qiegħed fid-dar.",
    "Ilbieraħ mort nixtri.",
    "It-tfal kienu jilgħabu barra.",
    "Il-baħar huwa kbir ħafna.",
    "Rajt film ġdid.",
    "Il-karozza tiegħi hija ħamra.",
    "Marru ħarġa fil-ġnien.",
    "Ix-xemx kienet sabiħa dak il-jum.",
    "L-għalliem spjega sew.",
    "Morna l-iskola kmieni.",
    "Ħadna l-ikel mal-familja.",
    "Il-qattus raqad fuq is-sufan.",
    "Jien ngħix Malta.",
    "L-uffiċċju huwa qrib id-dar.",
]


def _make_view(
    noisy: str,
    clean: str,
    parent: dict,
    view_index: int,
    filler_before: str | None,
    filler_after: str | None,
) -> dict:
    """Build a single augmented training example."""
    sep = " "
    aug_noisy = sep.join(filter(None, [filler_before, noisy, filler_after]))
    aug_clean = sep.join(filter(None, [filler_before, clean, filler_after]))
    return {
        **parent,
        "id": f"{parent['id']}:aug{view_index}",
        "noisy": aug_noisy,
        "clean": aug_clean,
        "source": parent.get("source", "augmented"),
        "source_group": parent.get("source_group", "manual"),  # same group → same split
        "augmented": True,
        "augmented_from": parent["id"],
        "augmented_view_index": view_index,
        "is_unchanged": aug_noisy == aug_clean,
    }


def augment_for_training(
    rows: list[dict],
    views_per_pair: int = 2,
    seed: int = 42,
    *,
    skip_unchanged: bool = True,
    skip_multiline: bool = True,
) -> list[dict]:
    """Return additional training-only views for each row.

    Parameters
    ----------
    rows:            Original training examples (already split into train set).
    views_per_pair:  How many extra views to generate per pair (1–4 recommended).
    seed:            RNG seed for reproducibility.
    skip_unchanged:  Don't augment identity pairs (input == output).
    skip_multiline:  Don't augment multi-line examples (complex enough already).

    Returns
    -------
    List of augmented rows (never includes the originals — caller must combine).
    """
    rng = random.Random(seed)
    fillers = _FILLER_SENTENCES
    extra: list[dict] = []

    for row in rows:
        if skip_unchanged and row.get("is_unchanged"):
            continue
        if skip_multiline and ("\n" in row["noisy"] or "\n" in row["clean"]):
            continue
        if not row.get("noisy") or not row.get("clean"):
            continue

        for view_index in range(views_per_pair):
            position = view_index % 3  # cycle: before-only, after-only, both
            if position == 0:
                # Correction at end: filler before, nothing after
                before = rng.choice(fillers)
                after = None
            elif position == 1:
                # Correction at start: nothing before, filler after
                before = None
                after = rng.choice(fillers)
            else:
                # Correction in middle: filler on both sides
                before = rng.choice(fillers)
                after = rng.choice([f for f in fillers if f != before])

            extra.append(
                _make_view(row["noisy"], row["clean"], row, view_index, before, after)
            )

    return extra
