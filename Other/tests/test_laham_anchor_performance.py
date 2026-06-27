# -*- coding: utf-8 -*-
import time
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import app


spellchecker = app.spellchecker


CORRECTION_CASES = {
    "laham": "laħam",
    "laghba": "lagħba",
    "gamel": "għamel",
    "ghamel": "għamel",
    "ghatba": "għatba",
    "baghad": "bagħad",
    "xaghar": "xagħar",
    "hames": "ħames",
    "habib": "ħabib",
    "hajja": "ħajja",
    "ghajn": "għajn",
    "gharaf": "għaraf",
    "qamh": "qamħ",
    "mahrug": "maħruġ",
}

# These are regression probes for the old anchor-candidate timeout path.
# Their preferred correction may still need linguistic tuning, so this test
# only asserts that the lookup returns promptly.
PERFORMANCE_ONLY_CASES = (
    "lahma",
    "lagham",
    "sehem",
)

MAX_SINGLE_WORD_SECONDS = 1.0
MAX_BATCH_SECONDS = 5.0


def timed_correct_word(word: str) -> tuple[str, float]:
    start = time.perf_counter()
    corrected = spellchecker.correct_word(word)
    return corrected, time.perf_counter() - start


batch_start = time.perf_counter()

for typo, expected in CORRECTION_CASES.items():
    actual, elapsed = timed_correct_word(typo)
    assert actual == expected, f"{typo!r}: expected {expected!r}, got {actual!r}"
    assert elapsed < MAX_SINGLE_WORD_SECONDS, (
        f"{typo!r}: correction took {elapsed:.3f}s, expected under "
        f"{MAX_SINGLE_WORD_SECONDS:.3f}s"
    )

for typo in PERFORMANCE_ONLY_CASES:
    actual, elapsed = timed_correct_word(typo)
    assert actual, f"{typo!r}: expected a non-empty correction"
    assert elapsed < MAX_SINGLE_WORD_SECONDS, (
        f"{typo!r}: correction took {elapsed:.3f}s, expected under "
        f"{MAX_SINGLE_WORD_SECONDS:.3f}s"
    )

lagham_suggestions = spellchecker.suggest("lagham", limit=10)
assert "laħam" not in lagham_suggestions, (
    "'lagham' contains medial gh, so it should not suggest 'laħam'"
)

batch_elapsed = time.perf_counter() - batch_start
assert batch_elapsed < MAX_BATCH_SECONDS, (
    f"laham-style anchor regression batch took {batch_elapsed:.3f}s, expected "
    f"under {MAX_BATCH_SECONDS:.3f}s"
)

print("laham anchor performance regression checks passed")
