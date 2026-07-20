# -*- coding: utf-8 -*-
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Essentials.helpers.performance_logging import (
    RequestProfiler,
    reset_current_profiler,
    set_current_profiler,
)
from Essentials import app


def test_long_text_candidate_retrieval_is_bounded():
    spellchecker = app.spellchecker
    fixture = ROOT / "Other" / "tests" / "fixtures" / "long_text_timeout_input.txt"
    text = fixture.read_text(encoding="utf-8")

    spellchecker._get_candidates_cached.cache_clear()
    profiler = RequestProfiler(profile_enabled=True)
    token = set_current_profiler(profiler)
    try:
        result = spellchecker.correct_text_rich(text, edit_distance_tolerance=1)
    finally:
        reset_current_profiler(token)

    assert result["corrected_text"]
    assert profiler.counters.get("dictionary_entries_inspected", 0) < len(
        spellchecker.dictionary
    )
    assert profiler.counters.get("anchor_entries_inspected", 0) < len(
        spellchecker.anchor_map
    )


def test_unresolved_noise_token_does_not_scan_complete_indexes():
    spellchecker = app.spellchecker
    spellchecker._get_candidates_cached.cache_clear()
    profiler = RequestProfiler(profile_enabled=True)
    token = set_current_profiler(profiler)
    try:
        spellchecker.correct_text_rich(
            "zzqxvbnmzz mhux kelma imma t-test irid ikompli",
            edit_distance_tolerance=1,
        )
    finally:
        reset_current_profiler(token)

    assert profiler.counters.get("dictionary_entries_inspected", 0) <= 256
    assert profiler.counters.get("anchor_entries_inspected", 0) <= 512
