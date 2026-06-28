# -*- coding: utf-8 -*-
import gc
import json
import sys
import time
import tracemalloc
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Essentials.helpers.performance_logging import rss_mb


CASES = [
    "l bajja alqet ax sabu",
    "alqet",
    "ax",
    "l bajja",
    "Jien naħseb li dan huwa tajjeb.",
    "għamililkom",
    "dari",
    "ghaliex",
    "laham laham laham",
    "Amy qalet 'good morning'",
    "qed isir dan",
]


def cache_info(spellchecker) -> dict:
    names = [
        "_normalize_word_cached",
        "_graphemes_cached",
        "_is_noun_tagged_word",
        "_is_dual_noun",
        "_is_pronoun_tagged_word",
        "_is_adverb_tagged_word",
        "_is_preposition_tagged_word",
        "_extract_consonant_anchor",
        "_vowel_slots",
        "_count_vowels",
        "_damerau_levenshtein_distance",
        "_word_distance",
        "_get_candidates_cached",
        "meaning_for",
    ]
    out = {}
    for name in names:
        fn = getattr(spellchecker, name, None)
        if fn is not None and hasattr(fn, "cache_info"):
            info = fn.cache_info()
            out[name] = {
                "hits": info.hits,
                "misses": info.misses,
                "maxsize": info.maxsize,
                "currsize": info.currsize,
            }
    return out


def post(client, text: str) -> dict:
    start = time.perf_counter()
    response = client.post("/check-text", json={"text": text})
    elapsed = time.perf_counter() - start
    assert response.status_code == 200, response.get_data(as_text=True)
    return {
        "text": text,
        "elapsed_s": elapsed,
        "rss_mb": rss_mb(),
        "corrected_text": response.get_json()["corrected_text"],
    }


def main() -> None:
    tracemalloc.start()
    before_import_rss = rss_mb()
    start = time.perf_counter()
    import app

    startup_s = time.perf_counter() - start
    current, peak = tracemalloc.get_traced_memory()
    client = app.app.test_client()

    report = {
        "startup": {
            "duration_s": startup_s,
            "rss_mb": rss_mb(),
            "rss_before_import_mb": before_import_rss,
            "tracemalloc_current_mb": current / 1024 / 1024,
            "tracemalloc_peak_mb": peak / 1024 / 1024,
            "dictionary_words": len(app.spellchecker.dictionary),
            "paradigms": len(app.spellchecker.paradigm_forms),
            "suffix_records": app.spellchecker.suffix_generator.verb_index.record_count(),
            "raw_entries_len": len(getattr(app.spellchecker, "raw_entries", ())),
            "cache_info": cache_info(app.spellchecker),
            "spellchecker_id": id(app.spellchecker),
        },
        "requests": [],
        "repeated": [],
    }

    for text in CASES:
        report["requests"].append(post(client, text))

    repeated_text = "l bajja alqet ax sabu"
    expected = None
    for index in range(20):
        result = post(client, repeated_text)
        if expected is None:
            expected = result["corrected_text"]
        assert result["corrected_text"] == expected
        result["index"] = index + 1
        result["spellchecker_id"] = id(app.spellchecker)
        report["repeated"].append(result)

    gc.collect()
    current, peak = tracemalloc.get_traced_memory()
    report["after_gc"] = {
        "rss_mb": rss_mb(),
        "tracemalloc_current_mb": current / 1024 / 1024,
        "tracemalloc_peak_mb": peak / 1024 / 1024,
        "cache_info": cache_info(app.spellchecker),
        "spellchecker_id": id(app.spellchecker),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
