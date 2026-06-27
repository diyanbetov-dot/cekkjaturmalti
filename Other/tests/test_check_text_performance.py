# -*- coding: utf-8 -*-
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import app


SCENARIOS = {
    "required_phrase": "alqu l bajja ax sabu 'shark'",
    "difficult_word": "introducieni",
    "four_easy_misspellings": "laham mahrug ghaliex nahseb",
    "four_rule_families": "minix zghazugha attegjament isir",
    "repeated_errors": "laham laham laham ghaliex ghaliex ghaliex",
    "correct_paragraph": "Jien naħseb li dan huwa tajjeb u għandu jibqa' hekk.",
    "mixed_social": (
        "Fan ta' Amy minix anzi, bhal hafna minnkom nixtieq li Luke jirbah "
        "ghax zghazugha kellha attegjament."
    ),
}

PREFIXES = [
    "alqu",
    "alqu l",
    "alqu l bajja",
    "alqu l bajja ax",
    "alqu l bajja ax sabu",
    "alqu l bajja ax sabu 'shark'",
]


def post_text(text: str) -> tuple[dict, float]:
    client = app.app.test_client()
    start = time.perf_counter()
    response = client.post(
        "/check-text",
        json={"text": text, "edit_distance_tolerance": 1},
    )
    elapsed = time.perf_counter() - start
    assert response.status_code == 200, response.get_data(as_text=True)
    return response.get_json(), elapsed


def run_samples(label: str, text: str, repeats: int = 3) -> dict:
    timings = []
    last_response = None
    for _ in range(repeats):
        last_response, elapsed = post_text(text)
        timings.append(elapsed)
    word_count = len([token for token in text.split() if token])
    return {
        "label": label,
        "chars": len(text),
        "words_estimate": word_count,
        "first_request_s": timings[0],
        "warm_median_s": statistics.median(timings[1:]) if repeats > 1 else timings[0],
        "per_token_warm_ms": (
            statistics.median(timings[1:]) * 1000 / max(word_count, 1)
            if repeats > 1
            else timings[0] * 1000 / max(word_count, 1)
        ),
        "changed": last_response["changed"],
        "corrected_text": last_response["corrected_text"],
    }


def main() -> None:
    reports = []
    for label, text in SCENARIOS.items():
        reports.append(run_samples(label, text))
    for prefix in PREFIXES:
        reports.append(run_samples(f"prefix:{prefix}", prefix))

    with ThreadPoolExecutor(max_workers=2) as executor:
        concurrent = list(
            executor.map(
                post_text,
                [SCENARIOS["required_phrase"], SCENARIOS["mixed_social"]],
            )
        )

    print(json.dumps(reports, ensure_ascii=False, indent=2))
    print(
        "concurrent_request_times_s="
        + json.dumps([elapsed for _, elapsed in concurrent])
    )


if __name__ == "__main__":
    main()
