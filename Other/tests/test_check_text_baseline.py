# -*- coding: utf-8 -*-
import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import app


FIXTURE = Path(__file__).with_name("fixtures") / "check_text_baseline.json"


def canonicalize_meanings(value):
    if isinstance(value, dict):
        return {key: canonicalize_meanings(item) for key, item in value.items()}
    if isinstance(value, list):
        return [canonicalize_meanings(item) for item in value]
    if isinstance(value, str) and " / " in value:
        parts = value.split(" / ")
        if all(part.startswith("F") and part[1:].isdigit() for part in parts):
            return " / ".join(sorted(parts))
    return value


def main() -> None:
    baseline = json.loads(FIXTURE.read_text(encoding="utf-8"))
    client = app.app.test_client()

    for case in baseline:
        text = case["json"]["original_text"]
        response = client.post(
            "/check-text",
            json={"text": text, "edit_distance_tolerance": 1},
        )
        actual = {
            "id": case["id"],
            "status_code": response.status_code,
            "json": response.get_json(),
        }
        assert canonicalize_meanings(actual) == canonicalize_meanings(case), (
            f"{case['id']} JSON changed.\n"
            f"Expected: {json.dumps(case, ensure_ascii=False, sort_keys=True)}\n"
            f"Actual:   {json.dumps(actual, ensure_ascii=False, sort_keys=True)}"
        )

    print(f"check-text baseline regression passed ({len(baseline)} cases)")


if __name__ == "__main__":
    main()
