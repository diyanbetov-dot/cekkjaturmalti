from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import CORRECTOR, normalize  # noqa: E402


SOURCE = ROOT / "dics" / "manualdics" / "single.dic"
REPORT = ROOT / "single_dic_evaluation.txt"
FAILURES = ROOT / "single_dic_failed_entries.txt"
PREVIOUS_PRIMARY_PASSES = 651


def main() -> None:
    rows: list[tuple[int, str, str]] = []
    for line_number, raw in enumerate(SOURCE.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#") or "-" not in line:
            continue
        source, expected = (part.strip() for part in line.split("-", 1))
        if source and expected:
            rows.append((line_number, source, expected))

    details: list[dict[str, object]] = []
    engine_counts: Counter[str] = Counter()
    for line_number, source, expected in rows:
        corrected, candidates, recognized = CORRECTOR.correct_word(source)
        candidate_words = [candidate.word for candidate in candidates]
        expected_key = normalize(expected)
        primary_pass = normalize(corrected) == expected_key
        candidate_pass = primary_pass or any(normalize(word) == expected_key for word in candidate_words)
        primary_source = candidates[0].source if candidates else ("exact" if recognized else "none")
        if primary_pass:
            engine_counts[primary_source] += 1
        details.append(
            {
                "line": line_number,
                "source": source,
                "expected": expected,
                "corrected": corrected,
                "recognized": recognized,
                "candidates": candidate_words,
                "primary_source": primary_source,
                "primary_pass": primary_pass,
                "candidate_pass": candidate_pass,
            }
        )

    total = len(details)
    primary_passes = sum(bool(row["primary_pass"]) for row in details)
    candidate_passes = sum(bool(row["candidate_pass"]) for row in details)
    failures = [row for row in details if not row["primary_pass"]]
    missing = [row for row in details if not row["candidate_pass"]]

    report_lines = [
        "HOPE single.dic evaluation (current pipeline)",
        "============================================",
        f"Total entries: {total}",
        f"Primary correction passes: {primary_passes}",
        f"Primary correction failures: {total - primary_passes}",
        f"Primary pass rate: {primary_passes / total:.2%}" if total else "Primary pass rate: 0.00%",
        f"Expected target available as primary or suggestion: {candidate_passes}",
        f"Expected target absent from all candidates: {total - candidate_passes}",
        f"Candidate coverage: {candidate_passes / total:.2%}" if total else "Candidate coverage: 0.00%",
        "",
        "BEFORE VS CURRENT",
        f"Previous primary passes: {PREVIOUS_PRIMARY_PASSES}",
        f"Current primary passes: {primary_passes}",
        f"Net primary change: {primary_passes - PREVIOUS_PRIMARY_PASSES:+d}",
        "",
        "PRIMARY PASSES BY ENGINE",
    ]
    report_lines.extend(f"{source}: {count}" for source, count in engine_counts.most_common())
    report_lines.extend(("", "ALL RESULTS"))
    for row in details:
        status = "PRIMARY" if row["primary_pass"] else "SUGGESTED" if row["candidate_pass"] else "FAILED"
        choices = ", ".join(row["candidates"]) or "(none)"
        report_lines.append(
            f"L{row['line']}: {row['source']} -> {row['expected']} | "
            f"actual: {row['corrected']} | {status} | {row['primary_source']} | choices: {choices}"
        )

    failure_lines = [
        "Failed primary single.dic entries from current evaluation",
        "=========================================================",
        f"Total primary failures: {len(failures)}",
        f"Expected target still suggested: {len(failures) - len(missing)}",
        f"Expected target entirely missing: {len(missing)}",
        "",
    ]
    for row in failures:
        coverage = "suggested" if row["candidate_pass"] else "missing"
        choices = ", ".join(row["candidates"]) or "(none)"
        failure_lines.append(
            f"L{row['line']}: {row['source']}-{row['expected']} | actual: {row['corrected']} | "
            f"expected: {coverage} | choices: {choices}"
        )

    REPORT.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    FAILURES.write_text("\n".join(failure_lines) + "\n", encoding="utf-8")

    print(f"total={total}")
    print(f"primary_passes={primary_passes}")
    print(f"primary_failures={len(failures)}")
    print(f"candidate_passes={candidate_passes}")
    print(f"candidate_missing={len(missing)}")
    print(f"primary_rate={primary_passes / total:.4%}" if total else "primary_rate=0.0000%")
    print(f"candidate_coverage={candidate_passes / total:.4%}" if total else "candidate_coverage=0.0000%")


if __name__ == "__main__":
    main()
