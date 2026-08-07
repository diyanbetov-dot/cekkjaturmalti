from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from neural_corrector.dataset.analyze_pairs import read_jsonl
from neural_corrector.evaluation.evaluate import evaluate_rows
from neural_corrector.inference.corrector import NeuralCorrector

DEFAULT_THRESHOLDS = (0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("neural_corrector/artifacts/char_edit_bigru_v1"),
    )
    parser.add_argument(
        "--pairs",
        type=Path,
        default=Path("neural_corrector/data/processed/all_pairs.jsonl"),
    )
    parser.add_argument(
        "--splits",
        type=Path,
        default=Path("neural_corrector/data/splits/LOCKED_SPLITS.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "neural_corrector/experiments/threshold_calibration.json"
        ),
    )
    args = parser.parse_args()

    rows_by_id = {row["id"]: row for row in read_jsonl(args.pairs)}
    split_payload = json.loads(args.splits.read_text(encoding="utf-8"))
    validation_rows = [
        rows_by_id[example_id]
        for example_id in split_payload["splits"]["validation"]
    ]
    corrector = NeuralCorrector(args.artifact_dir)
    candidates = []
    for threshold in DEFAULT_THRESHOLDS:
        corrector.threshold = threshold
        metrics = evaluate_rows(corrector, validation_rows)
        candidates.append({"threshold": threshold, "metrics": metrics})

    viable = [
        row
        for row in candidates
        if row["metrics"]["correction_precision"] >= 0.60
    ]
    if viable:
        selected = min(
            viable,
            key=lambda row: (
                row["metrics"]["character_error_rate"],
                row["metrics"]["false_correction_count"],
            ),
        )
        policy = "lowest validation CER among thresholds with precision >= 0.60"
    else:
        selected = max(
            candidates, key=lambda row: row["metrics"]["correction_f1"]
        )
        policy = "highest validation correction F1"

    report = {
        "selection_partition": "validation",
        "test_partitions_read": False,
        "policy": policy,
        "selected_threshold": selected["threshold"],
        "candidates": candidates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.artifact_dir / "inference_config.json").write_text(
        json.dumps(
            {
                "action_threshold": selected["threshold"],
                "calibration_report": str(args.output),
                "partition": "validation",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
