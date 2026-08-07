from __future__ import annotations

import argparse
import ctypes
import json
import statistics
import sys
import time
from pathlib import Path

from neural_corrector.dataset.analyze_pairs import read_jsonl
from neural_corrector.evaluation.metrics import correction_counts, edit_distance
from neural_corrector.inference.corrector import NeuralCorrector


def peak_working_set_bytes() -> int | None:
    if sys.platform != "win32":
        return None

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    get_current_process = ctypes.windll.kernel32.GetCurrentProcess
    get_current_process.restype = ctypes.c_void_p
    get_process_memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
    get_process_memory_info.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ProcessMemoryCounters),
        ctypes.c_ulong,
    ]
    get_process_memory_info.restype = ctypes.c_int
    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    process = get_current_process()
    succeeded = get_process_memory_info(
        process, ctypes.byref(counters), counters.cb
    )
    return int(counters.PeakWorkingSetSize) if succeeded else None


def evaluate_rows(corrector: NeuralCorrector, rows: list[dict]) -> dict:
    exact = 0
    unchanged_preserved = 0
    clean_total = 0
    tp = fp = fn = 0
    character_errors = character_total = 0
    word_errors = word_total = 0
    changed_characters = changed_words = 0
    timings: list[float] = []
    qualitative = []
    for row in rows:
        result = corrector.correct(row["noisy"])
        predicted = result["corrected_text"]
        timings.append(result["processing_time"])
        exact += predicted == row["clean"]
        if row["noisy"] == row["clean"]:
            clean_total += 1
            unchanged_preserved += predicted == row["clean"]
        row_tp, row_fp, row_fn = correction_counts(
            row["noisy"], predicted, row["clean"]
        )
        tp += row_tp
        fp += row_fp
        fn += row_fn
        character_errors += edit_distance(predicted, row["clean"])
        character_total += max(1, len(row["clean"]))
        word_errors += edit_distance(predicted.split(), row["clean"].split())
        word_total += max(1, len(row["clean"].split()))
        changed_characters += edit_distance(row["noisy"], predicted)
        changed_words += edit_distance(row["noisy"].split(), predicted.split())
        if predicted != row["clean"] and len(qualitative) < 40:
            qualitative.append(
                {
                    "id": row["id"],
                    "input": row["noisy"],
                    "expected": row["clean"],
                    "predicted": predicted,
                    "edits": result["edits"],
                }
            )
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {
        "examples": len(rows),
        "exact_sentence_match": exact / max(1, len(rows)),
        "correction_precision": precision,
        "correction_recall": recall,
        "correction_f1": f1,
        "character_error_rate": character_errors / max(1, character_total),
        "word_error_rate": word_errors / max(1, word_total),
        "clean_sentence_preservation": unchanged_preserved / max(1, clean_total),
        "false_correction_count": fp,
        "missed_correction_count": fn,
        "overcorrection_count": fp,
        "undercorrection_count": fn,
        "changed_characters": changed_characters,
        "changed_words": changed_words,
        "mean_inference_seconds": statistics.fmean(timings) if timings else 0.0,
        "max_inference_seconds": max(timings, default=0.0),
        "qualitative_errors": qualitative,
    }


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
        default=Path("neural_corrector/experiments/baseline_evaluation.json"),
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    rows = read_jsonl(args.pairs)
    by_id = {row["id"]: row for row in rows}
    splits = json.loads(args.splits.read_text(encoding="utf-8"))["splits"]
    corrector = NeuralCorrector(args.artifact_dir)
    report = {
        "model_version": corrector.model_version,
        "action_threshold": corrector.threshold,
        "model_size_bytes": (args.artifact_dir / "model.pt").stat().st_size,
        "components": {
            "custom_model": True,
            "bertu": False,
            "corpus": False,
            "dictionary_validation": False,
            "morphology": False,
            "suffix_runtime": False,
        },
        "splits": {
            split: evaluate_rows(corrector, [by_id[item] for item in ids])
            for split, ids in splits.items()
            if split != "train"
        },
    }
    report["peak_process_working_set_bytes"] = peak_working_set_bytes()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if not args.quiet:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(
            json.dumps(
                {
                    "model_version": report["model_version"],
                    "action_threshold": report["action_threshold"],
                    "model_size_bytes": report["model_size_bytes"],
                    "peak_process_working_set_bytes": report[
                        "peak_process_working_set_bytes"
                    ],
                    "splits": {
                        name: {
                            key: value
                            for key, value in metrics.items()
                            if key
                            in {
                                "examples",
                                "exact_sentence_match",
                                "correction_precision",
                                "correction_recall",
                                "correction_f1",
                                "character_error_rate",
                                "word_error_rate",
                                "clean_sentence_preservation",
                                "false_correction_count",
                                "missed_correction_count",
                                "mean_inference_seconds",
                            }
                        }
                        for name, metrics in report["splits"].items()
                    },
                },
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
