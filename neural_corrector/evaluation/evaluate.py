from __future__ import annotations

import argparse
import ctypes
import json
import statistics
import sys
import time
from pathlib import Path

from neural_corrector.dataset.analyze_pairs import length_bucket, read_jsonl
from neural_corrector.evaluation.metrics import correction_counts, edit_distance
from neural_corrector.inference.corrector import NeuralCorrector

_LENGTH_BUCKETS = ["short", "medium", "long"]


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


def evaluate_by_length(corrector: NeuralCorrector, rows: list[dict]) -> dict:
    """Run evaluation separately for short, medium, and long inputs."""
    buckets: dict[str, list[dict]] = {b: [] for b in _LENGTH_BUCKETS}
    for row in rows:
        buckets[length_bucket(len(row["noisy"]))].append(row)
    return {
        bucket: evaluate_rows(corrector, bucket_rows) if bucket_rows else {"examples": 0}
        for bucket, bucket_rows in buckets.items()
    }


def _isolate_suffix_word(row: dict) -> dict | None:
    """Return a stripped version of a row with only the suffix word.

    For a row like noisy='jien mort jafrhom' / clean='jien mort jarafhom',
    extracts the word that changed and returns a synthetic row with just that
    word as input and corrected form as output.  Returns None if the changed
    word cannot be isolated cleanly.
    """
    noisy_words = row["noisy"].split()
    clean_words = row["clean"].split()
    if len(noisy_words) != len(clean_words):
        return None  # insertion/deletion — can't safely isolate
    changed = [
        (n, c)
        for n, c in zip(noisy_words, clean_words)
        if n.casefold() != c.casefold()
    ]
    if len(changed) != 1:
        return None  # multiple words changed — ambiguous
    noisy_word, clean_word = changed[0]
    return {
        **row,
        "id": f"{row['id']}:isolated",
        "noisy": noisy_word,
        "clean": clean_word,
        "is_unchanged": noisy_word == clean_word,
    }


def evaluate_suffix_isolation(corrector: NeuralCorrector, rows: list[dict]) -> dict:
    """Evaluate suffix-tagged pairs both in-context and in isolation."""
    suffix_rows = [r for r in rows if "suffix" in r.get("error_tags", [])]
    isolated_rows = [iso for r in suffix_rows if (iso := _isolate_suffix_word(r))]
    return {
        "suffix_in_context": evaluate_rows(corrector, suffix_rows) if suffix_rows else {"examples": 0},
        "suffix_isolated": evaluate_rows(corrector, isolated_rows) if isolated_rows else {"examples": 0},
        "total_suffix_pairs": len(suffix_rows),
        "isolatable_pairs": len(isolated_rows),
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
    parser.add_argument(
        "--stratify-length",
        action="store_true",
        help="Report accuracy separately for short/medium/long inputs.",
    )
    parser.add_argument(
        "--suffix-isolation",
        action="store_true",
        help="Evaluate suffix-tagged pairs both in-context and as isolated words.",
    )
    args = parser.parse_args()
    rows = read_jsonl(args.pairs)
    by_id = {row["id"]: row for row in rows}
    splits = json.loads(args.splits.read_text(encoding="utf-8"))["splits"]
    corrector = NeuralCorrector(args.artifact_dir)
    all_eval_rows = {split: [by_id[item] for item in ids] for split, ids in splits.items() if split != "train"}
    report = {
        "model_version": corrector.model_version,
        "action_threshold": corrector.threshold,
        "model_size_bytes": (args.artifact_dir / "model.pt").stat().st_size,
        "components": {
            "custom_model": True,
            "bertu": False,
            "corpus": False,
            "dictionary_validation": corrector.dictionary_validation_enabled,
            "morphology": False,
            "suffix_runtime": False,
        },
        "splits": {
            split: evaluate_rows(corrector, rows)
            for split, rows in all_eval_rows.items()
        },
    }
    if args.stratify_length:
        report["splits_by_length"] = {
            split: evaluate_by_length(corrector, rows)
            for split, rows in all_eval_rows.items()
        }
    if args.suffix_isolation:
        combined = [row for rows in all_eval_rows.values() for row in rows]
        report["suffix_isolation"] = evaluate_suffix_isolation(corrector, combined)
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
