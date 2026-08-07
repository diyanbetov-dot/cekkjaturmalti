from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from neural_corrector.inference.corrector import NeuralCorrector


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = argparse.ArgumentParser(description="Neural-first Maltese corrector")
    parser.add_argument("text")
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("neural_corrector/artifacts/char_edit_bigru_v2"),
    )
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = NeuralCorrector(args.artifact_dir, args.threshold).correct(args.text)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(result["corrected_text"])
        print(
            f"confidence={result['confidence']:.4f} "
            f"time={result['processing_time']:.4f}s "
            f"model={result['model_version']}"
        )
        for edit in result["edits"]:
            print(json.dumps(edit, ensure_ascii=False))


if __name__ == "__main__":
    main()
