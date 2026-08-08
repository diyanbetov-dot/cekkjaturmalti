import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from spellchecker.config import DATA_ARTIFACTS_DIR


def analyze_missing():
    missing_file = DATA_ARTIFACTS_DIR / "oracle_missing_train.jsonl"
    if not missing_file.exists():
        print("Missing file does not exist.")
        return

    records = []
    with open(missing_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line.strip()))

    print(f"Total missing records: {len(records)}\n")
    print("Sample missing records (first 40):")
    for r in records[:40]:
        print(f"[{r['reason']}] '{r['gold_src']}' -> '{r['gold_tgt']}' | Context: {r['source_sentence']!r}")


if __name__ == "__main__":
    analyze_missing()
