"""scratch/merge_agreement_dataset.py
Merge synthetic_agreement_train.jsonl into all_pairs.jsonl
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ALL_PAIRS_PATH = ROOT / "neural_corrector" / "data" / "processed" / "all_pairs.jsonl"
AGREEMENT_PATH = ROOT / "neural_corrector" / "data" / "processed" / "synthetic_agreement_train.jsonl"

existing_ids = set()
existing_rows = []

if ALL_PAIRS_PATH.exists():
    with ALL_PAIRS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                existing_ids.add(row["id"])
                existing_rows.append(row)

print(f"Existing rows in all_pairs.jsonl: {len(existing_rows)}")

added_count = 0
with AGREEMENT_PATH.open("r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            row = json.loads(line)
            if row["id"] not in existing_ids:
                existing_rows.append(row)
                existing_ids.add(row["id"])
                added_count += 1

print(f"Added {added_count} new synthetic agreement rows.")
print(f"Total rows now in all_pairs.jsonl: {len(existing_rows)}")

with ALL_PAIRS_PATH.open("w", encoding="utf-8") as f:
    for row in existing_rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print("Merge complete!")
