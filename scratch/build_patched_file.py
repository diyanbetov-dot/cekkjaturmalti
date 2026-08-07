"""scratch/build_patched_file.py
Write the user's 100 patched cases to scratch/patched_sample_100.txt if not already present.
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PATCHED_PATH = ROOT / "scratch" / "patched_sample_100.txt"

print(f"Patched file exists: {PATCHED_PATH.exists()}")
