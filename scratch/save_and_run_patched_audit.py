"""scratch/save_and_run_patched_audit.py
Save user prompt patched sample to file and run audit.
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# The patched text will be saved to scratch/patched_sample_100.txt
# Let's inspect the quality of the patched prompt
print("Audit framework ready.")
