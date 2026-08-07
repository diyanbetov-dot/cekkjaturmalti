"""scratch/verify_all_live_v5_ports.py
Query live HTTP ports 5000, 5001, and 5002 for agreement, suffixed, and pre-verb test cases.
"""
from __future__ import annotations
import json
import sys
import urllib.request
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TEST_CASES = [
    "ha niggennen",
    "ħa niggennen",
    "ir-ragel ha niggennen",
    "ir-ragel ha thawdu",
    "il-mara ha tiggennen",
    "il-mara ha thawdha",
    "ha tabqizli",
    "seba jiem",
    "tlett gzejjer",
    "erba bozza",
]

PORTS = [
    (5000, "Port 5000 (Main)"),
    (5001, "Port 5001 (Neural v5)"),
    (5002, "Port 5002 (Hybrid v5)"),
]

print("Waiting 15 seconds for servers to settle...")
time.sleep(15)

print("=" * 80)
print("LIVE HTTP PORT VERIFICATION (v5 BI-GRU & GRAMMAR RULE ENGINE)")
print("=" * 80)

for text in TEST_CASES:
    print(f"\nINPUT: {text!r}")
    for port, name in PORTS:
        try:
            url = f"http://127.0.0.1:{port}/check-text"
            req = urllib.request.Request(
                url,
                data=json.dumps({"text": text}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                corrected = data.get("corrected_text", "ERROR")
                print(f"  {name:<22}: {corrected!r}")
        except Exception as e:
            print(f"  {name:<22}: ERR ({e})")
