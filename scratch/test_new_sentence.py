"""scratch/test_new_sentence.py
Test sentence: "Li namlu namluh mien qalbna habib Atlek taqtax qalbek ax tismakom daqs li kiku kont tifel taha."
"""
from __future__ import annotations
import json
import sys
import urllib.request

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

text = "Li namlu namluh mien qalbna habib Atlek taqtax qalbek ax tismakom daqs li kiku kont tifel taha."

for port, name in [(5000, "Port 5000 (Main)"), (5001, "Port 5001 (Neural v6)"), (5002, "Port 5002 (Hybrid v6)")]:
    try:
        url = f"http://127.0.0.1:{port}/check-text"
        req = urllib.request.Request(
            url,
            data=json.dumps({"text": text}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            print(f"=== {name} ===")
            print("Corrected:", data.get("corrected_text"))
    except Exception as e:
        print(f"=== {name} === ERROR: {e}")
