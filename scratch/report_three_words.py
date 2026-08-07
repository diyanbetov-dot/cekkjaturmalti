"""scratch/report_three_words.py
Query the 3 live web app endpoints (Port 5000, Port 5001, Port 5002)
for the three words: 'amilulu', 'ibatilu', 'israqomlu'
and compare outputs against expected outputs: 'agħmilhulu', 'ibgħathielu', 'israqhomlu'.
"""
from __future__ import annotations
import json
import sys
import urllib.request

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TEST_WORDS = [
    ("amilulu", "agħmilhulu"),
    ("ibatilu", "ibgħathielu"),
    ("israqomlu", "israqhomlu"),
]

PORTS = [
    (5000, "Port 5000 (Main)"),
    (5001, "Port 5001 (Neural v5)"),
    (5002, "Port 5002 (Hybrid v5)"),
]

results = {}

for word, expected in TEST_WORDS:
    results[word] = {"expected": expected, "ports": {}}
    for port, name in PORTS:
        try:
            url = f"http://127.0.0.1:{port}/check-text"
            req = urllib.request.Request(
                url,
                data=json.dumps({"text": word}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                corrected = data.get("corrected_text", "ERROR").rstrip(".")
                results[word]["ports"][port] = corrected
        except Exception as e:
            results[word]["ports"][port] = f"ERR: {e}"

print(json.dumps(results, indent=2, ensure_ascii=False))
