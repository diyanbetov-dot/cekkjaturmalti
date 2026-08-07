"""scratch/test_all_three_ports.py

Comparative evaluation of all three local spellchecking services:
- Port 5000: Main hybrid engine
- Port 5001: Neural-only (BiGRU v4)
- Port 5002: New Hybrid-First experimental branch
"""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TEST_SENTENCES = [
    "ilbierah mort sal bahar biex nowm ftit.",
    "mort nghum ilbierah.",
    "ktibt ittra lill-kbira tieghu imma ma wegibtx.",
    "xtaqt niehu kafè mal habib tieghi.",
    "kellimna dak ir-ragel li kien jaf kollox.",
    "rajt il-klieb jiġru fil-gnien.",
    "huwa jkellem lit-tfal b'mod sabih.",
    "il-qattus qieghed fuq il-mejda.",
]

PORTS = [
    (5000, "Port 5000 (Main Engine)"),
    (5001, "Port 5001 (Neural Only)"),
    (5002, "Port 5002 (Hybrid-First)"),
]

def check_port(port: int, text: str) -> str:
    url = f"http://127.0.0.1:{port}/check-text"
    req = urllib.request.Request(
        url,
        data=json.dumps({"text": text}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("corrected_text", "ERROR")
    except Exception as e:
        return f"ERR ({e})"

print("=" * 80)
print("THREE-PORT COMPARATIVE EVALUATION")
print("=" * 80)

for i, s in enumerate(TEST_SENTENCES, 1):
    print(f"\n[{i}] INPUT: {s}")
    for port, name in PORTS:
        res = check_port(port, s)
        print(f"    {name:<25}: {res}")

print("\n" + "=" * 80)
print("CHECKING SUGGESTION LABELS (NOISY MEANINGS CHECK)")
print("=" * 80)

for port, name in PORTS:
    url = f"http://127.0.0.1:{port}/check-text"
    req = urllib.request.Request(
        url,
        data=json.dumps({"text": "ilbierah mort sal bahar"}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            tokens = data.get("tokens", [])
            noisy_meanings = []
            for tok in tokens:
                if isinstance(tok, dict) and tok.get("type") == "word":
                    for ch in tok.get("choices", []):
                        if isinstance(ch, dict):
                            m = ch.get("meaning", "")
                            if "neural" in m.lower() or "alternative" in m.lower() or "predicts" in m.lower():
                                noisy_meanings.append(m)
            status = f"CLEAN (0 noisy labels)" if not noisy_meanings else f"NOISY: {noisy_meanings}"
            print(f"    {name:<25}: {status}")
    except Exception as e:
        print(f"    {name:<25}: ERR ({e})")
