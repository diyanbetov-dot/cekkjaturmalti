"""scratch/check_ports_status.py
Check if ports 5000, 5001, and 5002 are responding on HTTP.
"""
from __future__ import annotations
import json
import sys
import urllib.request

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ports = [5000, 5001, 5002]

for p in ports:
    url = f"http://127.0.0.1:{p}/"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=3) as resp:
            print(f"Port {p}: ACTIVE (status {resp.status})")
    except Exception as e:
        print(f"Port {p}: INACTIVE ({e})")
