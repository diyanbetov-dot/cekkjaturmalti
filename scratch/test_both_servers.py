# -*- coding: utf-8 -*-
import sys, os, urllib.request, json
sys.stdout.reconfigure(encoding="utf-8")

def check(port, name):
    url = f"http://127.0.0.1:{port}/check-text"
    data = json.dumps({"text": "mort nghum ilbierah"}).encode("utf-8")
    try:
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as resp:
            res = json.loads(resp.read().decode("utf-8"))
            print(f"--- Port {port} ({name}) ---")
            print("Corrected:", repr(res.get("corrected_text")))
            if "tokens" in res:
                for t in res["tokens"]:
                    print("  Token:", t)
    except Exception as e:
        print(f"Error on {port}: {e}")

check(5000, "Main Hybrid App")
check(5001, "Neural Corrector v4")
