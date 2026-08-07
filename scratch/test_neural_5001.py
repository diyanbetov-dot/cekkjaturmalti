# -*- coding: utf-8 -*-
import sys, os, urllib.request, json
sys.stdout.reconfigure(encoding="utf-8")

url = "http://127.0.0.1:5001/correct"
data = json.dumps({"text": "mort nghum ilbierah"}).encode("utf-8")

try:
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        print("Neural 5001 output:", res)
except Exception as e:
    print("Error querying 5001:", e)
