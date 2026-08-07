# -*- coding: utf-8 -*-
import sys, os
sys.stdout.reconfigure(encoding="utf-8")

log_path = r"C:\Users\diyan\.gemini\antigravity\brain\b3ce9f21-f30d-4a91-bfb2-0260219e983a\.system_generated\tasks\task-3651.log"

with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        if "spellchecker.py(" in line:
            # print lines that are between 14000 and 18000
            try:
                line_no = int(line.split("spellchecker.py(")[1].split(")")[0])
                if 14000 <= line_no <= 18000:
                    print(line.strip())
            except Exception:
                pass
