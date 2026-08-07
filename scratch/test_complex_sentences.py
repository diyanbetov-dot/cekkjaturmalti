"""scratch/test_complex_sentences.py

Diverse and longer sentence benchmark across:
- Port 5000: Main Engine
- Port 5001: Neural-Only (BiGRU v4)
- Port 5002: Hybrid-First Experiment
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TEST_CASES = [
    {
        "id": 1,
        "category": "Complex Multi-Clause Paragraph",
        "input": "Ilbierah meta kont miexi fit-triq, rajt ragel kbir bil-kelb tieghu li kien qieghed jigru fil-gnien pubbliku. Xtaqt nkellmu imma hu kien qed jghagel hafna biex jilhaq il-vapur tal-Ghawdex.",
    },
    {
        "id": 2,
        "category": "Preposition-Article Contractions & Epenthetic Vowels",
        "input": "L-istudenti marru fi il-librerija nazzjonali biex jaqraw dwar il-istorja ta' Malta minn il-kotba l-antiki li sabu fuq ix-xkaffa.",
    },
    {
        "id": 3,
        "category": "Verb Suffixes & Complex Clitics",
        "input": "Meta ghaddejna minn hemm, rajnihom u kellimnihom biex juruna it-triq t-tajba li twassalna sa il-pjazza ewlenija.",
    },
    {
        "id": 4,
        "category": "Contextual Homophones & Orthography",
        "input": "Huwa baghat ittra lill-amministrazzjoni ghax kien hemm zball kbir fir-rapport finanzjarju li gew pprezentat il-gimgha l-ohra.",
    },
    {
        "id": 5,
        "category": "Mixed Loanwords, Punctuation & Complex Structure",
        "input": "Il-professur stqarr li l-progett il-gdid dwar l-intelliġenza artifiċjali ser ikun b'xejn ghax il-gvern ta' Malta se jiffinanzjah kollu kemm hu.",
    },
]

PORTS = [
    (5000, "Port 5000 (Main Engine)"),
    (5001, "Port 5001 (Neural Only)"),
    (5002, "Port 5002 (Hybrid-First)"),
]

def check_port(port: int, text: str) -> tuple[str, float]:
    url = f"http://127.0.0.1:{port}/check-text"
    req = urllib.request.Request(
        url,
        data=json.dumps({"text": text}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            elapsed = time.perf_counter() - start
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("corrected_text", "ERROR"), elapsed
    except Exception as e:
        return f"ERR ({e})", time.perf_counter() - start

print("=" * 90)
print("BENCHMARK: DIVERSE AND LONG SENTENCES ACROSS PORTS 5000, 5001, 5002")
print("=" * 90)

results_summary = []

for case in TEST_CASES:
    print(f"\n[{case['id']}] Category: {case['category']}")
    print(f"INPUT:\n  {case['input']}\n")
    case_res = {"id": case["id"], "category": case["category"], "input": case["input"], "outputs": {}}
    for port, name in PORTS:
        output, elapsed = check_port(port, case["input"])
        print(f"  {name:<25} ({elapsed*1000:.1f} ms):\n    {output}\n")
        case_res["outputs"][name] = {"text": output, "ms": round(elapsed*1000, 1)}
    results_summary.append(case_res)

with open("scratch/complex_benchmark_results.json", "w", encoding="utf-8") as f:
    json.dump(results_summary, f, ensure_ascii=False, indent=2)

print("Benchmark complete. Saved to scratch/complex_benchmark_results.json")
