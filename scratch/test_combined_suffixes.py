"""scratch/test_combined_suffixes.py
Inspect and trace combined double-suffix verbs:
- 'amilulu' (għamlulu / għamluhulu)
- 'ibatilu' (jibgħatulu / ibgħatulu)
- 'israqomlu' (jisraqhomlu / israqhomlu)

Test candidate generation, suggestions, neural v5, and live endpoints.
"""
from __future__ import annotations
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from Essentials.app import spellchecker
from neural_corrector.inference.corrector import NeuralCorrector

V5_ARTIFACT = Path("neural_corrector/artifacts/char_edit_bigru_v5")
nc_v5 = NeuralCorrector(V5_ARTIFACT)

test_words = [
    "amilulu",
    "gamlulu",
    "gamluhulu",
    "ibatilu",
    "jibgatulu",
    "israqomlu",
    "jisraqhomlu",
]

print("=" * 80)
print("1. SPELLCHECKER CANDIDATE & SUGGESTION TRACE")
print("=" * 80)

for w in test_words:
    corr = spellchecker.correct_word(w)
    suggs = spellchecker.suggest(w)
    print(f"Word: {w!r:<15} -> Corrected: {corr!r:<15} Suggestions: {suggs[:4]}")

print("\n" + "=" * 80)
print("2. NEURAL MODEL BiGRU v5 PREDICTIONS")
print("=" * 80)

for w in test_words:
    res = nc_v5.correct(w)
    print(f"Word: {w!r:<15} -> Neural v5 Output: {res['corrected_text']!r:<20} Changed: {res['changed']}")

print("\n" + "=" * 80)
print("3. LIVE HTTP ENDPOINT RESULTS ACROSS PORTS")
print("=" * 80)

test_phrases = [
    "ma amilulu xejn",
    "huma amilulu il-favur",
    "ma ibatilu xejn",
    "israqomlu kollha",
]

for phrase in test_phrases:
    print(f"\nPhrase: {phrase!r}")
    for port in [5000, 5001, 5002]:
        try:
            url = f"http://127.0.0.1:{port}/check-text"
            req = urllib.request.Request(
                url,
                data=json.dumps({"text": phrase}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                print(f"  Port {port}: {data.get('corrected_text')!r}")
        except Exception as e:
            print(f"  Port {port}: ERR ({e})")
