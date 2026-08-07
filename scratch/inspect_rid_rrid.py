"""scratch/inspect_rid_rrid.py
Inspection script for 'rid' vs 'rrid' in context.
INSPECT ONLY - NO EDITS TO ENGINE CODE.
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

from Essentials.app import spellchecker, meaning_index, grammar_rule_engine
from neural_corrector.inference.corrector import NeuralCorrector

text = "manafx xha naqbad namel andi hafna dwejjaq li rid nehles minnu."
word = "rid"

print("=" * 80)
print(f"INSPECTION: '{word}' -> 'rrid' in context: \"{text}\"")
print("=" * 80)

# 1. Dictionary Presence & Meanings
print("\n1. DICTIONARY STATUS OF 'rid' AND 'rrid':")
print(f"  'rid' in dictionary_set: {word in spellchecker.dictionary_set}")
print(f"  'rid' meaning: {spellchecker.meaning_for(word)}")
print(f"  'rid' tags: {spellchecker.word_tags.get(word, set())}")

print(f"\n  'rrid' in dictionary_set: {'rrid' in spellchecker.dictionary_set}")
print(f"  'rrid' meaning: {spellchecker.meaning_for('rrid')}")
print(f"  'rrid' tags: {spellchecker.word_tags.get('rrid', set())}")

# 2. Standalone Spellchecker Correction
print("\n2. STANDALONE CORRECT_WORD:")
print(f"  correct_word('rid'): {spellchecker.correct_word('rid')}")
print(f"  suggest('rid'): {spellchecker.suggest('rid', limit=5)}")

# 3. Main Engine vs Neural vs Hybrid on Full Sentence
print("\n3. PIPELINE BEHAVIOR ON SENTENCE:")
for port, name in [(5000, "Main Engine (5000)"), (5001, "Neural Only (5001)"), (5002, "Hybrid-First (5002)")]:
    try:
        url = f"http://127.0.0.1:{port}/check-text"
        req = urllib.request.Request(
            url,
            data=json.dumps({"text": text}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            print(f"  {name:<25}: {data.get('corrected_text')}")
    except Exception as e:
        print(f"  {name:<25}: ERR ({e})")

# 4. Corpus Scorer Frequencies (if corpus available)
print("\n4. CORPUS SCORER / FREQUENCY INSPECTION:")
cs = getattr(spellchecker, "corpus_scorer", None)
if cs and hasattr(cs, "unigram_counts"):
    rid_count = cs.unigram_counts.get("rid", 0)
    rrid_count = cs.unigram_counts.get("rrid", 0)
    li_rid_count = cs.bigram_counts.get(("li", "rid"), 0)
    li_rrid_count = cs.bigram_counts.get(("li", "rrid"), 0)
    print(f"  Unigram count for 'rid': {rid_count}")
    print(f"  Unigram count for 'rrid': {rrid_count}")
    print(f"  Bigram count for ('li', 'rid'): {li_rid_count}")
    print(f"  Bigram count for ('li', 'rrid'): {li_rrid_count}")
else:
    print("  Corpus scorer unigram counts object:", type(cs))

# 5. Verb Record Lookup
print("\n5. VERB RECORD LOOKUP:")
print(f"  'rid' verb records: {spellchecker._verb_records_for_surface('rid')}")
print(f"  'rrid' verb records: {spellchecker._verb_records_for_surface('rrid')}")
