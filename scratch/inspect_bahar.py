# -*- coding: utf-8 -*-
import sys, os, json
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, ".")

from Essentials.app import spellchecker

sentence = "mort sal bahar biex nowm ftit."

print("=== INPUT SENTENCE ===")
print(sentence)
print()

print("=== 1. WORD SUGGESTIONS FOR 'bahar' ===")
print("suggest('bahar'):", spellchecker.suggest("bahar"))
print("is 'bahar' in dictionary_set?", "bahar" in spellchecker.dictionary_set)
print("is 'baħar' in dictionary_set?", "baħar" in spellchecker.dictionary_set)
print()

print("=== 2. WORD SUGGESTIONS FOR 'sal' ===")
print("suggest('sal'):", spellchecker.suggest("sal"))
print("is 'sal' in dictionary_set?", "sal" in spellchecker.dictionary_set)
print()

print("=== 3. WORD SUGGESTIONS FOR 'nowm' ===")
print("suggest('nowm'):", spellchecker.suggest("nowm"))
print()

print("=== 4. FULL SENTENCE CORRECTION (correct_text_rich) ===")
res = spellchecker.correct_text_rich(sentence)
print("Corrected text:", res["corrected_text"])
print("Tokens:")
for tok in res.get("tokens", []):
    if tok.get("type") == "word":
        print(f"  - '{tok.get('original')}' -> '{tok.get('corrected')}', choices={[c['word'] for c in tok.get('choices', [])[:3]]}")
