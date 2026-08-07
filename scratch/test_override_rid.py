"""scratch/test_override_rid.py
Test adding 'rid': ('rrid',) to EXACT_SUGGESTION_OVERRIDES.
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from Essentials.app import spellchecker

# Test before override
print("Suggestions for 'rid' BEFORE override:", spellchecker.suggest("rid"))

# Add override dynamically for test
spellchecker.EXACT_SUGGESTION_OVERRIDES["rid"] = ("rrid",)
spellchecker._request_suggestion_cache().clear()

# Test after override
print("Suggestions for 'rid' AFTER override:", spellchecker.suggest("rid"))

# Test rich text token choices
res = spellchecker.correct_text_rich("li rid nehles minnu")
print("\nTokens choices for 'rid':")
for tok in res.get("tokens", []):
    if tok.get("original") == "rid":
        print("  original:", tok.get("original"))
        print("  choices:", tok.get("choices"))
        print("  ambiguous:", tok.get("ambiguous"))
