import sys
from app import UniversalMalteseSpellchecker

spellchecker = UniversalMalteseSpellchecker()

cases = [
    "tal 'washing machine'",
    "tas 'supermarket'",
    "għall 'apple'",
    "mal 'drone'",
    "fil 'server'",
    "is 'washing machine'"
]

for c in cases:
    print(f"IN:  {c}")
    result = spellchecker.correct_text_rich(c)
    print(f"OUT: {result['corrected_text']}")
    print()
