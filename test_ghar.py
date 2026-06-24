import sys
from app import UniversalMalteseSpellchecker
spellchecker = UniversalMalteseSpellchecker()
result = spellchecker.correct_text_rich("ghar 'roundabout'")
print(f"OUT: {result['corrected_text']}")
