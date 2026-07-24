import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from Essentials import app

sc = app.spellchecker

CASES = [
    # digit + il- + noun  (three tokens via article_word path)
    "11 il-siegha",
    "11 is-siegha",
    "11 il-ktieb",
    # digit-il + noun  (two tokens, compact)
    "11-il siegha",
    "15-il ktieb",
    # digit standalone (should passthrough unchanged)
    "ghandha 12 sena",
    # word numerals for comparison
    "hdax il-siegha",
    "hmistax il-ktieb",
    # kemm
    "kemm il-siegha",
    "kemm is-siegha",
]

for text in CASES:
    result = sc.correct_text_rich(text, edit_distance_tolerance=1)
    print(f"IN:  {text}")
    print(f"OUT: {result['corrected_text']}")
    print()
