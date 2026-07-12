# -*- coding: utf-8 -*-
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Essentials import app


s = app.spellchecker

samples = [
    "nmut",
    "inmut",
    "inmutu",
    "landu",
    "landom",
    "inhalasa",
    "inhalasom",
    "xar",
    "ntbat",
    "ifitex",
    "xikun",
    "x'ikun",
    "x'imkien",
    "tibblukkali",
    "sidha",
    "paxxut",
    "paxxuta",
    "paxxuti",
    "irrid",
    "jara",
]

for word in samples:
    print(f"{word} => {s.correct_word(word)} | sugg: {s.suggest(word, limit=8)}")

texts = [
    "'Hi', fejn tista' ccempel ghax hawn karozza qed tibblukkali l- garaxx please u sidha qieghed paxxut jara n- nar x'imkien u jien irrid nohrog ghax- xoghol?",
    "wara inmut",
    "issa nmut",
    "xikun jaf",
    "x'ikun jaf",
]

for text in texts:
    print("TEXT:", text)
    print(s.correct_text_rich(text, edit_distance_tolerance=2)["corrected_text"])
