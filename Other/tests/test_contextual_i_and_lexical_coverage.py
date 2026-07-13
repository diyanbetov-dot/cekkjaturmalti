# -*- coding: utf-8 -*-
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Essentials import app


spellchecker = app.spellchecker


def corrected_text(source: str) -> str:
    return spellchecker.correct_text_rich(
        source,
        edit_distance_tolerance=2,
    )["corrected_text"]


def token_for(source: str, original: str):
    result = spellchecker.correct_text_rich(
        source,
        edit_distance_tolerance=2,
    )
    for token in result["tokens"]:
        if token.get("original") == original:
            return token
    raise AssertionError(f"{source!r}: token {original!r} not found")


assert corrected_text("qal, programm") == "Qal, programm."
assert corrected_text("qal programm") == "Qal programm."
assert corrected_text("Wara jmorru d-dar") == "Wara imorru d-dar."
assert corrected_text("Kien imorru d-dar") == "Kien jmorru d-dar."
assert corrected_text("uriet kollox") == "Wriet kollox."
assert corrected_text("ifitex") == "Ifittex."
assert spellchecker.correct_word("ifitex") == "jfittex"
assert spellchecker.correct_word("inkiser") == "nkiser"
assert spellchecker.correct_word("Inkiser") == "Nkiser"
assert corrected_text("xhar") == "Xahar."
assert "xahar" in spellchecker.suggest("xar", limit=8)
assert corrected_text("xgħar") == "Xgħar."
assert {"ntbagħat", "ntbagħt"}.issubset(set(spellchecker.suggest("ntbat", limit=8)))
assert spellchecker.correct_word("inhalasa") == "nħallasha"
assert spellchecker.correct_word("inhalasom") == "nħallashom"
assert spellchecker.correct_word("deru") == "deru"
assert "dehru" in spellchecker.suggest("deru", limit=8)
assert corrected_text("presentuza") == "Preżentuża."
assert spellchecker.correct_word("landu") == "l'għandu"
assert spellchecker.correct_word("landhom") == "l'għandhom"
assert spellchecker.correct_word("landom") == "l'għandhom"
assert spellchecker.correct_word("nmut") == "mmut"
assert spellchecker.correct_word("inmut") == "immut"
assert spellchecker.correct_word("nmutu") == "mmutu"
assert spellchecker.correct_word("inmutu") == "immutu"
assert spellchecker.correct_word("irrid") == "irrid"
assert spellchecker.correct_word("iddur") == "iddur"
assert spellchecker.correct_word("xikun") == "xi jkun"
assert spellchecker.correct_word("x'ikun") == "xi jkun"
assert spellchecker.correct_word("x'imkien") == "xi mkien"
assert corrected_text("uċuħ sbieħ") == "Uċuħ sbieħ."
assert corrected_text("wċuħ sbieħ") == "Uċuħ sbieħ."
assert corrected_text("uġigħ kbir") == "Uġigħ kbir."
assert corrected_text("wġigħ kbir") == "Uġigħ kbir."

assert corrected_text("misterjuz") == "Misterjuż."
assert corrected_text("gustuz") == "Gustuż."
assert corrected_text("ma' gustuz") == "ma' gustuż."

gustuz_token = token_for("ma' gustuz", "ma' gustuz")
assert gustuz_token["corrected"] == "ma' gustuż"
assert gustuz_token.get("ambiguous") is False
assert gustuz_token.get("crucial") in {False, None}

assert corrected_text("kontestant") == "Kontestant."
assert corrected_text("apparti") == "Apparti."
assert corrected_text("sew") == "Sew."

assert spellchecker.correct_word("misterjuż") == "misterjuż"
assert spellchecker.correct_word("gustuż") == "gustuż"
assert spellchecker.correct_word("kontestant") == "kontestant"
assert spellchecker.correct_word("apparti") == "apparti"
assert spellchecker.correct_word("sew") == "sew"
assert spellchecker.correct_word("wġigħ") == "wġigħ"
assert spellchecker.correct_word("uġigħ") == "uġigħ"

print("contextual i and lexical coverage checks passed")
