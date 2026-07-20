# -*- coding: utf-8 -*-
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Other.tools.repair_mojibake import repair_mojibake_text


def test_repairs_common_maltese_mojibake_sequences():
    broken = (
        "GÄ§andi bÅ¼onn niÄ‹Ä‹ekkja Ä¡urnata, Ä§obÅ¼a, "
        "Ä¦add, ÄŠikku, Å»ejtun."
    )
    assert (
        repair_mojibake_text(broken)
        == "Għandi bżonn niċċekkja ġurnata, ħobża, Ħadd, Ċikku, Żejtun."
    )


def test_preserves_already_correct_maltese_while_repairing_mixed_text():
    mixed = "Għandi gÄ§axar żwiemel u Å¼ewÄ¡ kelmiet."
    assert repair_mojibake_text(mixed) == "Għandi għaxar żwiemel u żewġ kelmiet."


def test_preserves_unrecoverable_replacement_characters():
    assert repair_mojibake_text("g�andek") == "g�andek"
