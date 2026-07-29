from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Essentials import app


spellchecker = app.spellchecker


def rich(source: str) -> dict:
    return spellchecker.correct_text_rich(source, edit_distance_tolerance=2)


def lexical_tokens(result: dict) -> list[dict]:
    return [
        token
        for token in result["tokens"]
        if token.get("type") in {"word", "phrase"}
    ]


def token_for(source: str, original: str) -> dict:
    for token in lexical_tokens(rich(source)):
        if token.get("original") == original:
            return token
    raise AssertionError(f"{source!r}: token {original!r} not found")


assert rich("iz-zija")["tokens"][0]["choices"] == []
assert rich("it-tifel")["tokens"][0]["choices"] == []
assert len(rich("il bahar")["tokens"][0]["choices"]) == 3
assert len(rich("l-bahar")["tokens"][0]["choices"]) == 3

ganna = token_for("iz-zija Ganna", "Ganna")
assert ganna["corrected"] == "Ganna"
assert "G\u0127anna" not in [choice["word"] for choice in ganna["choices"]]

assert rich("huma jistaw imorru")["corrected_text"] == (
    "Huma jistg\u0127u jmorru."
)
assert rich("kienu joqodu ixejjrula")["corrected_text"] == (
    "Kienu joqog\u0127du jxejrulha."
)

striehu = token_for("Striehu fis-sliem", "Striehu")
assert striehu["corrected"] == "Strie\u0127u"
assert "Stri\u0127u" not in [choice["word"] for choice in striehu["choices"]]

quoted = token_for('"aotomatic" illum', '"aotomatic"')
assert quoted["corrected"] == "aotomatic"
assert quoted["unrecognized"] is False
multiline_quoted = rich('"biex kultant\ninfakkar li ......."\nmhiex twila')
assert multiline_quoted["corrected_text"].startswith(
    "biex kultant\ninfakkar li......."
)
assert '"' not in multiline_quoted["corrected_text"]

for dual in ("jumejn", "sentejn"):
    dual_token = token_for(f"{dual} ilu", dual)
    assert dual_token["unrecognized"] is False

assert rich("Striehu fi sliem")["corrected_text"].endswith(".")

for source, expected in (
    ("nilaghbu", "Nilag\u0127bu."),
    ("jibatu", "Jibag\u0127tu."),
    ("joqodu", "Joqog\u0127du."),
):
    result = rich(source)
    assert result["corrected_text"] == expected
    choices = lexical_tokens(result)[0]["choices"]
    assert all("g\u0127a" not in choice["word"].casefold() for choice in choices)

assert spellchecker.correct_word("ghidtlu") == "g\u0127edtlu"
assert spellchecker.correct_word("jehduna") == "je\u0127duna"

for source, expected in (
    ("hadha", "\u0127adha"),
    ("hadni", "\u0127adni"),
    ("hadhom", "\u0127adhom"),
    ("hadhulhom", "\u0127adhulhom"),
):
    assert spellchecker.correct_word(source) == expected

for source in (
    "ffortunat",
    "ffortunata",
    "ffortunati",
    "\u0127dejja",
    "\u0127dejk",
    "\u0127dejh",
    "\u0127dejha",
    "\u0127dejna",
    "\u0127dejkom",
    "\u0127dejhom",
    "fuqi",
    "fuqek",
    "fuqu",
    "fuqha",
    "fuqna",
    "fuqkom",
    "fuqhom",
):
    assert token_for(f"illum {source}", source)["unrecognized"] is False

assert token_for("ghadek fl-ilma", "fl-ilma")["unrecognized"] is False
assert rich("mhiex twila hajjitna")["corrected_text"] == (
    "Mhix twila \u0127ajjitna."
)
assert token_for("mhiex twila hajjitna", "hajjitna")["choices"] == []
assert spellchecker.correct_word("hajjitnieha") == "\u0127ajjitnieha"
assert spellchecker.correct_word("ghamilnieha") == "g\u0127amilnieha"

assert rich("kuljum ghed tiqsar")["corrected_text"] == "Kuljum qed tiqsar."
assert rich("filoghodu")["corrected_text"] == "Filg\u0127odu."
assert rich("tkun tifel u tigber")["corrected_text"] == (
    "Tkun tifel u tikber."
)
assert rich("malajr issir xih")["corrected_text"] == "Malajr issir xi\u0127."

for source, alternative in (("binhar", "Bi nhar"), ("billejl", "Bil-lejl")):
    choices = lexical_tokens(rich(source))[0]["choices"]
    assert alternative in [choice["word"] for choice in choices]

niehdu = token_for("konna niehdu gost", "niehdu")
assert "nie\u0127du" in [choice["word"] for choice in niehdu["choices"]]

bulk_result = rich(" ".join(["konna niehdu gost"] * 25))
bulk_niehdu = next(
    token
    for token in lexical_tokens(bulk_result)
    if token.get("original") == "niehdu"
)
assert "nie\u0127du" in [choice["word"] for choice in bulk_niehdu["choices"]]
