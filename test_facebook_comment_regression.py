# -*- coding: utf-8 -*-
import app


spellchecker = app.spellchecker


COMMENT = """Hemm xi haga hazina ghax rajtu ragel jibki? 
Mela l-irgiel mhux umani wkoll? Ma jistax ikolna mumenti fejn inkunu vulnerabli?
Bhala ragel, kemm il-darba hbejt l-emozzjonijiet  tieghi, ghax nemmen illi figura ta ragel trid tider b'sahhita f'kull hin u mument. Forsi tinstema maskilista, pero dawk huma l-fatti. Pero hadd m'ghandu jidhaq meta jara ragel jibki, sinjal li tant tghabba b'piz u gi emozzjonalment imkisser li ma felahx iktar
Allahares tkunu tafu l'ammont ta irgiel li jiddeciedu li jihdu hajjithom b'idejhom ghax dejjem jibqu silenzjuzi u dan kollhu habba erba cwiec li jitnejku bihom
Zobb f'ghoxx kemm ghandkom!"""


EXPECTED_FRAGMENTS = (
    "mhux umani",
    "ikollna mumenti",
    "Bħala raġel",
    "ħbejt",
    "però dawk huma l-fatti",
    "Però ħadd",
    "dawk huma l-fatti",
    "tant tgħabba",
    "ġie emozzjonalment",
    "tkunu tafu",
    "l-ammont",
    "ta irġiel",
    "jieħdu ħajjithom",
    "b'idejhom",
    "dejjem jibqgħu silenzjużi",
    "dan kollu ħabba",
    "Żobb f'għoxx kemm għandkom",
)


FORBIDDEN_FRAGMENTS = (
    "imhux",
    "kielna",
    "Bħalma",
    "bejt l-emozzjonijiet",
    "kulma l-fatti",
    "t'għant",
    "itgħabba",
    "itkunu",
    "tafgħu",
    "jammonta",
    "għarbiel",
    "bigħ emozzjonalment",
    "biddedhom",
    "ibhejjem",
    "koljuh",
    "ħabbuha",
    "gindakom",
    "gindikom",
    "tinstemax",
    "jinstema",
    "ħajjathom",
)


def assert_word(word, expected, forbidden=()):
    actual = spellchecker.correct_word(word)
    assert actual == expected, f"{word!r}: expected {expected!r}, got {actual!r}"
    suggestions = spellchecker.suggest(word, limit=8, edit_distance_tolerance=1)
    for bad in forbidden:
        assert bad not in suggestions, f"{word!r}: bad suggestion {bad!r} in {suggestions!r}"


assert_word("mhux", "mhux", ["imhux"])
assert_word("ikolna", "ikollna", ["kielna"])
assert_word("bhala", "bħala", ["bħalma", "b'ħala"])
assert_word("hbejt", "ħbejt", ["bejt"])
assert_word("huma", "huma", ["kulma"])
assert_word("tant", "tant", ["t'għant"])
assert_word("tghabba", "tgħabba", ["itgħabba"])
assert_word("tkunu", "tkunu", ["itkunu"])
assert_word("tafu", "tafu", ["tafgħu"])
assert_word("l'ammont", "l-ammont", ["jammonta"])
assert_word("irgiel", "irġiel", ["għarbiel"])
assert_word("jihdu", "jieħdu", ["heddu"])
assert_word("gi", "ġie", ["bigħ"])
assert_word("b'idejhom", "b'idejhom", ["biddedhom"])
assert_word("dejjem", "dejjem", ["ibhejjem", "bhejjem"])
assert_word("silenzjuzi", "silenzjużi")
assert_word("kollhu", "kollu", ["koljuh"])
assert_word("habba", "ħabba", ["ħabbuha"])
assert "ħabbha" in spellchecker.suggest("habba", limit=8, edit_distance_tolerance=1)
assert_word("Zobb", "Żobb")
assert_word("ghandkom", "għandkom", ["gindakom", "gindikom"])
assert_word("pero", "però")
assert_word("Pero", "Però")
assert_word("erba", "erbgħa")
assert "erba'" in spellchecker.suggest("erba", limit=8, edit_distance_tolerance=1)
assert_word("seba", "sebgħa")
assert "seba'" in spellchecker.suggest("seba", limit=8, edit_distance_tolerance=1)
assert_word("disgha", "disgħa")
assert_word("kemmil", "kemm-il")
assert_word("kemm-il", "kemm-il")
assert spellchecker.correct_text_rich("kemm il darba", edit_distance_tolerance=1)[
    "corrected_text"
] == "kemm-il darba"
assert spellchecker.correct_text_rich("kemm id-darba", edit_distance_tolerance=1)[
    "corrected_text"
] == "kemm-il darba"
assert spellchecker.correct_text_rich("fil hanut", edit_distance_tolerance=1)[
    "corrected_text"
] == "fil-ħanut"
assert spellchecker.correct_text_rich("il hwejjeg", edit_distance_tolerance=1)[
    "corrected_text"
] == "il-ħwejjeġ"
assert spellchecker.correct_text_rich(
    "it-tifla marret fil hanut biex tixtri il hwejjeg.",
    edit_distance_tolerance=1,
)["corrected_text"] == "it-tifla marret fil-ħanut biex tixtri l-ħwejjeġ."

tinstema_suggestions = spellchecker.suggest(
    "tinstema",
    limit=8,
    edit_distance_tolerance=3,
)
assert "tinstema'" in tinstema_suggestions
assert "tinstemax" not in tinstema_suggestions
assert "jinstema'" not in tinstema_suggestions

hajjithom_suggestions = spellchecker.suggest(
    "hajjithom",
    limit=8,
    edit_distance_tolerance=3,
)
assert "ħajjithom" in hajjithom_suggestions
assert "ħajjitthom" in hajjithom_suggestions
assert "ħajjathom" not in hajjithom_suggestions


for tolerance in (1, 2, 3):
    corrected = spellchecker.correct_text_rich(
        COMMENT,
        edit_distance_tolerance=tolerance,
    )["corrected_text"]

    for fragment in EXPECTED_FRAGMENTS:
        assert fragment in corrected, (
            f"tick {tolerance}: expected {fragment!r} in {corrected!r}"
        )

    for fragment in FORBIDDEN_FRAGMENTS:
        assert fragment not in corrected, (
            f"tick {tolerance}: forbidden {fragment!r} in {corrected!r}"
        )


print("facebook comment regression checks passed")
