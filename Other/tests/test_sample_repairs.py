# -*- coding: utf-8 -*-
import app


spellchecker = app.spellchecker


def assert_correct(word, expected):
    actual = spellchecker.correct_word(word)
    assert actual == expected, f"{word!r}: expected {expected!r}, got {actual!r}"


def assert_first_suggestion(word, expected):
    suggestions = spellchecker.suggest(word, limit=8, edit_distance_tolerance=1)
    assert suggestions, f"{word!r}: expected suggestions, got none"
    assert (
        suggestions[0] == expected
    ), f"{word!r}: expected first suggestion {expected!r}, got {suggestions!r}"


CASES = {
    "bixiraq": "bix-xieraq",
    "ittoqba": "it-toqba",
    "toqot": "toqgħod",
    "tajat": "għajjat",
    "tajru": "tgħajru",
    "biddifeti": "bid-difetti",
    "ijskom": "qiskom",
    "manafx": "ma nafx",
    "di": "din",
    "issabih": "is-sabiħ",
    "tajar": "tgħajjar",
    "tijak": "tiegħek",
    "sabijh": "sabiħ",
    "hanaraw": "ħa naraw",
    "senaraw": "se naraw",
    "sanaraw": "sa naraw",
    "hanmur": "ħa mmur",
    "senmur": "se mmur",
    "sanmur": "sa mmur",
}


for typo, expected in CASES.items():
    assert_correct(typo, expected)
    assert_first_suggestion(typo, expected)

assert_correct("tajtu", "tajtu")
assert spellchecker.meaning_for("tajtu") == "to give"


mandix_suggestions = spellchecker.suggest("mandix", limit=8, edit_distance_tolerance=1)
assert "m'għandix" in mandix_suggestions
assert "m'għandhiex" in mandix_suggestions


dejaqni_suggestions = spellchecker.suggest("dejaqni", limit=8, edit_distance_tolerance=1)
assert dejaqni_suggestions[0] == "dejjaqni"
assert "dejjiqni" not in dejaqni_suggestions


sample = (
    "uwajma bixiraq andek ittoqba ta sormok 'loose' int ha toqot tajat ax "
    "mandix sider xorta 'sexy' issa jin vera dejaqni imma di li tajru "
    "biddifeti ijskom perfeti manafx jin kulhad andu issabih u likrah tijaw "
    "jew almenu qabel tajar biddifeti ajjar bil 'profile' tijak hanaraw kem "
    "int sabijh."
)

corrected = spellchecker.correct_text_rich(sample, edit_distance_tolerance=1)[
    "corrected_text"
]

for expected_fragment in (
    "bix-xieraq",
    "it-toqba",
    "toqgħod",
    "għajjat",
    "m'għandix",
    "bid-difetti",
    "qiskom",
    "ma nafx",
    "is-sabiħ",
    "tgħajjar",
    "tiegħek",
    "ħa naraw",
    "sabiħ",
):
    assert expected_fragment in corrected, (
        f"expected {expected_fragment!r} in corrected sample, got {corrected!r}"
    )

print("sample repair regression checks passed")
