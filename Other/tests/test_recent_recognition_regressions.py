# -*- coding: utf-8 -*-
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Essentials"))

import app


def _unrecognized_words(result):
    return [
        token.get("original") or token.get("corrected")
        for token in result["tokens"]
        if token.get("unrecognized") or token.get("force_unrecognized")
    ]


def _choice_words(result):
    return [
        choice["word"]
        for token in result["tokens"]
        for choice in token.get("choices", [])
    ]


def test_structural_suffix_repairs_precede_fuzzy_candidates():
    cases = {
        "jghamilha": "Jagħmilha.",
        "jbattalekx": "Jbattallekx.",
        "itijhom": "Itihom.",
        "jtijhom": "Jtihom.",
        "xtrajta": "Xtrajtha.",
        "rajtilom": "Rajtilhom.",
    }

    for original, expected in cases.items():
        assert app.spellchecker.correct_text_rich(original)["corrected_text"] == expected


def test_negative_ma_adds_x_only_without_negative_blocker():
    assert (
        app.spellchecker.correct_text_rich("ma tohrog il-verita")["corrected_text"]
        == "Ma toħroġx il-verità."
    )
    assert (
        app.spellchecker.correct_text_rich("qatt ma rajtilhom izjed")[
            "corrected_text"
        ]
        == "Qatt ma rajtilhom iżjed."
    )
    assert (
        app.spellchecker.correct_text_rich("ma insuqx")["corrected_text"]
        == "Ma nsuqx."
    )
    assert (
        app.spellchecker.correct_text_rich("ħadd ma johrog")["corrected_text"]
        == "Ħadd ma joħroġ."
    )


def test_function_word_phrases_do_not_receive_far_suggestions():
    for text in ("bhal din", "bhalom", "bhalhom", "izjed"):
        result = app.spellchecker.correct_text_rich(text)
        assert not _choice_words(result)
        assert not _unrecognized_words(result)


def test_adjacent_feminine_noun_selects_matching_adjective_form():
    result = app.spellchecker.correct_text_rich("ghalqa mitluq")

    assert result["corrected_text"] == "Għalqa mitluqa."
    assert not _choice_words(result)
    assert not _unrecognized_words(result)


def test_deterministic_suffix_choice_keeps_meanings():
    result = app.spellchecker.correct_text_rich("nuza")
    choices = {
        choice["word"]: choice["meaning"]
        for token in result["tokens"]
        for choice in token.get("choices", [])
    }

    assert choices == {
        "Nuża": "I use",
        "Nużaha": "I use her",
    }


def test_exact_english_and_recent_dictionary_words_are_recognized():
    result = app.spellchecker.correct_text_rich(
        "sales girl cashier grocery full time part time passatemp apparat"
    )

    assert not _unrecognized_words(result)


def test_name_repairs_are_light_and_do_not_rewrite_valid_words():
    result = app.spellchecker.correct_text_rich("Kieth Scembri wara")

    assert result["corrected_text"] == "Keith Schembri wara."
    assert not _unrecognized_words(result)


def test_exact_lowercase_place_phrase_restores_dictionary_capitalization():
    assert (
        app.spellchecker.correct_text_rich("lejn san pawl")["corrected_text"]
        == "Lejn San Pawl."
    )
    assert (
        app.spellchecker.correct_text_rich("san pawl il-bahar")["corrected_text"]
        == "San Pawl il-Baħar."
    )


def test_min_minn_choices_keep_meanings_on_function_word_only():
    result = app.spellchecker.correct_text_rich("min aw")
    choice_token = next(token for token in result["tokens"] if token.get("choices"))

    assert choice_token["original"] == "min"
    assert choice_token["corrected"] == "Min"
    assert choice_token["choices"] == [
        {"word": "Min", "meaning": "who"},
        {"word": "Minn", "meaning": "from"},
    ]


def test_ma_apostrophe_uses_name_noun_and_verb_context():
    cases = {
        "ma' Nicole": "Ma' Nicole.",
        "mort ma' Nicole": "Mort ma' Nicole.",
        "ma' Amy": "M'Amy.",
        "ma' membru": "Ma' membru.",
        "ma' marret": "Ma marritx.",
    }

    for original, expected in cases.items():
        assert app.spellchecker.correct_text_rich(original)["corrected_text"] == expected


def test_recent_exact_and_structural_regressions():
    assert (
        app.spellchecker.correct_text_rich("hajtek")["corrected_text"]
        == "Ħajtek."
    )
    assert (
        app.spellchecker.correct_text_rich("mid dinja")["corrected_text"]
        == "Mid-Dinja."
    )

    hu_choices = _choice_words(app.spellchecker.correct_text_rich("hu mar"))
    assert hu_choices == ["Hu", "Ħu", "U"]

    liktar_choices = _choice_words(app.spellchecker.correct_text_rich("liktar"))
    assert liktar_choices == ["L-iktar", "L'iktar", "'l-iktar"]


def test_recent_currency_usage_and_entity_preservation():
    assert (
        app.spellchecker.correct_text_rich("Resparmio Casa €1,99c u JB")[
            "corrected_text"
        ]
        == "Resparmio Casa €1.99 u JB."
    )
    assert (
        app.spellchecker.correct_text_rich("Mill JB")["corrected_text"]
        == "Mill-JB."
    )

    usage = app.spellchecker.correct_text_rich("tikkomplenja")
    usage_token = next(
        token for token in usage["tokens"] if token.get("type") == "maltese_usage"
    )
    assert usage["corrected_text"] == "Tikkomplenja."
    assert usage_token["maltese_suggestion"] == ["Tilmenta"]

    euro = app.spellchecker.correct_text_rich("Euro Euros")
    assert [
        token.get("maltese_suggestion")
        for token in euro["tokens"]
        if token.get("type") == "english_phrase"
    ] == [["Ewro"], ["Ewro"]]


def test_tagged_maltese_usage_verbs_follow_tense_and_person():
    cases = {
        "ddawnlowdja": ["Niżżel"],
        "ddawnlowdjat": ["Niżżlet"],
        "aplowdjajt": ["Tellajt"],
        "kkomplejnja": ["Ilmenta"],
        "kkomplejnjat": ["Ilmentat"],
        "kkomplenjat": ["Ilmentat"],
        "llegjat": ["Weħlet"],
        "jillegja": ["Jeħel"],
        "streccja": ["Tmattar"],
        "streċċjat": ["Tmattret"],
        "rreppjat": ["Geżwret"],
        "jixxerja": ["Jaqsam ma' ħaddieħor"],
        "xxerjat": ["Qasmet ma' ħaddieħor"],
        "xxerjaw": ["Qasmu ma' ħaddieħor", "Aqsmu ma' ħaddieħor"],
        "rrileksjat": ["Strieħet"],
        "jirrileksja": ["Jistrieħ"],
    }

    for source, expected in cases.items():
        result = app.spellchecker.correct_text_rich(source)
        token = next(
            item for item in result["tokens"] if item.get("type") == "maltese_usage"
        )
        assert token["maltese_suggestion"] == expected
        assert not token.get("unrecognized")


def test_cultural_names_places_and_kinship_context():
    assert (
        app.spellchecker.correct_text_rich("Ganni u ganni")["corrected_text"]
        == "Ġanni u għanni."
    )
    assert (
        app.spellchecker.correct_text_rich("Cina u cina")["corrected_text"]
        == "Ċina u Ċina."
    )
    assert (
        app.spellchecker.correct_text_rich("Mort Mdina u morna Mdina")[
            "corrected_text"
        ]
        == "Mort Imdina u morna Mdina."
    )
    assert (
        app.spellchecker.correct_text_rich("il Mdina")["corrected_text"]
        == "L-Imdina."
    )
    assert (
        app.spellchecker.correct_text_rich("hija ma jtikx")["corrected_text"]
        == "Ħija ma jtikx."
    )
    assert (
        app.spellchecker.correct_text_rich("hija ma tmurx")["corrected_text"]
        == "Hija ma tmurx."
    )


def test_tliet_and_ta_adverb_surface_choices():
    assert (
        app.spellchecker.correct_text_rich("tlieta tfal")["corrected_text"]
        == "Tlett itfal."
    )
    assert (
        app.spellchecker.correct_text_rich("tlieta baqriet")["corrected_text"]
        == "Tliet baqriet."
    )
    assert (
        app.spellchecker.correct_text_rich("tlett saqajn")["corrected_text"]
        == "Tliet saqajn."
    )
    result = app.spellchecker.correct_text_rich("ta imbagħad")
    assert result["corrected_text"] == "Ta' imbagħad."
    assert _choice_words(result) == ["Ta' imbagħad", "Ta imbagħad"]
