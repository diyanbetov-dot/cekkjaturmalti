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
