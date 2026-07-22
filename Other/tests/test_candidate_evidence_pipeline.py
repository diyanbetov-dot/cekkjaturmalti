# -*- coding: utf-8 -*-
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Essentials"))

import app


def test_candidate_evidence_debug_has_rankable_metadata():
    report = app.spellchecker.candidate_evidence_debug("namel", limit=5)

    assert report["normalized"] == "namel"
    assert report["candidates"]
    first = report["candidates"][0]
    assert "sources" in first
    assert "types" in first
    assert "reasons" in first
    assert "confidence" in first


def test_manual_repairs_precede_near_english_protection():
    result = app.spellchecker.correct_text_rich("aw awn tranport transport")

    assert result["corrected_text"] == "Hawn hawn transport transport."
    assert not [
        token
        for token in result["tokens"]
        if token.get("force_unrecognized") or token.get("unrecognized")
    ]


def test_min_minn_suggestions_attach_to_function_word_only():
    result = app.spellchecker.correct_text_rich("min aw")

    assert result["corrected_text"] == "Min hawn."
    choice_tokens = [token for token in result["tokens"] if token.get("choices")]
    assert choice_tokens
    assert choice_tokens[0]["original"] == "min"
    assert choice_tokens[0]["corrected"] == "Min"


def test_z_owned_phrases_win_before_generic_article_paths():
    cases = {
        "ma tantx": "Ma tantx.",
        "min hu": "Min hu.",
        "min kien min fejn": "Min kien min fejn.",
        "ma membru": "Ma' membru.",
        "ma' membru": "Ma' membru.",
    }

    for original, expected in cases.items():
        assert app.spellchecker.correct_text_rich(original)["corrected_text"] == expected


def test_apostrophe_compounds_stay_recognized_when_tail_is_valid():
    result = app.spellchecker.correct_text_rich("B'dak f'keffa b'kilometru")

    assert result["corrected_text"] == "B'dak f'keffa b'kilometru."
    assert not [token for token in result["tokens"] if token.get("unrecognized")]


def test_exact_place_and_article_spans_are_not_over_rewritten():
    cases = {
        "malta": "Malta.",
        "l-karozza": "L-karozza.",
        "5 huwa l-ghola grad": "5 huwa l-għola grad.",
        "bieba ta wara naha": "Bieba ta' wara naħa.",
    }

    for original, expected in cases.items():
        assert app.spellchecker.correct_text_rich(original)["corrected_text"] == expected


def test_structural_non_fuzzy_repairs_remain_available():
    cases = {
        "sperjenzajt": "Esperjenzajt.",
        "jorukhom": "Jurukom.",
        "importata": "Importata.",
    }

    for original, expected in cases.items():
        assert app.spellchecker.correct_text_rich(original)["corrected_text"] == expected


def test_preposition_article_variants_are_generated_generally():
    cases = {
        "bhal l siehbi": "Bħas-sieħbi.",
        "bhal- siehbi": "Bħas-sieħbi.",
        "bhall sihbi": "Bħal sieħbi.",
        "ghal l gid": "Għall-ġid.",
        "ghadd dinja": "Għad-Dinja.",
        "mis s-supermarket": "Mis-supermarket.",
    }

    for original, expected in cases.items():
        assert app.spellchecker.correct_text_rich(original)["corrected_text"] == expected


def test_structural_candidates_feed_evidence_pipeline():
    assert "xahar" in app.spellchecker.suggest("xar", limit=8)

    ntbat_suggestions = app.spellchecker.suggest("ntbat", limit=8)
    assert "ntbagħt" in ntbat_suggestions
    assert "ntbagħat" in ntbat_suggestions

    assert app.spellchecker.suggest("jitfahha", limit=5)[0] == "jitfagħha"
    assert app.spellchecker.suggest("jamila", limit=5)[0] == "jagħmilha"


def test_function_word_z_decisions_are_centralized():
    cases = {
        "ma membru": "Ma' membru.",
        "ma hawn": "m'hawn.",
        "ma għandu": "m'għandu.",
        "ma niflahx": "Ma niflaħx.",
        "ta quddiem": "Ta' quddiem.",
        "ta kemm": "Ta' kemm.",
    }
    for original, expected in cases.items():
        assert app.spellchecker.correct_text_rich(original)["corrected_text"] == expected

    min_result = app.spellchecker.correct_text_rich("min aw")
    assert min_result["corrected_text"] == "Min hawn."
    choice_tokens = [token for token in min_result["tokens"] if token.get("choices")]
    assert choice_tokens[0]["original"] == "min"
    assert choice_tokens[0]["corrected"] == "Min"
