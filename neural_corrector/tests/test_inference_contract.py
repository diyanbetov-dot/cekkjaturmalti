from __future__ import annotations

from pathlib import Path

from neural_corrector.inference.dictionary_index import DictionaryIndex
from neural_corrector.inference.corrector import NeuralCorrector
from neural_corrector.inference.edits import structured_edits
from neural_corrector.web.app import is_bare_word_input, tokens_from_result

ROOT = Path(__file__).resolve().parents[2]
INDEX_PATH = (
    ROOT / "neural_corrector/data/indexes/maltese_dictionary.sqlite3"
)


def test_structured_edits_include_ui_contract_fields() -> None:
    edits = structured_edits(
        "amel",
        "għamel",
        [0.96, 0.97, 0.98, 0.99],
        lambda start, end, replacement, original: [
            replacement,
            "agħmel",
            original,
        ],
    )
    assert edits
    edit = edits[0]
    assert {
        "original",
        "replacement",
        "alternatives",
        "type",
        "confidence",
        "start",
        "end",
        "corrected_start",
        "corrected_end",
        "explanation",
    } <= set(edit)
    assert edit["alternatives"][0] == edit["replacement"]


def test_alternative_ui_route_is_limited_to_a_bare_word() -> None:
    edit = {
        "original": "amel",
        "replacement": "għamel",
        "alternatives": ["għamel", "agħmel"],
        "type": "spelling",
        "confidence": 0.97,
        "start": 0,
        "end": 4,
        "corrected_start": 0,
        "corrected_end": 5,
        "explanation": "Neural spelling correction",
    }
    bare_tokens = tokens_from_result(
        {
            "original_text": "amel",
            "corrected_text": "għamel",
            "edits": [edit],
            "sequence_alternatives": ["għamel", "agħmel"],
        }
    )
    contextual_tokens = tokens_from_result(
        {
            "original_text": "mur amel",
            "corrected_text": "mur għamel",
            "edits": [
                {
                    **edit,
                    "start": 4,
                    "end": 8,
                    "corrected_start": 4,
                    "corrected_end": 9,
                }
            ],
            "sequence_alternatives": [],
        }
    )
    assert bare_tokens[0]["type"] == "word"
    assert bare_tokens[0]["choices"][1]["word"] == "agħmel"
    assert all(token["type"] == "text" for token in contextual_tokens)
    assert is_bare_word_input("għamel")
    assert not is_bare_word_input("mur għamel")


def test_dictionary_index_preserves_valid_input_and_rejects_invented_word() -> None:
    index = DictionaryIndex(INDEX_PATH)
    assert index.contains_surface_form("tazez")
    assert index.contains_surface_form("nagħmel")
    assert not index.contains_surface_form("tażeż")
    guarded, decisions = index.guard_text("tazez", "Tażeż")
    assert guarded == "tazez"
    assert decisions[0]["decision"] == "reject_unknown_candidate"
    assert decisions[0]["source_is_dictionary_word"] is True


def test_dictionary_guard_keeps_valid_neural_edits_and_reverts_invalid_ones() -> None:
    index = DictionaryIndex(INDEX_PATH)
    guarded, decisions = index.guard_text(
        "mort namel ikel", "Mort nagħmel ikel."
    )
    assert guarded == "Mort nagħmel ikel."
    assert any(
        decision["decision"] == "accept_dictionary_candidate"
        and decision["candidate"] == "nagħmel"
        for decision in decisions
    )
    guarded, decisions = index.guard_text(
        "hafna snin ilu", "hafn snin ilu."
    )
    assert guarded == "hafna snin ilu."
    assert decisions[0]["decision"] == "reject_unknown_candidate"


def test_guarded_bare_word_does_not_return_an_identical_suggestion() -> None:
    corrector = NeuralCorrector(
        ROOT / "neural_corrector/artifacts/char_edit_bigru_v2"
    )
    result = corrector.correct("tazez")
    assert result["corrected_text"] == "tazez"
    assert result["changed"] is False
    assert result["sequence_alternatives"] == []
    assert result["dictionary_validation"]["decisions"][0]["decision"] == (
        "reject_unknown_candidate"
    )


def test_dictionary_rescue_recovers_high_confidence_valid_word() -> None:
    corrector = NeuralCorrector(
        ROOT / "neural_corrector/artifacts/char_edit_bigru_v2"
    )
    result = corrector.correct("amilt erba tazez")
    assert result["corrected_text"] == "Għamilt erba tazez"
    assert any(
        decision["decision"] == "accept_dictionary_rescue"
        and decision["candidate"] == "Għamilt"
        for decision in result["dictionary_validation"]["decisions"]
    )


def test_plural_noun_selects_short_attributive_number_candidate() -> None:
    corrector = NeuralCorrector(
        ROOT / "neural_corrector/artifacts/char_edit_bigru_v2"
    )
    result = corrector.correct("erbgha tazez")
    assert result["corrected_text"] == "erba' tazez"
    assert any(
        decision["decision"] == "accept_contextual_number_candidate"
        and decision["candidate"] == "erba'"
        for decision in result["dictionary_validation"]["decisions"]
    )


def test_low_confidence_prefix_candidate_is_not_rescued() -> None:
    corrector = NeuralCorrector(
        ROOT / "neural_corrector/artifacts/char_edit_bigru_v2"
    )
    result = corrector.correct("mort namel erba tazez")
    assert result["corrected_text"] == "Mort nagħmel erba' tazez"
    assert "x'nagħmel" not in result["corrected_text"]


def test_generated_suffix_validation_and_specialist_ranking() -> None:
    corrector = NeuralCorrector(
        ROOT / "neural_corrector/artifacts/char_edit_bigru_v2"
    )
    cases = {
        "jamila": "jagħmilha",
        "namillek": "nagħmillek",
        "nitfalom": "nitfagħlhom",
        "ghidtlu": "għedtlu",
        "jiehduna": "jeħduna",
        "hallieha": "Ħallieha",
        "jibatu": "jibagħtu",
        "ixejjrula": "jxejrulha",
        "ghamililhom": "għamilhielhom",
        "inhalasom": "inħallashom",
        "jircevuhom": "jirċivuhom",
        "rnexxielha": "rnexxielha",
    }
    for source, expected in cases.items():
        result = corrector.correct(source)
        assert result["corrected_text"].removesuffix(".") == expected
