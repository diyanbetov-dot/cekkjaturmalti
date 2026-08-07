from __future__ import annotations

from neural_corrector.inference.edits import structured_edits
from neural_corrector.web.app import is_bare_word_input, tokens_from_result


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
