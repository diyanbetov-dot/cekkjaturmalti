from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Essentials import app


client = app.app.test_client()


def check(source: str) -> dict:
    response = client.post(
        "/check-text",
        json={
            "text": source,
            "edit_distance_tolerance": 1,
            "include_grammar": True,
        },
    )
    assert response.status_code == 200
    return response.get_json()


def lexical_tokens(result: dict) -> list[dict]:
    return [
        token
        for token in result["tokens"]
        if token.get("type") in {"word", "phrase"}
    ]


def choice_words(result: dict) -> list[str]:
    return [
        choice["word"]
        for token in lexical_tokens(result)
        for choice in token.get("choices", [])
    ]


def test_literal_article_choices_require_governing_ta_or_ha() -> None:
    ordinary = check("mar il bajja")
    assert ordinary["corrected_text"] == "Mar il-bajja."
    assert all(not word.startswith("'") for word in choice_words(ordinary))

    split = check("mar l bajja")
    assert split["corrected_text"] == "Mar l-bajja."
    assert all(not word.startswith("'") for word in choice_words(split))

    assert check("mar r Rabat")["corrected_text"] == "Mar ir-Rabat."
    assert check("mar d dar")["corrected_text"] == "Mar id-dar."
    assert check("mar s supermarket")["corrected_text"] == (
        "Mar is-supermarket."
    )

    governed = check("ta l kaxxiera")
    assert governed["corrected_text"] == "Tal-kaxxiera."
    assert {"ta 'l kaxxiera", "ta' l-kaxxiera"} & set(choice_words(governed))


def test_suffix_bearing_verbs_are_not_reconjugated_by_surface_ambiguity() -> None:
    assert check("ghogbu jikkuntattjani")["corrected_text"] == (
        "Għoġbu jikkuntattjani."
    )
    assert check("basta tfejqu")["corrected_text"] == "Basta tfejqu."


def test_existing_productive_grammar_rewrite_still_applies() -> None:
    assert check("il mara marret jixtri")["corrected_text"] == (
        "Il-mara marret tixtri."
    )


def test_article_surface_repairs_do_not_require_a_known_tail() -> None:
    assert check("tat traffiku")["corrected_text"] == "Tat-traffiku."
    assert check("mill l isptar")["corrected_text"] == "Mill-isptar."


def test_priority_repairs_from_the_two_regression_texts() -> None:
    assert check("jaghmilha dritta")["corrected_text"] == "Jagħmilha dritta."
    assert check("qedha l bajja")["corrected_text"] == "Qiegħda l-bajja."
    assert check("baqa sfigurat")["corrected_text"] == "Baqa' sfigurat."
    assert check("kulljumm nirringrazzja")["corrected_text"] == (
        "Kuljum nirringrazzja."
    )
    assert check("minn fejn")["corrected_text"] == "Minn fejn."


def test_article_repairs_preserve_explicit_name_case() -> None:
    assert check("l Alla")["corrected_text"] == "L-Alla."
    assert check("l alla")["corrected_text"] == "L-alla."


def test_no_far_apostrophe_choice_for_lilek() -> None:
    assert "l'ħiellek" not in choice_words(check("ghandi lilek"))


def test_possessive_noun_wins_over_later_pattern_guess() -> None:
    assert check("gismu")["corrected_text"] == "\u0120ismu."


def test_suffix_repair_composes_only_to_generated_verb_surface() -> None:
    result = check("illibsuwha")
    assert result["corrected_text"] == "Ilibsuha."
    assert not any(token.get("unrecognized") for token in lexical_tokens(result))


def test_blank_line_closes_each_prose_paragraph() -> None:
    assert check("Grazzi\n\nGrazzi!!")["corrected_text"] == (
        "Grazzi.\n\nGrazzi!!"
    )
