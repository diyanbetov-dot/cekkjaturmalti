# -*- coding: utf-8 -*-

from Essentials.app import app, corpus_scorer, spellchecker


def _check(text: str) -> dict:
    response = app.test_client().post("/check-text", json={"text": text})
    assert response.status_code == 200
    return response.get_json()


def _token(result: dict, original: str) -> dict:
    return next(
        token
        for token in result["tokens"]
        if token.get("original") == original
    )


def test_live_corpus_indexes_are_available_and_well_formed():
    assert corpus_scorer.is_available()
    assert corpus_scorer.status == "CORPUS_SCORER_READY"
    assert "kif" in corpus_scorer.unigrams
    assert not any("\t" in key for key in list(corpus_scorer.unigrams)[:2000])


def test_corpus_resolves_kief_from_recurrent_left_and_right_context():
    result = _check("Ma nafx kief ghamel hekk")
    token = _token(result, "kief")

    assert result["corrected_text"] == "Ma nafx kif għamel hekk."
    assert token["corpus_context"]["winner"] == "kif"
    assert token["corpus_context"]["accepted"] is True
    assert token["corpus_context"]["scores"]["kif"]["left_bigram"] > 0.0


def test_morphology_blocks_frequency_trap_after_mur():
    result = _check("mur amel")
    token = _token(result, "amel")

    assert result["corrected_text"] == "Mur agħmel."
    assert token["corpus_context"]["hard_reason"] == "serial_imperative"


def test_article_surface_is_scored_as_corpus_token_sequence():
    result = _check("Int liktar wiehed ikrah")
    token = _token(result, "liktar")

    assert result["corrected_text"] == "Int l-iktar wieħed ikrah."
    assert token["corpus_context"]["scores"]["l-iktar"]["internal_bigram"] > 0.0


def test_non_family_l_no_longer_depends_on_ta_or_ha_verb_context():
    result = _check("mur tieh liktar wiehed ta lil hemm")
    token = _token(result, "liktar")

    assert result["corrected_text"].startswith("Mur tieh l-iktar wieħed")
    assert token["corpus_context"]["winner"] == "l-iktar"


def test_ambiguous_family_term_uses_lil_and_keeps_three_surfaces():
    result = _check("tkellimx lommi")
    token = _token(result, "lommi")

    assert result["corrected_text"] == "Tkellimx 'l ommi."
    assert [choice["word"] for choice in token["choices"]] == [
        "'l ommi",
        "l-ommi",
        "'l-ommi",
    ]
    assert token["corpus_context"]["hard_reason"] == "family_term_requires_lil"


def test_family_term_rule_supports_il_input_and_possessive_base():
    result = _check("ghid il missieri")
    token = _token(result, "il missieri")

    assert result["corrected_text"] == "Għid 'il missieri."
    assert [choice["word"] for choice in token["choices"]] == [
        "'il missieri",
        "il-missieri",
        "'il-missieri",
    ]


def test_base_family_term_before_ta_prefers_lill_hyphen_surface():
    result = _check("cempel lomm ta wiehed mit tfal")
    token = _token(result, "lomm")

    assert result["corrected_text"] == (
        "Ċempel 'l-omm ta' wieħed mit-tfal."
    )
    assert [choice["word"] for choice in token["choices"]] == [
        "'l-omm",
        "l-omm",
        "'l omm",
    ]


def test_explicit_family_article_without_ta_remains_an_article():
    result = _check("l-omm marret")
    assert result["corrected_text"] == "L-omm marret."


def test_family_possessive_is_corrected_before_lil_surface_selection():
    result = _check("rajt l-ohti")
    token = _token(result, "l-ohti")

    assert result["corrected_text"] == "Rajt 'l oħti."
    assert [choice["word"] for choice in token["choices"]] == [
        "'l oħti",
        "l-oħti",
        "'l-oħti",
    ]


def test_selector_can_be_disabled_without_removing_the_pipeline():
    selector = spellchecker.corpus_context_selector
    previous_mode = selector.mode
    try:
        selector.mode = "off"
        result = _check("Ma nafx kief ghamel hekk")
        assert result["corrected_text"] == "Ma nafx kief għamel hekk."
        assert not any(
            token.get("corpus_context")
            for token in result["tokens"]
            if isinstance(token, dict)
        )
    finally:
        selector.mode = previous_mode
