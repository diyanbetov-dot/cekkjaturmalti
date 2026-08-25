from corpus_ranker import CorpusCandidateRanker
from app import (
    ARTICLE_RESOLVER,
    CORPUS_RANKER,
    CORRECTOR,
    ENABLE_SUFFIXATION,
    MORPHOLOGY_RESOLVER,
    NUMERAL_RESOLVER,
    correct_text,
)


def test_article_dictionary_span_resolution_and_ambiguity():
    cases = {
        "id dar": "Id-dar.",
        "ghar rebha": "Għar-rebħa.",
        "fis skola": "Fis-skola.",
        "il Ġumhur": "Il-Ġumhur.",
        "mal hanut": "Mal-ħanut.",
        "dan ragel": "Dar-raġel.",
    }
    for source, expected in cases.items():
        corrected, _tokens = correct_text(source)
        assert corrected == expected

    corrected, tokens = correct_text("ghar rebha")
    prefix = next(token for token in tokens if token.get("original") == "ghar")
    assert prefix["article_phrase_corrected"] is True
    assert [choice["word"] for choice in prefix["choices"]] == ["għar-"]
    assert [choice["separator_after"] for choice in prefix["choices"]] == [""]

    corrected, tokens = correct_text("id dar")
    prefix = next(token for token in tokens if token.get("original") == "id")
    assert corrected == "Id-dar."
    assert [choice["word"] for choice in prefix["choices"]] == ["id-"]

    corrected, tokens = correct_text("id- dar")
    prefix = next(token for token in tokens if token.get("original") == "id")
    assert corrected == "Id-dar."
    assert [choice["word"] for choice in prefix["choices"]] == ["id-"]


def test_article_resolution_respects_assimilation_and_boundaries():
    cases = {
        "id rebha": "Id rebħa.",
        "ghar dar": "Għar dar.",
        "id, dar": "Id, dar.",
    }
    for source, expected in cases.items():
        corrected, _tokens = correct_text(source)
        assert corrected == expected

    assert ARTICLE_RESOLVER.status_payload()["forms"] == 131


def test_bahar_uses_dictionary_backed_h_pair():
    corrected, candidates, recognized = CORRECTOR.correct_word("bahar")
    assert recognized is True
    assert corrected == "baħar"
    assert candidates[0].word == "baħar"
    assert candidates[0].distance == 0.2


def test_exact_dictionary_word_is_never_rewritten():
    corrected, candidates, recognized = CORRECTOR.correct_word("baħar")
    assert (corrected, candidates, recognized) == ("baħar", [], True)


def test_unrelated_unknown_word_has_no_candidate():
    corrected, candidates, recognized = CORRECTOR.correct_word("xyzq")
    assert (corrected, candidates, recognized) == ("xyzq", [], False)


def test_i_ie_and_single_double_repairs_are_general_dictionary_paths():
    cases = {
        "bix": ("biex", "basedics_i_ie"),
        "mortt": ("mort", "basedics_single_double"),
    }
    for source, (expected, expected_source) in cases.items():
        corrected, candidates, recognized = CORRECTOR.correct_word(source)
        assert recognized is True
        assert corrected == expected
        assert candidates[0].source == expected_source


def test_general_structural_paths_do_not_invent_words():
    for source in ("xiexq", "mortttz"):
        corrected, candidates, recognized = CORRECTOR.correct_word(source)
        assert (corrected, candidates, recognized) == (source, [], False)


def test_final_sentence_formatting_preserves_suggestion_metadata():
    corrected, tokens = correct_text("bix qed tidhaq")
    words = [token for token in tokens if token["type"] == "word"]
    assert corrected == "Biex qed tidħaq."
    assert words[0]["original"] == "bix"
    assert words[0]["corrected"] == "Biex"
    assert [choice["word"] for choice in words[0]["choices"]] == ["biex", "bixx"]
    assert words[2]["original"] == "tidhaq"
    assert words[2]["corrected"] == "tidħaq"
    assert [choice["word"] for choice in words[2]["choices"]] == ["tidħaq"]


def test_corpus_indexes_are_available_and_surface_keyed():
    status = CORPUS_RANKER.status_payload()
    assert status["available"] is True
    assert status["unigrams"] > 30_000
    assert CORPUS_RANKER.evidence("biex").unigram > 0.0
    assert CORPUS_RANKER.evidence("bixx").unigram == 0.0


def test_corpus_evidence_annotates_without_removing_candidates():
    corrected, tokens = correct_text("bix qed tidhaq")
    first = next(token for token in tokens if token["type"] == "word")
    assert corrected == "Biex qed tidħaq."
    assert [choice["word"] for choice in first["choices"]] == ["biex", "bixx"]
    assert first["choices"][0]["corpus_score"] > first["choices"][1]["corpus_score"]


def test_corpus_dominance_hides_alternatives_at_eighty_percent():
    corrected, tokens = correct_text("Ara veru")
    ara = next(token for token in tokens if token.get("original") == "Ara")
    assert corrected == "Ara veru."
    assert [choice["word"] for choice in ara["choices"]] == ["Ara"]


def test_exact_english_is_attached_only_to_final_tokens():
    corrected, tokens = correct_text("il mobile ikun hawn")
    english = next(token for token in tokens if token.get("type") == "english_phrase")
    assert corrected == "Il-mobile jkun hawn."
    assert english["corrected"] == "mobile"
    assert english["maltese_suggestion"] == ["mowbajl"]
    assert english["unrecognized"] is False


def test_fused_function_words_are_resolved_structurally():
    assert correct_text("jien manafx xihaga")[0] == "Jien ma nafx xi ħaġa."
    assert correct_text("ma nafx xini trid")[0] == "Ma nafx x'inhi trid."
    assert correct_text("xini")[0] == "xini"


def test_attested_context_can_rerank_equal_distance_candidates():
    corrected, tokens = correct_text("kien amel hekk")
    amel = next(token for token in tokens if token.get("original") == "amel")
    assert corrected == "Kien għamel hekk."
    assert [choice["word"] for choice in amel["choices"]] == ["għamel", "agħmel"]
    assert amel["corpus_reordered"] is True
    assert amel["choices"][0]["corpus_left_bigram"] > 0.0


def test_corpus_can_resolve_manual_kif_kief_context():
    corrected, tokens = correct_text("hekk kif baqa")
    kif = next(token for token in tokens if token.get("original") == "kif")
    assert corrected == "Hekk kif baqa'."
    assert kif["corrected"] == "kif"


def test_article_and_corpus_paths_cover_sample_sentence():
    source = (
        "hekk kif baqa tiela Trejqet il Ġumhur, dar mal hanut tal gagagi "
        "mistici, hekk kif baqa sejer dritt fi sqaq maluqa."
    )
    corrected, _tokens = correct_text(source)
    assert corrected == (
        "Hekk kif baqa' tiela Trejqet il-Ġumhur, dar mal-ħanut tal-ġagagi "
        "mistiċi, hekk kif baqa' sejer dritt fi sqaq magħluqa."
    )


def test_disabled_corpus_ranker_is_neutral():
    ranker = CorpusCandidateRanker(CORPUS_RANKER.corpus_dir, enabled=False)
    assert ranker.available is False
    assert ranker.evidence("biex", following="qed").score == 0.0


def test_dictionary_and_corpus_morphology_correct_noun_adjective_agreement():
    cases = {
        "tifla kbir": "Tifla kbira.",
        "tifel kbira": "Tifel kbir.",
        "tfajliet kbir": "Tfajliet kbar.",
        "karozza sabih": "Karozza sabiħa.",
    }
    for source, expected in cases.items():
        corrected, tokens = correct_text(source)
        assert corrected == expected
        adjective = [token for token in tokens if token["type"] == "word"][-1]
        assert adjective["agreement_corrected"] is True
        assert adjective["choices"][0]["source"] == "dictionary_corpus_morphology_agreement"
        assert [choice["word"] for choice in adjective["choices"]] == [
            expected.rstrip(".").split()[-1].casefold()
        ]


def test_agreement_preserves_clean_and_nonadjacent_phrases():
    cases = {
        "tifla kbira": "Tifla kbira.",
        "tifla, kbir": "Tifla, kbir.",
        "tifla hafna": "Tifla ħafna.",
    }
    for source, expected in cases.items():
        corrected, _tokens = correct_text(source)
        assert corrected == expected


def test_morphology_index_combines_corpus_lemma_with_dictionary_tags():
    assert MORPHOLOGY_RESOLVER.available is True
    candidates = MORPHOLOGY_RESOLVER.agreement_candidates("tifla", "kbir")
    assert candidates[0].word == "kbira"
    assert candidates[0].lemma == "kbir"
    assert candidates[0].noun_tag == "SINGNOUNF"
    assert candidates[0].adjective_tag == "SINGADJF"


def test_attributive_numeral_normalizes_all_short_long_ikk_combinations():
    cases = {
        "tliet tfal": "Tlett itfal.",
        "tliet itfal": "Tlett itfal.",
        "tlett tfal": "Tlett itfal.",
        "tlett itfal": "Tlett itfal.",
        "sitt tfal": "Sitt itfal.",
        "żewġ tfal": "Żewġt itfal.",
        "ġiex tfal": "Ġixt itfal.",
        "tliet jiem": "Tlett ijiem.",
        "tliet ijiem": "Tlett ijiem.",
        "tlett jiem": "Tlett ijiem.",
        "żewġ jiem": "Żewġt ijiem.",
    }
    for source, expected in cases.items():
        corrected, _tokens = correct_text(source)
        assert corrected == expected


def test_attributive_numeral_uses_tags_and_respects_boundaries():
    cases = {
        "mitt tfal": "Mitt tfal.",
        "tliet baqra": "Tliet baqra.",
        "tliet, tfal": "Tliet, tfal.",
        "itfal": "itfal",
        "jiem": "jiem",
    }
    for source, expected in cases.items():
        corrected, _tokens = correct_text(source)
        assert corrected == expected


def test_attributive_numeral_carries_dictionary_and_corpus_evidence():
    corrected, tokens = correct_text("tliet tfal")
    words = [token for token in tokens if token["type"] == "word"]
    assert corrected == "Tlett itfal."
    assert NUMERAL_RESOLVER.status_payload()["numeral_pairs"] == 10
    assert words[0]["choices"][0]["numeral_rule"] == "LONGATTNUM+iKK"
    assert words[1]["choices"][0]["noun_tag"] == "PLUNOUN"
    assert words[1]["choices"][0]["corpus_left_bigram"] > 0.0
    assert [choice["word"] for choice in words[0]["choices"]] == ["tlett"]
    assert [choice["word"] for choice in words[1]["choices"]] == ["itfal"]


def test_manual_dictionary_is_not_used():
    corrected, candidates, recognized = CORRECTOR.correct_word("tijaj")
    assert (corrected, candidates, recognized) == ("tijaj", [], False)


def test_punctuation_and_spacing_are_preserved():
    corrected, _ = correct_text("mort il bahar!")
    assert corrected == "mort il baħar!"


def test_lowercase_input_prefers_lowercase_dictionary_surface():
    corrected, candidates, recognized = CORRECTOR.correct_word("hadd")
    assert recognized is True
    assert corrected == "ħadd"
    assert candidates[0].word == "ħadd"


def test_final_surface_examples():
    cases = {
        "bahar": "baħar",
        "mort bahar": "Mort baħar.",
        "BAHAR": "BAĦAR",
        "mort      bahar": "Mort baħar.",
        "mort bahar   .": "mort baħar.",
        "mort baħar ..": "mort baħar.",
        "mort baħar ...": "mort baħar...",
    }
    for source, expected in cases.items():
        corrected, _ = correct_text(source)
        assert corrected == expected


def test_question_and_exclamation_terminal_rules():
    cases = {
        "mort bahar ?": "mort baħar?",
        "mort bahar ??": "mort baħar??",
        "mort bahar ???": "mort baħar?",
        "mort bahar !": "mort baħar!",
        "mort bahar !!": "mort baħar!!",
        "mort bahar !!!": "mort baħar!",
    }
    for source, expected in cases.items():
        corrected, _ = correct_text(source)
        assert corrected == expected


def test_missing_gh_h_skeleton_fallbacks():
    cases = {
        "loba": "logħba",
        "jider": "jidher",
        "fem": "fehem",
        "bat": "bagħat",
        "aghsafar": "għasafar",
    }
    for source, expected in cases.items():
        corrected, candidates, recognized = CORRECTOR.correct_word(source)
        assert recognized is True
        assert corrected == expected
        assert candidates[0].source == "basedics_missing_gh_h_skeleton"


def test_skeleton_fallback_does_not_invent_unindexed_words():
    corrected, candidates, recognized = CORRECTOR.correct_word("qxvz")
    assert (corrected, candidates, recognized) == ("qxvz", [], False)


def test_suffixation_is_enabled_for_testing():
    assert ENABLE_SUFFIXATION is True
    assert CORRECTOR.suffix_engine is not None


def test_generated_suffix_forms_are_recognized():
    for word in ("għamilhom", "għamilhulhom", "għamluhom"):
        corrected, candidates, recognized = CORRECTOR.correct_word(word)
        assert recognized is True
        assert corrected == word
        assert candidates == []


def test_ascii_gh_normalization_reaches_suffixed_verb_records():
    cases = {
        "aghtihieli": "agħtihieli",
        "aghtihomli": "agħtihomli",
        "atiha": "agħtiha",
    }
    for source, expected in cases.items():
        corrected, _candidates, recognized = CORRECTOR.correct_word(source)
        assert recognized is True
        assert corrected == expected


def test_required_ec_to_ic_suppresses_untransformed_suffix_candidate():
    corrected, candidates, recognized = CORRECTOR.correct_word("amilhulu")
    assert recognized is True
    assert corrected == "għamilhulu"
    assert "għamelhulu" not in {candidate.word for candidate in candidates}


def test_suffix_inverse_lookup_supports_i_ie_and_single_double_variants():
    cases = {
        "nesejtni": "nessejtni",
        "għamielhulu": "għamilhulu",
    }
    for source, expected in cases.items():
        corrected, _candidates, recognized = CORRECTOR.correct_word(source)
        assert recognized is True
        assert corrected == expected

    adapter = CORRECTOR.suffix_engine.adapter
    assert "mil" in adapter.i_ie_variants("miel")
    assert "miel" in adapter.i_ie_variants("mil")
    assert "ness" in adapter.single_double_variants("nes")
    assert "nes" in adapter.single_double_variants("ness")


def test_bounded_hybrid_repairs_compose_existing_transformation_families():
    cases = {
        "arwenien": "għarwenin",
        "aghtihili": "agħtihieli",
    }
    for source, expected in cases.items():
        corrected, candidates, recognized = CORRECTOR.correct_word(source)
        assert recognized is True
        assert corrected == expected
        assert candidates[0].source.startswith("hybrid_")


def test_ascii_gh_suffix_form_uses_suffix_suggestions():
    corrected, candidates, recognized = CORRECTOR.correct_word("ghamilhom")
    assert recognized is True
    assert corrected == "għamilhom"
    assert candidates[0].source == "basedics_suffix_generator"


def test_exact_suffix_surface_remains_an_alternative_to_stronger_diacritic_mapping():
    corrected, candidates, recognized = CORRECTOR.correct_word("nsieh")
    assert recognized is True
    assert corrected == "nsieħ"
    assert [candidate.word for candidate in candidates[:2]] == ["nsieħ", "nsieh"]
    assert candidates[1].source == "basedics_suffix_exact_alternative"


def test_final_weak_direct_object_families_are_kept_distinct():
    cases = {
        "ferah": "feraħ",
        "ferieh": "ferieh",
        "feraha": "ferieha",
        "qraħ": "qrah",
        "qarah": "qaraħ",
        "darah": "drah",
        "fagah": "fgah",
        "nesieh": "nsieh",
        "bedieh": "bdieh",
    }
    for source, expected in cases.items():
        corrected, _candidates, recognized = CORRECTOR.correct_word(source)
        assert recognized is True
        assert corrected == expected


def test_nesietu_retains_both_valid_single_double_analyses():
    corrected, candidates, recognized = CORRECTOR.correct_word("nesietu")
    surfaces = {corrected, *(candidate.word for candidate in candidates)}
    assert recognized is True
    assert corrected in {"nessietu", "nsietu"}
    assert {"nessietu", "nsietu"}.issubset(surfaces)
    assert "nessitu" not in surfaces


def test_qara_weak_suffix_is_an_alternative_to_valid_qara_h_verb():
    corrected, candidates, recognized = CORRECTOR.correct_word("qarah")
    assert recognized is True
    assert corrected == "qaraħ"
    assert [candidate.word for candidate in candidates[:2]] == ["qaraħ", "qrah"]


def test_non_f1_final_weak_default_add_is_not_generated():
    matches = CORRECTOR.suffix_engine.generator.exact_suffix_matches("ferah")
    assert matches == []


def test_fused_preposition_recovery_accepts_possessive_complements():
    cases = {
        "bidek": "b'idek",
        "bmohhok": "b'moħħok",
        "xandek": "x'għandek",
    }
    for source, expected in cases.items():
        corrected, _candidates, recognized = CORRECTOR.correct_word(source)
        assert recognized is True
        assert corrected == expected


def test_literal_dictionary_word_is_not_split_as_a_fused_preposition():
    corrected, candidates, recognized = CORRECTOR.correct_word("bint")
    assert recognized is True
    assert corrected == "bint"
    assert "b'int" not in {candidate.word for candidate in candidates}
