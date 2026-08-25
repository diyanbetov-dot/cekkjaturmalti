from app import correct_text


def _word_tokens(tokens):
    return [token for token in tokens if token.get("type") == "word"]


def test_dictionary_compound_and_lil_article_resolution():
    assert correct_text("bil lejl")[0] == "Billejl."
    assert correct_text("cempel lil mara")[0] == "Ċempel lill-mara."
    assert correct_text("cempel lil John")[0] == "Ċempel lil John."
    assert correct_text("cempel lil ommu")[0] == "Ċempel lil ommu."


def test_context_combines_apostrophe_and_diacritic_candidates():
    assert correct_text("Qaltlu x gara dot?")[0] == "Qaltlu x'ġara dot?"


def test_context_selects_imperative_and_reporting_suffix():
    assert correct_text("Qallu hu dal vitamin")[0] == "Qallu ħu dal-vitamin."
    assert correct_text("Qalla ghax ma setax")[0] == "Qalilha għax ma setax."


def test_apostrophe_vowel_and_gh_root_epenthesis():
    assert correct_text("qieghed jerga jbul")[0] == "Qiegħed jerġa' jbul."
    assert correct_text("qieghed jghid")[0] == "Qiegħed jgħid."


def test_ghej_contracts_before_suffix_validation():
    assert correct_text("tghejdlix")[0] == "tgħidlix"


def test_contextual_missing_consonant_does_not_become_global_fuzzy_repair():
    assert correct_text("tid tkun taf")[0] == "Trid tkun taf."
    assert correct_text("tid")[0] != "tbid"


def test_distant_skeletal_suggestions_are_removed():
    _text, tokens = correct_text("ma nafx ghala. Kif hareg cempell.")
    by_original = {token.get("original"): token for token in _word_tokens(tokens)}
    assert [choice["word"] for choice in by_original["ghala"]["choices"]] == ["għala"]
    assert [choice["word"] for choice in by_original["cempell"]["choices"]] == ["ċempel"]


def test_fused_x_ha_and_suffix_validated_diacritics():
    output, _ = correct_text("Qazzistni u ma nafx xha naqbad nagħmel iktar.")
    assert output == "Qażżiżtni u ma nafx x'ħa naqbad nagħmel iktar."


def test_explicit_go_and_valid_l_article_surfaces():
    output, tokens = correct_text(
        "L-oħrajn jgħidu l-istess ħaġa u daqsek ħa mmur ġo dicca."
    )
    assert output == "L-oħrajn jgħidu l-istess ħaġa u daqshekk ħa mmur ġo diċċa."
    choices_by_original = {
        str(token.get("original", "")).casefold(): [
            str(choice.get("word", "")).casefold()
            for choice in token.get("choices", [])
        ]
        for token in tokens
        if token.get("type") == "word"
    }
    assert choices_by_original["ġo"] == []
    assert choices_by_original["l-oħrajn"] == []
    assert choices_by_original["l-istess"] == []
    assert "ġod-" not in choices_by_original["ġo"]
    assert "l-stess" not in choices_by_original["l-istess"]
    assert "l'stess" not in choices_by_original["l-istess"]


def test_gh_root_vowel_recovery_and_terminal_apostrophe_skeleton():
    assert correct_text("jixghal")[0] == "jixgħel"
    assert correct_text("jixal")[0] == "jixgħel"
    assert correct_text("tixghal")[0] == "tixgħel"
    assert correct_text("tixal")[0] == "tixgħel"
    output, _ = correct_text("jkun jistgha jirrangah")
    assert output == "Ikun jista' jirranġah."


def test_proper_name_blocks_bare_preposition_article_and_demonstrative_double_parse():
    output, tokens = correct_text("ghal Samsung tv. Dan it tv intefa")
    assert output == "Għal Samsung tv. Dan it-tv intefa."
    by_original = {token.get("original"): token for token in _word_tokens(tokens)}
    assert "għas-" not in [choice["word"] for choice in by_original["ghal"]["choices"]]
    assert by_original["Dan"]["corrected"] == "Dan"


def test_filtered_candidate_cannot_remain_as_output():
    output, tokens = correct_text("ghax mux tv zghir")
    assert output == "Għax mhux tv żgħir."
    mux = next(token for token in _word_tokens(tokens) if token.get("original") == "mux")
    assert mux["corrected"] == "mhux"
    assert "għmux" not in [choice["word"] for choice in mux["choices"]]


def test_detached_assimilated_article_and_contextual_hh_to_gh_h():
    output, tokens = correct_text("u d dinja kollha maħħa")
    assert output == "U d-dinja kollha magħha."
    by_original = {token.get("original"): token for token in _word_tokens(tokens)}
    assert by_original["d"]["corrected"] == "d-"
    assert by_original["maħħa"]["corrected"] == "magħha"

    # The written form is also a real verb and must remain available without
    # contextual evidence for the prepositional reading.
    assert correct_text("maħħa")[0] == "maħħa"


def test_numeral_phrase_normalizes_noun_number_without_breaking_i_form():
    assert correct_text("tini erba birra")[0] == "Tini erba' birer."
    assert correct_text("tliet tfal")[0] == "Tlett itfal."


def test_latest_contextual_regression_passage():
    source = (
        "manafx ux hekk ta. nahseb jien qed thawwad langli u d dinja kolla "
        "maħħa. mur ifem xi problemi taqla. it tielet grad dejjem ola mir "
        "raba. tini erba birra"
    )
    expected = (
        "Ma nafx hux hekk ta. Naħseb jien qed tħawwad l-anġli u d-dinja "
        "kollha magħha. Mur ifhem xi problemi taqla'. It-tielet grad dejjem "
        "ogħla mir-raba'. Tini erba' birer."
    )
    output, tokens = correct_text(source)
    assert output == expected
    langli = next(token for token in _word_tokens(tokens) if token.get("original") == "langli")
    assert [choice["word"] for choice in langli["choices"]] == ["l-anġli"]


def test_clean_paragraph_preserves_lexical_content_and_apostrophes():
    source = (
        "Caritas Malta se torganizza Thrift Day b’risq il-komunità "
        "tal-parroċċa ta’ Swatar nhar il-5 ta’ Settembru, fil-Knisja "
        "Parrokkjali u s-Sala tal-Parroċċa ta’ Swatar.\n\n"
        "L-attività se tibda wara l-quddiesa tad-9am, bil-ħanut jibqa’ "
        "miftuħ sal-3:30pm. Matul il-ġurnata se jkun hemm disponibbli "
        "ħwejjeġ second-hand, b’inizjattiva li tħeġġeġ lill-pubbliku "
        "jagħmel għażliet żgħar li jistgħu jwasslu għal impatt pożittiv "
        "fil-komunità.\n\n"
        "L-attività qed tingħata l-messaġġ “Small choices, big impact”, "
        "u għandha l-għan li tappoġġja l-komunità tal-parroċċa tas-Swatar.\n\n"
        "Caritas Malta qed tistieden lill-pubbliku jattendi u jappoġġja "
        "din l-inizjattiva, li tgħaqqad ix-xiri ta’ ħwejjeġ ma’ għajnuna "
        "lill-komunità lokali."
    )
    expected = source.replace("’", "'")
    output, tokens = correct_text(source)
    assert output == expected
    assert not any(token.get("unrecognized") for token in _word_tokens(tokens))


def test_noisy_paragraph_reconstructs_clean_surface():
    source = (
        "Caritas Malta se torganizza Thrift Day brisq il komunita tal "
        "parrocca ta’ Swatar nar il 5 ta’ Setembru, fil Knisja Parrokkjali "
        "u s Sala tal Parrocca ta’ Swatar.\n"
        "L attivita se tibda wara l quddisa tad-9am, bil hanut jibqa "
        "miftuh sal 3:30pm. Matul il gurnata se jkun hemm disponibli "
        "hwejjeg second-hand, b inizjattiva li theggeg lill publiku jaghmel "
        "ghazliet zghar li jistghu jwasslu ghal impatt pozittiv fil komunita.\n"
        "L attivita qed tinata l messagg “Small choices, big impact”, u "
        "ghandha l ghan li tappoggja l komunita tal parrocca tas Swatar.\n"
        "Caritas Malta qed tistieden lill publiku jattendi u jappogja din l "
        "inizjattiva, li taqqad ix xiri ta’ hwejjeg ma ajnuna lill komunita lokali."
    )
    expected = (
        "Caritas Malta se torganizza Thrift Day b'risq il-komunità "
        "tal-parroċċa ta' Swatar nhar il-5 ta' Settembru, fil-Knisja "
        "Parrokkjali u s-Sala tal-Parroċċa ta' Swatar.\n"
        "L-attività se tibda wara l-quddiesa tad-9am, bil-ħanut jibqa' "
        "miftuħ sal-3:30pm. Matul il-ġurnata se jkun hemm disponibbli "
        "ħwejjeġ second-hand, b'inizjattiva li tħeġġeġ lill-pubbliku jagħmel "
        "għażliet żgħar li jistgħu jwasslu għal impatt pożittiv fil-komunità.\n"
        "L-attività qed tingħata l-messaġġ “Small choices, big impact”, u "
        "għandha l-għan li tappoġġja l-komunità tal-parroċċa tas-Swatar.\n"
        "Caritas Malta qed tistieden lill-pubbliku jattendi u jappoġġja din "
        "l-inizjattiva, li tgħaqqad ix-xiri ta' ħwejjeġ ma' għajnuna "
        "lill-komunità lokali."
    )
    assert correct_text(source)[0] == expected


def test_directional_l_before_spatial_deictic_is_not_an_article():
    output, tokens = correct_text("mur lhemm")
    assert output == "Mur 'l hemm."
    choices = [
        choice["word"]
        for token in _word_tokens(tokens)
        for choice in token.get("choices", [])
    ]
    assert "l-hemm" not in choices
    assert "l'ħemm" not in choices
