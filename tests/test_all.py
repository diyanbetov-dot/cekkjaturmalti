import pytest
from pathlib import Path
from spellchecker.pipeline import SpellcheckerPipeline
from neural.dataset import parse_ai_corrections_file


@pytest.fixture(scope="module")
def pipeline():
    return SpellcheckerPipeline()


# A. DATA / ASSETS
def test_data_assets_loading(pipeline):
    assert len(pipeline.lexicon.word_map) > 10000
    pairs = parse_ai_corrections_file()
    assert len(pairs) > 50


# B. KEEP SAFETY
@pytest.mark.parametrize(
    "clean_text",
    [
        "Il-kafè għadu sħun.",
        "Ma fhimtx x'ridt tgħid.",
        "Erbatax-il persuna attendew il-laqgħa.",
        "Wieħed u għoxrin persuna attendew.",
        "Ħdax-il baqra kienu mexjin fit-triq.",
        "Ħames baqriet kienu fl-għalqa.",
        "Mort l-isptar imma ma sibtx parking.",
    ],
)
def test_keep_safety(pipeline, clean_text):
    res = pipeline.check(clean_text)
    assert res["corrected_text"] == clean_text


# C. INITIAL i
@pytest.mark.parametrize(
    "inp,expected",
    [
        ("il bieb nkiser", "Il-bieb inkiser"),
        ("it tieqa inkisret", "It-tieqa nkisret"),
        ("Morna induru dawra mal belt", "Morna nduru dawra mal-belt"),
        ("Mort ndur dawra mal belt", "Mort indur dawra mal-belt"),
        ("wara inmut", "Wara mmut"),
        ("nmut", "immut"),
        ("mbierek", "imbierek"),
        ("sperjenzajt", "esperjenzajt"),
    ],
)
def test_initial_i(pipeline, inp, expected):
    res = pipeline.check(inp)
    assert res["corrected_text"].rstrip(".").casefold() == expected.rstrip(".").casefold()


# D. NUMERALS
@pytest.mark.parametrize(
    "inp,expected",
    [
        ("erba bozza", "Erba' bozoz"),
        ("hamsa baqar", "Ħames baqriet"),
        ("hdax baqra", "Ħdax-il baqra"),
        ("għoxrin baqar", "Għoxrin baqra"),
        ("ibni ghandu tlett snin u binti hamsa", "Ibni għandu tliet snin u binti għandha ħamsa"),
    ],
)
def test_numerals(pipeline, inp, expected):
    res = pipeline.check(inp)
    assert res["corrected_text"].rstrip(".") == expected.rstrip(".")


# E. ENGLISH
def test_english_protection(pipeline):
    res = pipeline.check("il battery")
    assert "battery" in res["corrected_text"]
    assert res["corrected_text"] in ("l-battery", "L-battery", "il-battery", "Il-battery")


# F. NAMES / PLACES
def test_names_places(pipeline):
    res1 = pipeline.check("ismu john imma kulhadd isejjah jack")
    assert "John" in res1["corrected_text"]
    assert "Jack" in res1["corrected_text"]

    res2 = pipeline.check("Jisimni amy")
    assert "Amy" in res2["corrected_text"]

    res3 = pipeline.check("san pawl")
    assert "San Pawl" in res3["corrected_text"] or "san Pawl" in res3["corrected_text"]


# G. għ / h / ħ
@pytest.mark.parametrize(
    "inp,expected",
    [
        ("laham", "laħam"),
        ("namel", "nagħmel"),
        ("namillek", "nagħmillek"),
        ("joqodu", "joqogħdu"),
        ("noqot", "noqgħod"),
        ("ghalmenu", "almenu"),
    ],
)
def test_gh_h_ħ(pipeline, inp, expected):
    res = pipeline.check(inp)
    assert res["corrected_text"].rstrip(".") == expected.rstrip(".")


# H. VOICING
def test_voicing(pipeline):
    res = pipeline.check("skond")
    assert res["corrected_text"] == "skont"
    res2 = pipeline.check("impjieg")
    assert res2["corrected_text"] == "impjieg"


# I. SUFFIX / MORPHOLOGY
def test_suffix_morphology(pipeline):
    res = pipeline.check("sidha")
    assert res["corrected_text"] == "sidha"


# J. ARTICLE / PREPOSITION
@pytest.mark.parametrize(
    "inp,expected",
    [
        ("ghall xi hadd", "Għal xi ħadd"),
        ("f idejk", "F'idejk"),
        ("ma membru", "Ma' membru"),
        ("ma niflahx", "Ma niflaħx"),
        ("mar r Rabat", "Mar ir-Rabat"),
        ("mar d dar", "Mar id-dar"),
        ("mar s supermarket", "Mar is-supermarket"),
        ("mill l isptar", "Mill-isptar"),
    ],
)
def test_article_preposition(pipeline, inp, expected):
    res = pipeline.check(inp)
    assert res["corrected_text"].rstrip(".") == expected.rstrip(".")


# K. AGREEMENT
@pytest.mark.parametrize(
    "inp,expected",
    [
        ("triq twil", "Triq twila"),
        ("il bieb inkisret", "Il-bieb inkiser"),
        ("it tieqa nkiser", "It-tieqa nkisret"),
    ],
)
def test_agreement(pipeline, inp, expected):
    res = pipeline.check(inp)
    assert res["corrected_text"].rstrip(".") == expected.rstrip(".")


# L. RECENT OVERCORRECTION GUARDS
def test_overcorrection_guards(pipeline):
    res = pipeline.check("Ir-raġel mar u ġie lura.")
    assert "mar-u" not in res["corrected_text"]
    assert "mar u" in res["corrected_text"]

    res2 = pipeline.check("Pilloli tad-dieta għalik.")
    assert "Ippillali" not in res2["corrected_text"]
    assert "Pilloli" in res2["corrected_text"]


# M. ERROR MEMORY
@pytest.mark.parametrize(
    "inp,expected",
    [
        ("mijaw", "miegħu"),
        ("jajdula", "jgħidulha"),
        ("jajdulek", "jgħidulek"),
        ("jorukhom", "jurukom"),
        ("imbad", "mbagħad"),
        ("gimghatejn", "ġimagħtejn"),
        ("xolhom", "xogħolhom"),
    ],
)
def test_error_memory(pipeline, inp, expected):
    res = pipeline.check(inp)
    assert res["corrected_text"] == expected


# N. DECODER / RENDERER
def test_decoder_renderer(pipeline):
    text = "Mela  darba   l  mara  qalet   lir-raġel."
    res = pipeline.check(text)
    assert "  " in res["corrected_text"]  # Whitespace preserved
