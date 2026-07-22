import pytest
from Essentials.app import spellchecker, app
from Essentials.helpers.composite_generator import MalteseCompositeGenerator

def test_composite_generator_instantiation():
    comp = MalteseCompositeGenerator(spellchecker)
    assert comp is not None
    assert comp.spellchecker == spellchecker

def test_composite_generator_grapheme_and_suffix():
    comp = MalteseCompositeGenerator(spellchecker)
    # nitfahha -> nitfagħha
    cands = comp.generate_candidates("nitfahha")
    assert "nitfagħha" in cands or "Nitfagħha" in cands

def test_composite_generator_apostrophe_compound():
    comp = MalteseCompositeGenerator(spellchecker)
    # bkelma -> b'kelma
    cands = comp.generate_candidates("bkelma")
    assert "b'kelma" in cands

def test_composite_generator_generalized_hom_om_suffixes():
    comp = MalteseCompositeGenerator(spellchecker)
    # narhom -> narahom, narom -> narahom, jarhom -> jarahom, tarom -> tarahom
    cands_narhom = comp.generate_candidates("narhom")
    cands_narom = comp.generate_candidates("narom")
    cands_jarhom = comp.generate_candidates("jarhom")
    cands_tarom = comp.generate_candidates("tarom")

    assert "narahom" in cands_narhom
    assert "narahom" in cands_narom
    assert "jarahom" in cands_jarhom
    assert "tarahom" in cands_tarom

def test_no_h_to_gh_replacement():
    comp = MalteseCompositeGenerator(spellchecker)
    # Single 'h' should not be transformed to 'għ'
    # E.g. 'huma' has 'h' -> 'ħuma' (not 'għuma')
    cands = comp.generate_candidates("huma")
    assert not any("għuma" in c for c in cands)

def test_composite_pipeline_integration():
    # End-to-end rich correction test
    res = spellchecker.correct_text_rich("nitfahha bkelma")
    corrected = res["corrected_text"]
    assert "Nitfagħha" in corrected or "nitfagħha" in corrected
    assert "b'kelma" in corrected
