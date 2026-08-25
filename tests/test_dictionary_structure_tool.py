from pathlib import Path

from dictionary_structure_tool import DictionaryStructureIndex, graphemes


def test_special_graphemes_occupy_one_slot():
    assert graphemes("għie") == ("għ", "ie")


def test_k_and_v_structure_search(tmp_path: Path):
    dics = tmp_path / "dics"
    dics.mkdir()
    (dics / "words.dic").write_text(
        "baħar/SINGNOUNM-sea\nċans/SINGNOUNM-chance\n",
        encoding="utf-8",
    )
    index = DictionaryStructureIndex(dics)
    results, total = index.search(["K", "V", "K", "V", "K"])
    assert total == 1
    assert results[0]["word"] == "baħar"
    assert results[0]["meaning"] == "sea"


def test_exact_gh_and_ie_slots(tmp_path: Path):
    dics = tmp_path / "dics"
    dics.mkdir()
    (dics / "words.dic").write_text("għie/SINGNOUN-example\n", encoding="utf-8")
    index = DictionaryStructureIndex(dics)
    results, total = index.search(["għ", "ie"])
    assert total == 1
    assert results[0]["word"] == "għie"
