# -*- coding: utf-8 -*-
import app


spellchecker = app.spellchecker


def assert_text(source, expected):
    actual = spellchecker.correct_text_rich(
        source,
        edit_distance_tolerance=2,
    )["corrected_text"]
    assert actual == expected, f"{source!r}: expected {expected!r}, got {actual!r}"


def assert_suggestions(word, expected):
    actual = spellchecker.suggest(
        word,
        limit=8,
        edit_distance_tolerance=2,
    )
    assert actual == expected, f"{word!r}: expected {expected!r}, got {actual!r}"


# mill- and its assimilated forms
assert_text("mid dinja", "mid-dinja")
assert_text("mid dar", "mid-dar")
assert_text("mill skola", "mis-skola")
assert_text("mil tifel", "mit-tifel")
assert_text("mic Cina", "miċ-Ċina")
assert_text("minn ic Cina", "miċ-Ċina")

# ma + article must contract; bare ma before a noun/verb must remain separate.
assert_text("ma l omm", "mal-omm")
assert_text("ma' l-omm", "mal-omm")
assert_text("ma d dar", "mad-dar")
assert_text("ma d-dar", "mad-dar")
assert_text("ma dar", "ma dar")
assert_text("ma marret", "ma marret")
assert_text("ma għid", "ma tgħidx")
assert_text("ma għidu", "ma tgħidux")
assert_text("ma ikser", "ma tiksirx")
assert_text("ma iksru", "ma tiksrux")
assert_text("ma iksrux", "ma tiksrux")
assert_text("ma tiksirx", "ma tiksirx")
assert_text("ma tiksrux", "ma tiksrux")
assert_text("ma kisser", "ma kisser")

# Other article-preposition families.
assert_text("għall skola", "għas-skola")
assert_text("lil tifla", "lit-tifla")
assert_text("fil dar", "fid-dar")
assert_text("bl idejn", "bl-idejn")
assert_text("sal belt", "sal-belt")

# għi is never an i/ie replacement path.
assert spellchecker.correct_word("mid") == "mid"
assert "mgħid" not in spellchecker.suggest("mid", limit=8, edit_distance_tolerance=2)
assert spellchecker.correct_word("mghid") == "mgħid"
assert_suggestions("mghid", ["mgħid"])

# Capitalized unknown names remain untouched, while close known places may fix.
# "Marsaskla" and "Berlni" below are synthetic typo inputs, not place variants.
assert_suggestions("Noel", [])
assert_suggestions("Cina", ["Ċina"])
assert_suggestions("Parigi", ["Pariġi"])
assert_suggestions("Marsaskla", ["Marsaskala"])
assert_suggestions("Berlni", [])
assert_suggestions("Ħ'Attard", ["Ħ'Attard"])
assert_suggestions("L-Imsida", ["L-Imsida"])
assert_suggestions("Msida", ["Msida"])
assert_suggestions("Il-Gżira", ["Il-Gżira"])
assert_suggestions("Gżira", ["Gżira"])

# A distant place must not replace an unknown capitalized token.
assert_suggestions("Qzvra", [])

print("place and article regression checks passed")
