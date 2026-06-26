# -*- coding: utf-8 -*-
import app


spellchecker = app.spellchecker


RAW = """Ma tafx ala??? Mela ha nghidlek.. nahseb kollox tghalmet fhajjitha barra titlef!! Mil bidu uriet x'hini li jew kif tghid hi u taqbel maghha jew issabat siqa u tiggagbina, di l-aqwa! git ma lahhar 3 u tiġi tghid quddiem Malta kolla li nies mux qed jaraw il karatru taghha u huma biss jafu xem hemm gew.. nahseb trid tkun vera egoista u l 'ego' malla ta. meta tfalli flok tara xmar hazin u fix tista' tirranġa tara f'min ha twehhel, in nies ma tistax tatakhom allura daret ghal Drinu li dan qiegħed 'coupled' maghha igifiri jekk titlaq hi irid jitlaq hu ukoll, belhitu tifel. Bin nies themm ġew titnejjek imma bin nies li qed jarawha ma tista titnejjek qatt!! Drinu dahal jithaq u hireg min hemm b’ ‘depression' ta l ostja habba fiha. Tirbah trid u la tafhom u lanqas jafuwha.. Ilum ikkonfermajt 100.101% min vera qieghed hemm ghal flus!! Vera logħba u alkemm memx konnessjonijiet, però hi 'topp' f4 snin qatt hadd mghamila ovja daqshekk li jkunu em ghal flus!! Nixtieq inkun naf xha jiġri minnha jekk ma terbahx.."""


EXPECTED_FRAGMENTS = (
    "Ma tafx għala???",
    "Mela ħa ngħidlek",
    "naħseb kollox tgħallmet f'ħajjitha",
    "Mil-bidu wriet",
    "tgħid hi u taqbel magħha",
    "issabbat sieqha u tiġġakbina",
    "dil-aqwa",
    "ġiet ma l-aħħar 3",
    "Malta kollha",
    "nies mhux qed jaraw il-karattru tagħha",
    "jafu x'hemm hemm ġew",
    "x'mar ħażin u fiex tista",
    "f'min ħa tweħħel",
    "in-nies ma tistax tattakkhom",
    "għal Drinu",
    "coupled magħha iġifieri",
    "jitlaq hu wkoll",
    "bellhitu tifel",
    "t'hemm ġew",
    "Drinu daħal jidħaq u ħiereġ minn hemm",
    "tal-ostja ħabba fiha",
    "Tirbaħ trid",
    "lanqas jafuha",
    "llum ikkonfermajt",
    "qiegħed hemm għal flus",
    "għalkemm m'hemmx konnessjonijiet",
    "hi topp f'4 snin",
    "ħadd m'għamilha ovvja",
    "jkunu hemm",
    "x'ħa jiġri minnha jekk ma tirbaħx",
)


FORBIDDEN_FRAGMENTS = (
    "ifhajjitha",
    "għerietx",
    "is-sabat",
    "isqra",
    "tiggabbaxha",
    "din l-aqwa",
    "kolla",
    "xhew",
    "mgħnin",
    "twebbel",
    "għin nies",
    "tkakkahom",
    "Trinu",
    "għattaq",
    "iret min hemm",
    "jafuwha",
    "bilkemm m'hemmx",
    "imghamila",
    "trabbahx",
)


corrected = spellchecker.correct_text_rich(RAW, edit_distance_tolerance=1)[
    "corrected_text"
]

for fragment in EXPECTED_FRAGMENTS:
    assert fragment in corrected, f"expected {fragment!r} in {corrected!r}"

for fragment in FORBIDDEN_FRAGMENTS:
    assert fragment not in corrected, f"forbidden {fragment!r} in {corrected!r}"

for word, expected in {
    "kolla": "kollha",
    "ukoll": "wkoll",
    "alkemm": "għalkemm",
    "jafuwha": "jafuha",
    "tiggagbina": "tiġġakbina",
    "themm": "t'hemm",
}.items():
    suggestions = spellchecker.suggest(word, limit=8, edit_distance_tolerance=1)
    assert expected in suggestions, f"{word!r}: expected {expected!r} in {suggestions!r}"

print("social comment regression checks passed")
