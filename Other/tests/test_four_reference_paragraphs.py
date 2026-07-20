# -*- coding: utf-8 -*-
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Essentials import app


spellchecker = app.spellchecker


CASES = (
    (
        """Meta tixtri halib mhux skadut , pero l pakkett kien qisu minfuh (ir ragel xtrah sewwa mhux jien ) , gejt id dar u kif ftahtu kellu riha tinten kbira .
Tal hanut qalli li mhux tort tieghu . Min ghandu jtini l flus lura ? PLUS LI WEHILT 7 EURO WOLT biex gibt il halib ghax kien ghat tifel . Fir rahal hawn il festa u ma tistax iccaqlaq il karozza PLUS li il hdax u nofs ta bil lejl""",
        """Meta tixtri ħalib mhux skadut, però l-pakkett kien qisu minfuħ (ir-raġel xtrah sewwa mhux jien), ġejt id-dar u kif ftaħtu kellu riħa tinten kbira.
Tal-ħanut qalli li mhux tort tiegħu. Min għandu jtini l-flus lura? PLUS LI WEĦILT 7 EURO WOLT biex ġibt il-ħalib għax kien għat-tifel. Fir-raħal hawn il-festa u ma tistax iċċaqlaq il-karozza PLUS li il-ħdax u nofs ta' billejl.""",
    ),
    (
        """Bongu lil kullhadd
Ghandi mobile 'i phone 17 pro max' u nizilt update tal beta 27.0 u ghandi bzonn inhejih ghax qed jehel u mhux jahdem sew tafu il xi hadd biex jerga jaghmiluli u jigi updated kif supost dak il hin u minajr ma hassar xej south area hekk jista' ikun grazzi""",
        """Bonġu lil kulħadd
Għandi mobile iphone 17 pro max u niżżilt update tal-beta 27.0 u għandi bżonn inħejjih għax qed jeħel u mhux jaħdem sew tafu 'il xi ħadd biex jerġa' jagħmilhuli u jiġi updated kif suppost dak il-ħin u mingħajr ma ħassar xejn south area hekk jista' jkun grazzi.""",
    ),
    (
        """Grazzi ghal owner tal blokk ta apartamenti li jikri 'air b n b' ahna r residenti rridu ta kuljum inhabtu wicna ma din il mandra fuq il bankini ghadni qed nistenna l pulizija ha jigu fuq il post triq il grigal marsaskala.""",
        """Grazzi għall-owner tal-blokk t'appartamenti li jikri air b n b aħna r-residenti rridu ta' kuljum inħabbtu wiċċna ma din il-mandra fuq il-bankini għadni qed nistenna l-pulizija ħa jiġu fuq il-post triq il-grigal Marsaskala.""",
    ),
    (
        """Ghar mill-fakulta tal-ligi m’hawx.
L-istudenti qatt ma kienu priorita.
Mhux talli jaghmlu hajjet l istudenti mizerja matul is sena imma issa studenti tat-tlieta qeghdin tkarrbu ghal marki filwaqt li tas-snin l ohra qeghdin jircivuhom.
Fakulta li hlief tmur dejjem ghal ghar u taqa ghan nejk ma taghmilx.
Hasra li hemmek biss hawn ghal min jixtieq jigradwa fil ligi.""",
        """Agħar mill-fakultà tal-liġi m'hawnx.
L-istudenti qatt ma kienu prijorità.
Mhux talli jagħmlu ħajjet l-istudenti miżerja matul is-sena imma issa studenti tat-tlieta qegħdin tkarrbu għall-marki filwaqt li tas-snin l-oħra qegħdin jirċivuhom.
Fakultà li ħlief tmur dejjem għall-agħar u taqa' għan-nejk ma tagħmilx.
Ħasra li hemmhekk biss hawn għal min jixtieq jiggradwa fil-liġi.""",
    ),
)


for source, expected in CASES:
    actual = spellchecker.correct_text_rich(source, edit_distance_tolerance=2)[
        "corrected_text"
    ]
    assert actual == expected, f"expected:\n{expected}\n\nactual:\n{actual}"


print("four reference paragraph checks passed")
