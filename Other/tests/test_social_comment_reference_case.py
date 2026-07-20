from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Essentials import app


SOURCE = (
    "Ma smajniex hekk qabel l-elezzjoni. Ghall-gid taghna u tal-ambjent hemm bzonn "
    "inaqqas il-popolazzjoni mela jzidha !\n"
    "Qabel Mejju ma qalx hekk! Issa qed jghidilkom grazzi talli telajtuh u njorakom "
    "ta kemm gergirtu bl 'overpopulation'!!\n"
    "xi hadd nizzel karozza min Cipru....min aw jirranga 'tranport' dak x lahjar "
    "'container' jew awn xi hadd min jgibom ?? ghax behsibni  nixtri karozza min hemm"
)

EXPECTED = (
    "Ma smajniex hekk qabel l-elezzjoni. Għall-ġid tagħna u tal-ambjent hemm bżonn "
    "inaqqas il-popolazzjoni mela jżidha!\n"
    "Qabel Mejju ma qalx hekk! Issa qed jgħidilkom grazzi talli tellajtuh u injorakom "
    "ta' kemm gergirtu bl-overpopulation!!\n"
    "Xi ħadd niżżel karozza min ċipru....Min hawn jirranġa transport dak x'l-aħjar "
    "container jew hawn xi ħadd min jġibhom?? Għax beħsibni nixtri karozza minn hemm."
)


def _choices(result, original):
    for token in result["tokens"]:
        if token.get("original") == original:
            return [choice["word"] for choice in token.get("choices", [])]
    raise AssertionError(f"Missing token: {original!r}")


def test_social_comment_reference_case():
    result = app.spellchecker.correct_text_rich(SOURCE)

    assert result["corrected_text"] == EXPECTED
    assert _choices(result, "nizzel")[:2] == ["niżżel", "niżel"]
    assert _choices(result, "ta kemm")[:2] == ["ta' kemm", "ta kemm"]
    assert _choices(result, "min Cipru")[:2] == ["min ċipru", "minn ċipru"]
    assert _choices(result, "min aw")[:2] == ["Min hawn", "Minn hawn"]
