from typing import List, Optional, Tuple
from spellchecker.schema import MorphAnalysis

DO_SUFFIXES = ["ni", "k", "ek", "ok", "u", "h", "ha", "na", "kom", "hom"]
IDO_SUFFIXES = ["li", "lek", "lok", "lu", "lha", "lna", "lkom", "lhom"]


def parse_attached_pronoun(surface: str) -> Optional[Tuple[str, str, str]]:
    """
    Returns (base_stem, do_suffix, ido_suffix) if surface can be validly decomposed.
    """
    for ido in IDO_SUFFIXES:
        if surface.endswith(ido):
            stem = surface[: -len(ido)]
            for do in DO_SUFFIXES:
                if stem.endswith(do):
                    base = stem[: -len(do)]
                    if len(base) >= 2:
                        return base, do, ido
            if len(stem) >= 2:
                return stem, "", ido

    for do in DO_SUFFIXES:
        if surface.endswith(do):
            base = surface[: -len(do)]
            if len(base) >= 2:
                return base, do, ""

    return None


def validate_suffixed_verb(base_verb: str, do: str, ido: str, lexicon) -> bool:
    if not base_verb:
        return False

    # Normalize weak verb vowels before suffix
    candidates_to_check = [base_verb]
    if base_verb.endswith("ie"):
        candidates_to_check.append(base_verb[:-2] + "a")
    if base_verb.endswith("i"):
        candidates_to_check.append(base_verb[:-1] + "a")
        candidates_to_check.append(base_verb)

    for cand in candidates_to_check:
        if lexicon.contains(cand):
            return True
    return False
