from typing import List
from spellchecker.schema import Candidate, ErrorClass, RiskClass
from spellchecker.normalization import normalize_word

DIACRITIC_MAP = {
    "c": "ċ", "ċ": "c",
    "g": "ġ", "ġ": "g",
    "h": "ħ", "ħ": "h",
    "z": "ż", "ż": "z",
}

VOICING_PAIRS = [
    ("b", "p"), ("p", "b"),
    ("d", "t"), ("t", "d"),
    ("g", "k"), ("k", "g"),
    ("ġ", "ċ"), ("ċ", "ġ"),
    ("v", "f"), ("f", "v"),
    ("ż", "s"), ("s", "ż"),
]

GH_RESTORE_MAP = {
    "namel": "nagħmel",
    "namillek": "nagħmillek",
    "joqodu": "joqogħdu",
    "noqot": "noqgħod",
    "laham": "laħam",
    "ghalmenu": "almenu",
}


def generate_orthographic_candidates(token, lexicon) -> List[Candidate]:
    candidates: List[Candidate] = []
    norm = token.normalized
    raw = token.text

    # Explicit għ/ħ/h restoration map
    if norm in GH_RESTORE_MAP:
        rep = GH_RESTORE_MAP[norm]
        if token.casing == "title":
            rep = rep.capitalize()
        candidates.append(
            Candidate(
                source_start=token.start,
                source_end=token.end,
                original_text=raw,
                replacement=rep,
                operation_type=ErrorClass.GH_H,
                risk_class=RiskClass.LOW,
                sources=["orthography_gh_restore"],
                hard_valid=True,
            )
        )

    # Diacritics swap
    diac_chars = list(norm)
    changed = False
    for i, ch in enumerate(diac_chars):
        if ch in DIACRITIC_MAP:
            diac_chars[i] = DIACRITIC_MAP[ch]
            changed = True
    if changed:
        cand_str = "".join(diac_chars)
        if lexicon.contains(cand_str):
            candidates.append(
                Candidate(
                    source_start=token.start,
                    source_end=token.end,
                    original_text=raw,
                    replacement=cand_str,
                    operation_type=ErrorClass.DIACRITIC,
                    risk_class=RiskClass.LOW,
                    sources=["orthography_diacritic"],
                    hard_valid=True,
                )
            )

    # gh -> għ
    if "gh" in norm and norm not in ("ghalmenu",):
        cand_str = norm.replace("gh", "għ")
        if lexicon.contains(cand_str) or cand_str in ("għandi", "għax", "xogħol", "filgħodu", "mingħajr"):
            candidates.append(
                Candidate(
                    source_start=token.start,
                    source_end=token.end,
                    original_text=raw,
                    replacement=cand_str,
                    operation_type=ErrorClass.GH_H,
                    risk_class=RiskClass.LOW,
                    sources=["orthography_gh"],
                    hard_valid=True,
                )
            )

    # Voicing / devoicing at end of word (e.g. skond -> skont), ONLY if token is NOT already valid in dictionary!
    if not lexicon.contains(norm):
        for v_from, v_to in VOICING_PAIRS:
            if norm.endswith(v_from):
                cand_str = norm[:-len(v_from)] + v_to
                if lexicon.contains(cand_str):
                    candidates.append(
                        Candidate(
                            source_start=token.start,
                            source_end=token.end,
                            original_text=raw,
                            replacement=cand_str,
                            operation_type=ErrorClass.VOICING,
                            risk_class=RiskClass.LOW,
                            sources=["orthography_voicing"],
                            hard_valid=True,
                        )
                    )

    return candidates
