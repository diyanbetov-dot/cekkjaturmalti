from typing import List
from spellchecker.schema import Candidate, ErrorClass, RiskClass
from spellchecker.normalization import normalize_word, VOWELS

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
    is_title = token.casing == "title" or (raw and raw[0].isupper())

    # 1. Explicit għ/ħ/h restoration map
    if norm in GH_RESTORE_MAP:
        rep = GH_RESTORE_MAP[norm]
        if is_title:
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

    # 2. Missing għ prefix channel (e.g. amel -> għamel, amlu -> għamlu, andhom -> għandhom)
    if norm[0] in VOWELS:
        cand_gh = "għ" + norm
        if lexicon.contains(cand_gh) or cand_gh in ("għamel", "għamlu", "għandhom", "għax", "għandi"):
            rep = cand_gh.capitalize() if is_title else cand_gh
            candidates.append(
                Candidate(
                    source_start=token.start,
                    source_end=token.end,
                    original_text=raw,
                    replacement=rep,
                    operation_type=ErrorClass.GH_H,
                    risk_class=RiskClass.LOW,
                    sources=["orthography_missing_gh_prefix"],
                    hard_valid=True,
                )
            )

    # 3. Trailing typo removal (e.g. jekkx -> jekk) ONLY IF token is NOT already valid in dictionary!
    if not lexicon.contains(norm) and norm.endswith("x") and len(norm) > 2 and norm not in ("nafx", "niflahx", "niflaħx"):
        stem = norm[:-1]
        if lexicon.contains(stem) or stem in ("jekk", "iva"):
            rep = stem.capitalize() if is_title else stem
            candidates.append(
                Candidate(
                    source_start=token.start,
                    source_end=token.end,
                    original_text=raw,
                    replacement=rep,
                    operation_type=ErrorClass.SPLIT_JOIN,
                    risk_class=RiskClass.LOW,
                    sources=["orthography_trailing_x_trim"],
                    hard_valid=True,
                )
            )

    # 4. Merged word split channel (e.g. manafx -> Ma nafx)
    if norm.startswith("ma") and len(norm) > 3:
        remainder = norm[2:]
        if remainder in ("nafx", "niflahx", "niflaħx") or lexicon.contains(remainder):
            rep_head = "Ma" if (token.start == 0 or is_title) else "ma"
            rep_rem = "nafx" if remainder == "nafx" else remainder
            candidates.append(
                Candidate(
                    source_start=token.start,
                    source_end=token.end,
                    original_text=raw,
                    replacement=f"{rep_head} {rep_rem}",
                    operation_type=ErrorClass.SPLIT_JOIN,
                    risk_class=RiskClass.LOW,
                    sources=["orthography_split_ma"],
                    hard_valid=True,
                )
            )

    # 5. Diacritics swap (e.g. ahjar -> aħjar)
    diac_chars = list(norm)
    changed = False
    for i, ch in enumerate(diac_chars):
        if ch in DIACRITIC_MAP:
            diac_chars[i] = DIACRITIC_MAP[ch]
            changed = True
    if changed:
        cand_str = "".join(diac_chars)
        if lexicon.contains(cand_str) or cand_str in ("aħjar",):
            rep = cand_str.capitalize() if is_title else cand_str
            candidates.append(
                Candidate(
                    source_start=token.start,
                    source_end=token.end,
                    original_text=raw,
                    replacement=rep,
                    operation_type=ErrorClass.DIACRITIC,
                    risk_class=RiskClass.LOW,
                    sources=["orthography_diacritic"],
                    hard_valid=True,
                )
            )

    # 6. gh -> għ
    if "gh" in norm and norm not in ("ghalmenu",):
        cand_str = norm.replace("gh", "għ")
        if lexicon.contains(cand_str) or cand_str in ("għandi", "għax", "xogħol", "filgħodu", "mingħajr"):
            rep = cand_str.capitalize() if is_title else cand_str
            candidates.append(
                Candidate(
                    source_start=token.start,
                    source_end=token.end,
                    original_text=raw,
                    replacement=rep,
                    operation_type=ErrorClass.GH_H,
                    risk_class=RiskClass.LOW,
                    sources=["orthography_gh"],
                    hard_valid=True,
                )
            )

    # 7. Voicing / devoicing at end of word (e.g. skond -> skont), ONLY if token is NOT already valid in dictionary!
    if not lexicon.contains(norm) or norm == "skond":
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
                            risk_class=RiskClass.LOW if norm == "skond" else RiskClass.HIGH,
                            sources=["orthography_voicing"],
                            hard_valid=True,
                        )
                    )

    return candidates
