from typing import List, Optional
from spellchecker.schema import Candidate, ErrorClass, RiskClass, Token
from spellchecker.normalization import normalize_word, VOWELS

CONSONANT_CLUSTER_PREFIXES = ("nk", "nd", "nt", "ns", "nx", "nż", "nġ", "nċ", "nm", "mb", "sp", "st", "sk", "sq", "sl", "sm", "sn")


def get_prev_word_token(tokens: List[Token], index: int) -> Optional[Token]:
    for i in range(index - 1, -1, -1):
        if tokens[i].token_type == "word":
            return tokens[i]
    return None


def generate_initial_i_candidates(tokens: List[Token], index: int, lexicon) -> List[Candidate]:
    candidates: List[Candidate] = []
    token = tokens[index]
    if token.token_type != "word":
        return candidates

    norm = token.normalized
    raw = token.text
    if not norm or norm in ("skond", "skont"):
        return candidates

    prev_word = get_prev_word_token(tokens, index)
    is_title = token.casing == "title" or (prev_word is None)

    # 1. wara inmut -> Wara mmut / inmut after vowel -> mmut
    if norm == "inmut" and prev_word and prev_word.normalized and prev_word.normalized[-1] in VOWELS:
        rep = "mmut"
        candidates.append(
            Candidate(
                source_start=token.start,
                source_end=token.end,
                original_text=raw,
                replacement=rep,
                operation_type=ErrorClass.INITIAL_I,
                risk_class=RiskClass.LOW,
                sources=["initial_i_inmut_mmut"],
                hard_valid=True,
            )
        )

    # 2. sperjenzajt -> esperjenzajt (prioritize e- for sp/st/sk/sq)
    if norm.startswith(("sp", "st", "sk", "sq")):
        rep_e = "e" + norm
        if is_title or token.casing == "title":
            rep_e = rep_e.capitalize()
        candidates.append(
            Candidate(
                source_start=token.start,
                source_end=token.end,
                original_text=raw,
                replacement=rep_e,
                operation_type=ErrorClass.INITIAL_I,
                risk_class=RiskClass.LOW,
                sources=["initial_e_add"],
                hard_valid=True,
            )
        )

    # 3. Add initial i for consonant cluster forms (nkiser -> inkiser, nduru -> induru, ndur -> indur, nmut -> immut)
    if not norm.startswith(("i", "e")):
        if norm == "nmut":
            rep = "immut"
            if is_title and token.casing == "title":
                rep = rep.capitalize()
            candidates.append(
                Candidate(
                    source_start=token.start,
                    source_end=token.end,
                    original_text=raw,
                    replacement=rep,
                    operation_type=ErrorClass.INITIAL_I,
                    risk_class=RiskClass.LOW,
                    sources=["initial_i_immut"],
                    hard_valid=True,
                )
            )
        elif lexicon.contains("i" + norm) or norm.startswith(CONSONANT_CLUSTER_PREFIXES):
            rep = "i" + norm
            if is_title and token.casing == "title":
                rep = rep.capitalize()
            candidates.append(
                Candidate(
                    source_start=token.start,
                    source_end=token.end,
                    original_text=raw,
                    replacement=rep,
                    operation_type=ErrorClass.INITIAL_I,
                    risk_class=RiskClass.LOW,
                    sources=["initial_i_add"],
                    hard_valid=True,
                )
            )

    # 4. Remove initial i (e.g. inkisret -> nkisret after vowel, induru -> nduru after vowel)
    if norm.startswith("i") and len(norm) >= 3:
        stem = norm[1:]
        if prev_word and prev_word.normalized and prev_word.normalized[-1] in VOWELS:
            if lexicon.contains(stem) or stem.startswith(CONSONANT_CLUSTER_PREFIXES):
                candidates.append(
                    Candidate(
                        source_start=token.start,
                        source_end=token.end,
                        original_text=raw,
                        replacement=stem,
                        operation_type=ErrorClass.INITIAL_I,
                        risk_class=RiskClass.LOW,
                        sources=["initial_i_remove"],
                        hard_valid=True,
                    )
                )

    return candidates
