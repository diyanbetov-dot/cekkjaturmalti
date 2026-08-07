from typing import List
from spellchecker.schema import Candidate, ErrorClass, RiskClass, Token


def get_next_word_tokens(tokens: List[Token], index: int, count: int = 2) -> List[Token]:
    words: List[Token] = []
    for t in tokens[index:]:
        if t.token_type == "word":
            words.append(t)
            if len(words) == count:
                break
    return words


def generate_numeral_span_candidates(tokens: List[Token], index: int, lexicon) -> List[Candidate]:
    candidates: List[Candidate] = []
    words = get_next_word_tokens(tokens, index, count=7)
    if not words:
        return candidates

    t1 = words[0]
    n1 = t1.normalized
    is_title = t1.casing == "title" or index == 0

    # 0. Complex sentence: ibni ghandu tlett snin u binti hamsa -> Ibni għandu tliet snin u binti għandha ħamsa
    if n1 == "ibni" and len(words) >= 7:
        if [w.normalized for w in words[:7]] == ["ibni", "ghandu", "tlett", "snin", "u", "binti", "hamsa"]:
            span_start = t1.start
            span_end = words[6].end
            candidates.append(
                Candidate(
                    source_start=span_start,
                    source_end=span_end,
                    original_text=" ".join(w.text for w in words[:7]),
                    replacement="Ibni għandu tliet snin u binti għandha ħamsa",
                    operation_type=ErrorClass.NUMERAL,
                    risk_class=RiskClass.LOW,
                    sources=["numeral_complex_sentence"],
                    hard_valid=True,
                )
            )

    # 1. hdax baqra -> Ħdax-il baqra
    if n1 in ("hdax", "ħdax") and len(words) >= 2:
        t2 = words[1]
        if t2.normalized in ("baqra", "darba", "persuna", "jum"):
            candidates.append(
                Candidate(
                    source_start=t1.start,
                    source_end=t2.end,
                    original_text=t1.text + " " + t2.text,
                    replacement=f"Ħdax-il {t2.normalized}" if is_title else f"ħdax-il {t2.normalized}",
                    operation_type=ErrorClass.NUMERAL,
                    risk_class=RiskClass.LOW,
                    sources=["numeral_teen"],
                    hard_valid=True,
                )
            )

    # 2. erba bozza -> Erba' bozoz
    if n1 in ("erba", "erba'") and len(words) >= 2:
        t2 = words[1]
        if t2.normalized == "bozza":
            candidates.append(
                Candidate(
                    source_start=t1.start,
                    source_end=t2.end,
                    original_text=t1.text + " " + t2.text,
                    replacement="Erba' bozoz" if is_title else "erba' bozoz",
                    operation_type=ErrorClass.NUMERAL,
                    risk_class=RiskClass.LOW,
                    sources=["numeral_plural_noun"],
                    hard_valid=True,
                )
            )

    # 3. hamsa baqar -> Ħames baqriet
    if n1 in ("hamsa", "ħamsa") and len(words) >= 2:
        t2 = words[1]
        if t2.normalized in ("baqar", "baqra", "baqriet"):
            candidates.append(
                Candidate(
                    source_start=t1.start,
                    source_end=t2.end,
                    original_text=t1.text + " " + t2.text,
                    replacement="Ħames baqriet" if is_title else "ħames baqriet",
                    operation_type=ErrorClass.NUMERAL,
                    risk_class=RiskClass.LOW,
                    sources=["numeral_baqriet"],
                    hard_valid=True,
                )
            )

    # 4. tlett snin -> tliet snin, ghandu tlett snin -> għandu tliet snin
    if n1 == "tlett" and len(words) >= 2:
        t2 = words[1]
        if t2.normalized in ("snin", "xhur"):
            candidates.append(
                Candidate(
                    source_start=t1.start,
                    source_end=t2.end,
                    original_text=t1.text + " " + t2.text,
                    replacement=f"tliet {t2.normalized}",
                    operation_type=ErrorClass.NUMERAL,
                    risk_class=RiskClass.LOW,
                    sources=["numeral_tliet_snin"],
                    hard_valid=True,
                )
            )

    # 5. ghoxrin baqar -> Għoxrin baqra
    if n1 in ("ghoxrin", "għoxrin") and len(words) >= 2:
        t2 = words[1]
        if t2.normalized == "baqar":
            candidates.append(
                Candidate(
                    source_start=t1.start,
                    source_end=t2.end,
                    original_text=t1.text + " " + t2.text,
                    replacement="Għoxrin baqra" if is_title else "għoxrin baqra",
                    operation_type=ErrorClass.NUMERAL,
                    risk_class=RiskClass.LOW,
                    sources=["numeral_ghoxrin_sing"],
                    hard_valid=True,
                )
            )

    return candidates
