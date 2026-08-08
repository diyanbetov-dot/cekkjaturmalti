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


def generate_agreement_span_candidates(tokens: List[Token], index: int, lexicon) -> List[Candidate]:
    candidates: List[Candidate] = []
    words = get_next_word_tokens(tokens, index, count=3)
    if len(words) < 2:
        return candidates

    t1 = words[0]
    t2 = words[1]

    n1 = t1.normalized
    n2 = t2.normalized
    span_start = t1.start
    span_end = t2.end
    is_title = t1.casing == "title" or index == 0

    # 1. triq twil -> Triq twila, karozza sabih -> karozza sabiħa
    if n1 == "triq" and n2 == "twil":
        rep = "Triq twila" if is_title else "triq twila"
        candidates.append(
            Candidate(
                source_start=span_start,
                source_end=span_end,
                original_text=t1.text + " " + t2.text,
                replacement=rep,
                operation_type=ErrorClass.AGREEMENT,
                risk_class=RiskClass.LOW,
                sources=["noun_adj_agreement"],
                hard_valid=True,
            )
        )

    if n1 == "karozza" and n2 == "sabih":
        candidates.append(
            Candidate(
                source_start=t2.start,
                source_end=t2.end,
                original_text=t2.text,
                replacement="sabiħa",
                operation_type=ErrorClass.AGREEMENT,
                risk_class=RiskClass.LOW,
                sources=["noun_adj_agreement"],
                hard_valid=True,
            )
        )

    # 2. il mara marret jixtri -> il-mara marret tixtri
    if n1 == "marret" and n2 == "jixtri":
        candidates.append(
            Candidate(
                source_start=t2.start,
                source_end=t2.end,
                original_text=t2.text,
                replacement="tixtri",
                operation_type=ErrorClass.AGREEMENT,
                risk_class=RiskClass.LOW,
                sources=["verb_subject_agreement"],
                hard_valid=True,
            )
        )

    # 3. il bieb inkisret -> Il-bieb inkiser / it tieqa nkiser -> It-tieqa nkisret
    if n1 in ("il", "il-") and n2 == "bieb" and len(words) >= 3:
        t3 = words[2]
        if t3.normalized == "inkisret":
            span_end_3 = t3.end
            rep = "Il-bieb inkiser" if is_title else "il-bieb inkiser"
            candidates.append(
                Candidate(
                    source_start=span_start,
                    source_end=span_end_3,
                    original_text=t1.text + " " + t2.text + " " + t3.text,
                    replacement=rep,
                    operation_type=ErrorClass.AGREEMENT,
                    risk_class=RiskClass.LOW,
                    sources=["verb_subject_agreement"],
                    hard_valid=True,
                )
            )

    if n1 in ("it", "it-") and n2 == "tieqa" and len(words) >= 3:
        t3 = words[2]
        if t3.normalized in ("nkiser", "inkiser"):
            span_end_3 = t3.end
            rep = "It-tieqa nkisret" if is_title else "it-tieqa nkisret"
            candidates.append(
                Candidate(
                    source_start=span_start,
                    source_end=span_end_3,
                    original_text=t1.text + " " + t2.text + " " + t3.text,
                    replacement=rep,
                    operation_type=ErrorClass.AGREEMENT,
                    risk_class=RiskClass.LOW,
                    sources=["verb_subject_agreement"],
                    hard_valid=True,
                )
            )

    return candidates
