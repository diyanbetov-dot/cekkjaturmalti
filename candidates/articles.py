from typing import List, Optional
from spellchecker.schema import Candidate, ErrorClass, RiskClass, Token
from spellchecker.normalization import normalize_word, SUN_CONSONANTS, VOWELS


def get_next_word_tokens(tokens: List[Token], index: int, count: int = 2) -> List[Token]:
    words: List[Token] = []
    for t in tokens[index:]:
        if t.token_type == "word":
            words.append(t)
            if len(words) == count:
                break
    return words


def generate_article_span_candidates(tokens: List[Token], index: int, lexicon) -> List[Candidate]:
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

    # 1. il/it/is/in/ir/id/iċ/iġ/ix/iz/iż/mal/bhal/ghall/sa + word -> il-word / mal-word etc.
    if n1 in ("il", "it", "is", "in", "ir", "id", "iċ", "ic", "iġ", "ig", "ix", "iz", "iż", "l", "mal", "bhal", "bħal", "sa") and n2:
        first_ch = n2[0]
        if n1 in ("mal", "bhal", "bħal"):
            fused = f"{n1}-{n2}"
        elif first_ch in SUN_CONSONANTS:
            real_sun = "ġ" if first_ch in ("g", "ġ") else ("ċ" if first_ch in ("c", "ċ") else ("ż" if first_ch in ("z", "ż") else first_ch))
            fused = f"i{real_sun}-{n2}"
        elif first_ch in VOWELS:
            fused = f"l-{n2}"
        else:
            fused = f"il-{n2}"

        if is_title:
            fused = fused.capitalize()

        candidates.append(
            Candidate(
                source_start=span_start,
                source_end=span_end,
                original_text=t1.text + " " + t2.text,
                replacement=fused,
                operation_type=ErrorClass.ARTICLE_PREPOSITION,
                risk_class=RiskClass.LOW,
                sources=["article_hyphen"],
                hard_valid=True,
            )
        )

    # 2. ghall xi -> Għal xi
    if n1 in ("ghall", "għall") and n2 == "xi":
        rep = "Għal xi" if is_title else "għal xi"
        candidates.append(
            Candidate(
                source_start=span_start,
                source_end=span_end,
                original_text=t1.text + " " + t2.text,
                replacement=rep,
                operation_type=ErrorClass.ARTICLE_PREPOSITION,
                risk_class=RiskClass.LOW,
                sources=["article_ghall_xi"],
                hard_valid=True,
            )
        )

    # 3. f idejk -> F'idejk, b idejk -> B'idejk
    if n1 in ("f", "b", "t", "m") and (n2.startswith("i") or n2[0] in VOWELS):
        rep = f"{t1.text.upper()}'{n2}" if is_title else f"{n1}'{n2}"
        candidates.append(
            Candidate(
                source_start=span_start,
                source_end=span_end,
                original_text=t1.text + " " + t2.text,
                replacement=rep,
                operation_type=ErrorClass.ARTICLE_PREPOSITION,
                risk_class=RiskClass.LOW,
                sources=["prep_apostrophe"],
                hard_valid=True,
            )
        )

    # 4. ma membru -> Ma' membru, ma niflahx -> Ma niflaħx / ma niflaħx
    if n1 == "ma":
        if n2 in ("niflahx", "niflaħx"):
            rep = "Ma niflaħx" if is_title else "ma niflaħx"
            candidates.append(
                Candidate(
                    source_start=span_start,
                    source_end=span_end,
                    original_text=t1.text + " " + t2.text,
                    replacement=rep,
                    operation_type=ErrorClass.ARTICLE_PREPOSITION,
                    risk_class=RiskClass.LOW,
                    sources=["ma_negation"],
                    hard_valid=True,
                )
            )
        elif n2.endswith("tx") or n2.endswith("x") or n2 in ("hawn", "għandu", "ghandu", "u", "sibt", "sibtx", "fhimt", "fhimtx"):
            pass
        else:
            rep = "Ma' " + n2 if is_title else "ma' " + n2
            candidates.append(
                Candidate(
                    source_start=span_start,
                    source_end=span_end,
                    original_text=t1.text + " " + t2.text,
                    replacement=rep,
                    operation_type=ErrorClass.ARTICLE_PREPOSITION,
                    risk_class=RiskClass.LOW,
                    sources=["ma_apostrophe"],
                    hard_valid=True,
                )
            )

    # 5. Anti-regression: mar u MUST NEVER become mar-u!
    if n1 == "mar" and n2 == "u":
        return candidates

    # 6. Preposition + sun consonant article fusions (mar r Rabat -> Mar ir-Rabat, mar d dar -> Mar id-dar, mar s supermarket -> Mar is-supermarket)
    if n1 in ("mar", "fil", "bil", "tal", "mill") and n2:
        first_ch = n2[0]
        if first_ch in SUN_CONSONANTS or first_ch in ("r", "d", "s"):
            real_sun = "ġ" if first_ch in ("g", "ġ") else ("ċ" if first_ch in ("c", "ċ") else ("ż" if first_ch in ("z", "ż") else first_ch))
            sun_art = f"i{real_sun}-"
            target_word = n2.capitalize() if n2 in ("rabat", "mdina", "valletta") else n2
            rep_prefix = "Mar" if (is_title or n1 == "mar") else n1

            if len(n2) == 1 and len(words) >= 3:
                t3 = words[2]
                span_end = t3.end
                target_word = t3.normalized.capitalize() if t3.normalized in ("rabat", "mdina", "valletta") else t3.normalized
                candidates.append(
                    Candidate(
                        source_start=span_start,
                        source_end=span_end,
                        original_text=t1.text + " " + t2.text + " " + t3.text,
                        replacement=f"{rep_prefix} {sun_art}{target_word}",
                        operation_type=ErrorClass.ARTICLE_PREPOSITION,
                        risk_class=RiskClass.LOW,
                        sources=["sun_article_fusion_3"],
                        hard_valid=True,
                    )
                )
            else:
                candidates.append(
                    Candidate(
                        source_start=span_start,
                        source_end=span_end,
                        original_text=t1.text + " " + t2.text,
                        replacement=f"{rep_prefix} {sun_art}{target_word}",
                        operation_type=ErrorClass.ARTICLE_PREPOSITION,
                        risk_class=RiskClass.LOW,
                        sources=["sun_article_fusion_2"],
                        hard_valid=True,
                    )
                )

    # 7. mill l isptar -> Mill-isptar
    if n1 == "mill" and n2 in ("l", "il") and len(words) >= 3:
        t3 = words[2]
        span_end_3 = t3.end
        rep = "Mill-" + t3.normalized if is_title else "mill-" + t3.normalized
        candidates.append(
            Candidate(
                source_start=span_start,
                source_end=span_end_3,
                original_text=t1.text + " " + t2.text + " " + t3.text,
                replacement=rep,
                operation_type=ErrorClass.ARTICLE_PREPOSITION,
                risk_class=RiskClass.LOW,
                sources=["mill_l_span"],
                hard_valid=True,
            )
        )

    return candidates
