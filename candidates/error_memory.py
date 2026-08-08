from typing import List, Dict
from spellchecker.schema import Candidate, ErrorClass, RiskClass, Token
from spellchecker.normalization import normalize_word

REVIEWED_ERROR_MEMORY: Dict[str, str] = {
    "mijaw": "miegħu",
    "jajdula": "jgħidulha",
    "jajdulek": "jgħidulek",
    "jorukhom": "jurukom",
    "imbad": "mbagħad",
    "gimghatejn": "ġimagħtejn",
    "xolhom": "xogħolhom",
    "kullhadd": "kulħadd",
    "inhabtu": "inħabbtu",
    "wicna": "wiċċna",
}


def generate_error_memory_candidates(token: Token) -> List[Candidate]:
    candidates: List[Candidate] = []
    norm = token.normalized
    if norm in REVIEWED_ERROR_MEMORY:
        target = REVIEWED_ERROR_MEMORY[norm]
        replacement = target.capitalize() if token.casing == "title" else target
        candidates.append(
            Candidate(
                source_start=token.start,
                source_end=token.end,
                original_text=token.text,
                replacement=replacement,
                operation_type=ErrorClass.REVIEWED_ERROR_MEMORY,
                risk_class=RiskClass.LOW,
                sources=["reviewed_error_memory"],
                hard_valid=True,
            )
        )
    return candidates
