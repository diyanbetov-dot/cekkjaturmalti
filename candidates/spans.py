from typing import List
from spellchecker.schema import Candidate


def deduplicate_candidates(candidates: List[Candidate]) -> List[Candidate]:
    seen = set()
    unique: List[Candidate] = []
    for cand in candidates:
        key = (cand.source_start, cand.source_end, cand.replacement)
        if key not in seen:
            seen.add(key)
            unique.append(cand)
    return unique
