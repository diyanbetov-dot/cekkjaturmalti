from typing import List
from spellchecker.schema import Candidate


def is_overlapping(c1: Candidate, c2: Candidate) -> bool:
    return max(c1.source_start, c2.source_start) < min(c1.source_end, c2.source_end)
