from typing import Dict, List, Set, Optional
from collections import defaultdict
from rapidfuzz import process, fuzz
from spellchecker.schema import MorphAnalysis
from spellchecker.normalization import normalize_word
from spellchecker.config import FINAL_DICS_DIR
from .loader import load_all_finaldics


class LexiconIndexes:
    def __init__(self, finaldics_dir=FINAL_DICS_DIR) -> None:
        self.word_map, self.names_set, self.no_possession_set = load_all_finaldics(finaldics_dir)
        self.all_words_list = list(self.word_map.keys())

        # Build fast length buckets for O(1) candidate lookup
        self.length_buckets: Dict[int, List[str]] = defaultdict(list)
        for w in self.all_words_list:
            self.length_buckets[len(w)].append(w)

    def contains(self, word: str) -> bool:
        return normalize_word(word) in self.word_map

    def get_analyses(self, word: str) -> List[MorphAnalysis]:
        return self.word_map.get(normalize_word(word), [])

    def is_name(self, word: str) -> bool:
        return normalize_word(word) in self.names_set

    def is_no_possession(self, word: str) -> bool:
        return normalize_word(word) in self.no_possession_set

    def fuzzy_search(self, word: str, limit: int = 5, score_cutoff: float = 78.0) -> List[str]:
        norm = normalize_word(word)
        if not norm or len(norm) < 3:
            return []

        w_len = len(norm)
        pool = []
        for l in range(max(1, w_len - 2), w_len + 3):
            pool.extend(self.length_buckets.get(l, []))

        if not pool:
            return []

        matches = process.extract(
            norm,
            pool,
            scorer=fuzz.ratio,
            limit=limit,
            score_cutoff=score_cutoff,
        )
        return [match[0] for match in matches]
