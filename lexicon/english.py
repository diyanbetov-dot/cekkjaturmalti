import wordfreq
from rapidfuzz import process, fuzz
from spellchecker.normalization import normalize_word

MALTESE_GRAMMAR_WORDS = {
    "il", "in", "is", "it", "ir", "id", "iċ", "ic", "iġ", "ig", "ix", "iz", "iż",
    "l", "b", "f", "t", "m", "ma", "ta", "na", "le", "sa", "fi", "bi", "ki", "li",
    "għall", "ghall", "mill", "mit", "mis", "min", "mir", "mid", "miċ", "miġ", "mix",
    "del", "der", "des", "den", "dan", "din", "dawk", "dan", "dik", "huwa", "hija", "huma"
}

COMMON_ENGLISH_WORDS = {
    "parking", "mechanic", "battery", "training", "lectures", "workshops",
    "bathrooms", "series", "interviews", "deadlines", "assignment", "sales",
    "girl", "cashier", "grocery", "full", "time", "part", "wheelchair", "bolt"
}


class EnglishLexicon:
    def __init__(self) -> None:
        self.common_set = COMMON_ENGLISH_WORDS

    def is_english(self, word: str) -> bool:
        norm = normalize_word(word)
        if norm in MALTESE_GRAMMAR_WORDS:
            return False
        if norm in self.common_set:
            return True
        freq = wordfreq.word_frequency(norm, "en")
        return freq > 1e-4 and len(norm) >= 3

    def get_candidates(self, word: str, limit: int = 3) -> list[str]:
        norm = normalize_word(word)
        matches = process.extract(
            norm,
            list(self.common_set),
            scorer=fuzz.ratio,
            limit=limit,
            score_cutoff=75.0,
        )
        return [m[0] for m in matches]
