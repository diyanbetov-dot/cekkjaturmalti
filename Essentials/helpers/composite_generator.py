import re
from typing import Iterable, List, Set, Tuple

class MalteseCompositeGenerator:
    """
    Composite Candidate Generator for Maltese spellchecking.

    Combines multiple orthographic transformations (apostrophe compounds,
    gh/ħ/h grapheme repairs, doubled letter adjustments, and generalized verb suffix repairs)
    in a unified pass when single-edit candidate generators return empty.
    """

    APOSTROPHE_PREFIXES = ("b'", "f'", "t'", "x'", "m'", "d'", "l'", "ta'", "ma'")
    # Removed ("h", "għ") per requirement. Single 'h' maps only to 'ħ'.
    GRAPHEME_REPLACEMENTS = [
        ("gh", "għ"),
        ("h", "ħ"),
        ("z", "ż"),
        ("g", "ġ"),
        ("c", "ċ"),
    ]
    SUFFIX_REWRITES = [
        ("ahha", "agħha"),
        ("ahħa", "agħha"),
        ("aħha", "agħha"),
        ("aħħa", "agħha"),
        ("ahhom", "agħhom"),
    ]

    def __init__(self, spellchecker):
        self.spellchecker = spellchecker

    def generate_candidates(self, word: str) -> List[str]:
        """
        Generates candidates by composing multiple orthographic transformations.
        Returns a list of valid dictionary-recognized candidate surfaces in priority order.
        """
        if not word or len(word) < 2:
            return []

        norm = self.spellchecker._normalize_word(word)
        results: List[str] = []
        seen: Set[str] = set()

        def add_candidate(cand: str):
            if cand and cand not in seen:
                seen.add(cand)
                if (
                    self.spellchecker._is_recognized_surface(cand)
                    or getattr(self.spellchecker, "_valid_generated_surface", lambda x: False)(cand)
                ):
                    results.append(cand)

        # 1. Generalized Suffix Rewrites (narhom -> narahom, narom -> narahom, nitfahhom -> nitfagħhom, etc.)
        for old_sfx, new_sfx in self.SUFFIX_REWRITES:
            if norm.endswith(old_sfx):
                base_stem = norm[:-len(old_sfx)]
                stem_variants = self._grapheme_variants(base_stem)
                for var in stem_variants:
                    cand = var + new_sfx
                    add_candidate(cand)

        # Generalized -hom / -om connecting vowel repairs
        if norm.endswith("hom") or norm.endswith("om"):
            ending_len = 3 if norm.endswith("hom") else 2
            stem = norm[:-ending_len]
            if stem:
                stem_variants = self._grapheme_variants(stem)
                for var in stem_variants:
                    add_candidate(var + "ahom")
                    add_candidate(var + "agħhom")
                    if norm.endswith("om"):
                        add_candidate(var + "hom")

        # 2. Apostrophe compound prefix + Grapheme / Doubled letter repair
        for prefix in ("b", "f", "t", "x", "m", "d"):
            if norm.startswith(prefix) and len(norm) > 2:
                tail = norm[1:]
                # Check prefix + apostrophe + grapheme variants of tail
                for tail_var in self._grapheme_variants(tail):
                    cand = f"{prefix}'{tail_var}"
                    add_candidate(cand)
                    # Also try doubled letter variations of tail_var
                    for doubled in self._doubled_letter_variants(tail_var):
                        cand_doubled = f"{prefix}'{doubled}"
                        add_candidate(cand_doubled)

        # 3. Multi-grapheme replacement composition (e.g., gh + h in same word)
        multi_variants = self._multi_grapheme_variants(norm)
        for var in multi_variants:
            add_candidate(var)

        return results

    def _grapheme_variants(self, stem: str) -> List[str]:
        variants = [stem]
        for src, tgt in self.GRAPHEME_REPLACEMENTS:
            if src in stem:
                variants.append(stem.replace(src, tgt))
        return list(dict.fromkeys(variants))

    def _doubled_letter_variants(self, stem: str) -> List[str]:
        variants = []
        # Deduplicate consecutive identical consonants
        dedup = re.sub(r'([bdfgjklmnprstvwzżġċħ])\1+', r'\1', stem)
        if dedup != stem:
            variants.append(dedup)
        return variants

    def _multi_grapheme_variants(self, word: str) -> List[str]:
        variants = [word]
        # Replace gh -> għ, h -> ħ simultaneously (NOT h -> għ)
        v1 = word.replace("gh", "għ").replace("h", "ħ")
        if v1 != word:
            variants.append(v1)
        v2 = word.replace("z", "ż").replace("h", "ħ")
        if v2 != word:
            variants.append(v2)
        v3 = word.replace("g", "ġ").replace("h", "ħ")
        if v3 != word:
            variants.append(v3)
        return list(dict.fromkeys(variants))
