import sys
import time
from Essentials.app import spellchecker

def p(s): sys.stdout.buffer.write((str(s) + '\n').encode('utf-8'))

word = 'iddecidew'
norm = spellchecker._normalize_word(word)

# Check what _get_candidates returns now
t0 = time.time()
cands = list(spellchecker._get_candidates_cached(norm))
p(f'get_candidates: {len(cands)} in {(time.time()-t0)*1000:.1f}ms')
for c in cands[:10]:
    p(f'  {c}')

# Check what shortcut_letter_variants gives
ortho = spellchecker.orthographic_generator
sc_vars = ortho.shortcut_letter_variants(norm)
p(f'\nshortcut_letter_variants: {sc_vars}')

for sc in sc_vars:
    sc_n = spellchecker._normalize_word(sc)
    t0 = time.time()
    sc_cands = list(spellchecker._get_candidates_cached(sc_n))
    p(f'{sc} -> {len(sc_cands)} candidates in {(time.time()-t0)*1000:.1f}ms')
    for c in sc_cands[:5]:
        p(f'  {c}')

# Simulate the scoring loop of _best_ranked_candidate
typo_anchor = spellchecker._extract_consonant_anchor(norm)
p(f'\ntypo_anchor: {typo_anchor}')

all_cands = set(cands)
for sc in sc_vars:
    sc_n = spellchecker._normalize_word(sc)
    all_cands.update(spellchecker._get_candidates_cached(sc_n))

p(f'total candidates before cap: {len(all_cands)}')

# Check what anchor pre-sort does
SCORE_CAP = 48
if len(all_cands) > SCORE_CAP:
    def _anchor_dist(c):
        ca = spellchecker._extract_consonant_anchor(c)
        return spellchecker._damerau_levenshtein_distance(tuple(typo_anchor), tuple(ca))
    t0 = time.time()
    top48 = sorted(all_cands, key=_anchor_dist)[:SCORE_CAP]
    p(f'anchor pre-sort took {(time.time()-t0)*1000:.1f}ms')
    p(f'Top {len(top48)} candidates after sort:')
    for c in top48[:10]:
        ca = spellchecker._extract_consonant_anchor(c)
        dist = spellchecker._damerau_levenshtein_distance(tuple(typo_anchor), tuple(ca))
        p(f'  {c} (anchor={ca}, anchor_dist={dist})')
else:
    p(f'all {len(all_cands)} candidates (under cap)')
    for c in sorted(all_cands):
        ca = spellchecker._extract_consonant_anchor(c)
        dist = spellchecker._damerau_levenshtein_distance(tuple(typo_anchor), tuple(ca))
        p(f'  {c} (anchor={ca}, anchor_dist={dist})')
