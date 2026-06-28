import sys
from Essentials.app import spellchecker

def p(s): sys.stdout.buffer.write((str(s) + '\n').encode('utf-8'))

word = 'iddecidew'
norm = spellchecker._normalize_word(word)

broad_best = spellchecker._best_ranked_candidate(
    norm, stage='broad_score', score_limit=0.52,
    max_distance=spellchecker._max_distance(norm)
)
p(f'broad_best: {broad_best}')
if broad_best:
    target_vowel_sequence = spellchecker._vowel_sequence(norm)
    candidate_seq = spellchecker._vowel_sequence(broad_best.candidate)
    p(f'target_vowel_sequence: {target_vowel_sequence}')
    p(f'candidate_seq: {candidate_seq}')
    p(f'score <= 0.38: {broad_best.score <= 0.38}')
    p(f'would return: {candidate_seq == target_vowel_sequence or broad_best.score <= 0.38}')
    
    # Check the prefix recovery logic
    result_candidate = broad_best.candidate
    p(f'\nresult_candidate[:1]: {result_candidate[:1]!r}')
    p(f'normalized[:1]: {norm[:1]!r}')
    p(f'different initial: {result_candidate[:1] != norm[:1]}')
    
    prefixed = norm[:1] + result_candidate
    prefixed_norm = spellchecker._normalize_word(prefixed)
    p(f'prefixed: {prefixed!r}')
    p(f'prefixed in dict: {prefixed_norm in spellchecker.dictionary_set}')
    
    ortho = spellchecker.orthographic_generator
    sc_variants = ortho.shortcut_letter_variants(norm)
    p(f'shortcut_letter_variants: {sc_variants}')
    for sc_v in sc_variants:
        sc_n = spellchecker._normalize_word(sc_v)
        p(f'  {sc_v} -> {sc_n} in dict: {sc_n in spellchecker.dictionary_set}')
