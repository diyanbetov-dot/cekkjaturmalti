import sys
from Essentials.app import spellchecker

word = 'iddecidew'
normalized = spellchecker._normalize_word(word)

def p(s): sys.stdout.buffer.write((s + '\n').encode('utf-8'))

p(f'normalized: {normalized}')
p(f'in dict: {normalized in spellchecker.dictionary_set}')
p(f'len tokens: {len(spellchecker._letter_tokens(normalized))}')
p(f'max_distance: {spellchecker._max_distance(normalized)}')

ortho = spellchecker.orthographic_generator

# Check each step
p('\n--- shortcut_letters ---')
sc = ortho.correct_shortcut_letters(word)
p(f'correct_shortcut_letters: {sc}')

p('\n--- gh_priority ---')
gh = ortho.correct_gh_priority(word)
p(f'correct_gh_priority: {gh}')

p('\n--- i_ie ---')
ii = list(spellchecker._dictionary_i_ie_shortcut_variants(normalized))
p(f'_dictionary_i_ie_shortcut_variants: {ii}')

p('\n--- close_apostrophe ---')
ca = spellchecker._close_apostrophe_ranked_match(word)
p(f'_close_apostrophe_ranked_match: {ca}')

p('\n--- lexicalized_forms ---')
lf = spellchecker._lexicalized_form_variants(normalized)
p(f'_lexicalized_form_variants: {lf}')

p('\n--- pattern_repairs ---')
pr = spellchecker._pattern_repair_variants(normalized)
p(f'_pattern_repair_variants: {pr}')

p('\n--- correct_strict ---')
cs2 = ortho.correct_strict(word)
p(f'correct_strict: {cs2}')

p('\n--- suffix_generator.correct_suffix ---')
cs = spellchecker.suffix_generator.correct_suffix(word)
p(f'correct_suffix: {cs}')

p('\n--- suffix_parse_guard ---')
hsp = spellchecker.suffix_generator.has_suffix_parse(word)
p(f'has_suffix_parse: {hsp}')

p('\n--- remove_h ---')
rh = spellchecker._remove_token(normalized, 'h')
erh = spellchecker._try_exact_variants(word, rh)
p(f'remove_h exact: {erh}')

p('\n--- insert_gh ---')
igh = spellchecker._insert_token_next_to_vowels(normalized, 'għ')
eigh = spellchecker._try_exact_variants(word, igh)
p(f'insert_gh exact: {eigh}')

p('\n--- insert_h ---')
ih = spellchecker._insert_token_next_to_vowels(normalized, 'h')
eih = spellchecker._try_exact_variants(word, ih)
p(f'insert_h exact: {eih}')

p('\n--- same_vowel_best (Stage 5) ---')
target_vowel_count = spellchecker._count_vowels(normalized)
same_vowel_best = spellchecker._best_ranked_candidate(
    normalized, stage='same_vowel_count',
    candidate_filter=lambda c: spellchecker.word_vowel_counts[c] == target_vowel_count,
    score_limit=0.48, max_distance=spellchecker._max_distance(normalized)
)
p(f'same_vowel_best: {same_vowel_best}')

p('\n--- broad_best (Stage 8) ---')
broad_best = spellchecker._best_ranked_candidate(
    normalized, stage='broad_score', score_limit=0.52,
    max_distance=spellchecker._max_distance(normalized)
)
p(f'broad_best: {broad_best}')
