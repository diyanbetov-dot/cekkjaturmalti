import sys
import time
import cProfile
import pstats
from Essentials.app import spellchecker

sentence = 'Ma waqaftx nivvotta al Amy biex tibqa hemm gew imma le in nies iddecidew'

def run_test():
    return spellchecker.correct_text_rich(sentence)

print("Starting profile...")
t0 = time.time()
result = run_test()
elapsed = time.time() - t0
sys.stdout.buffer.write(f'Time taken: {elapsed:.3f}s\n\nTokens:\n'.encode('utf-8'))
for tok in result.get('tokens', []):
    if tok.get('type') == 'word':
        line = f"  {tok['original']:20s} -> {tok['corrected']}\n"
        sys.stdout.buffer.write(line.encode('utf-8'))

print("\nProfiling (slow path with cleared caches)...")
# Clear caches to simulate cold request
spellchecker._word_distance.cache_clear()
spellchecker._damerau_levenshtein_distance.cache_clear()
spellchecker._get_candidates_cached.cache_clear()
spellchecker._extract_consonant_anchor.cache_clear()
spellchecker._vowel_slots.cache_clear()
spellchecker._count_vowels.cache_clear()
spellchecker._letter_tokens_raw.cache_clear()

cProfile.run('run_test()', 'profile_sentence')
with open('profile_sentence.txt', 'w', encoding='utf-8') as f:
    p = pstats.Stats('profile_sentence', stream=f)
    p.sort_stats('tottime').print_stats(25)
print("Profile written to profile_sentence.txt")

# Word-by-word timing
print("\nWord-by-word timing:")
for tok in result.get('tokens', []):
    if tok.get('type') == 'word':
        word = tok['original']
        spellchecker._word_distance.cache_clear()
        spellchecker._damerau_levenshtein_distance.cache_clear()
        spellchecker._get_candidates_cached.cache_clear()
        spellchecker._extract_consonant_anchor.cache_clear()
        spellchecker._vowel_slots.cache_clear()
        spellchecker._count_vowels.cache_clear()
        t0 = time.time()
        spellchecker.correct_word(word)
        ms = (time.time() - t0) * 1000
        cands = list(spellchecker._get_candidates_cached(spellchecker._normalize_word(word)))
        sys.stdout.buffer.write(f"  {word:20s}  {ms:7.1f}ms  {len(cands)} candidates\n".encode('utf-8'))
