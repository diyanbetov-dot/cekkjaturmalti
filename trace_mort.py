import sys
import time
from Essentials.app import spellchecker

calls = []

def make_wrapper(name, func):
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        res = func(*args, **kwargs)
        dt = (time.perf_counter() - t0) * 1000
        calls.append((name, args, dt, res))
        return res
    return wrapper

# Wrap key methods
spellchecker.correct_word = make_wrapper("correct_word", spellchecker.correct_word)
spellchecker._correct_word_uncached = make_wrapper("_correct_word_uncached", spellchecker._correct_word_uncached)
spellchecker.suggest = make_wrapper("suggest", spellchecker.suggest)
spellchecker.ambiguity_choices = make_wrapper("ambiguity_choices", spellchecker.ambiguity_choices)
spellchecker._get_candidates = make_wrapper("_get_candidates", spellchecker._get_candidates)
spellchecker._best_ranked_candidate = make_wrapper("_best_ranked_candidate", spellchecker._best_ranked_candidate)
if hasattr(spellchecker, "article_phrase_rules"):
    spellchecker.article_phrase_rules.match_split_article = make_wrapper(
        "match_split_article", spellchecker.article_phrase_rules.match_split_article
    )

print("Running correct_text_rich('mort il bahar')...")
spellchecker._word_distance.cache_clear()
spellchecker._damerau_levenshtein_distance.cache_clear()
spellchecker._get_candidates_cached.cache_clear()
spellchecker._extract_consonant_anchor.cache_clear()
spellchecker._vowel_slots.cache_clear()
spellchecker._count_vowels.cache_clear()

t_start = time.perf_counter()
res = spellchecker.correct_text_rich("mort il bahar")
t_end = time.perf_counter()

print(f"\nTotal time: {(t_end - t_start)*1000:.2f}ms")
sys.stdout.buffer.write(f"Corrected text: {res['corrected_text']}\n".encode('utf-8'))

print("\nCall trace:")
for name, args, dt, ret in calls:
    clean_args = []
    for arg in args:
        if isinstance(arg, str):
            clean_args.append(f"'{arg}'")
        elif isinstance(arg, (int, float)):
            clean_args.append(str(arg))
        else:
            clean_args.append(arg.__class__.__name__)
    # print return value representation
    ret_str = repr(ret)
    if len(ret_str) > 100:
        ret_str = ret_str[:100] + "..."
    sys.stdout.buffer.write(f"  {name}({', '.join(clean_args)}) took {dt:.2f}ms -> {ret_str}\n".encode('utf-8'))
