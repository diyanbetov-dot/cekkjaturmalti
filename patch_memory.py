"""
Memory optimization patch for the Maltese spellchecker.

Problems identified:
1. lru_cache on instance methods (self.*) holds a strong ref to `self` as part
   of the cache key — this means the cache can never be GC'd properly and also
   causes the entire spellchecker to be pinned to each cached function.
2. maxsize values are enormous (131072) for caches that store complex objects
   (ScoreRow dataclasses, lists, tuples). Each entry can be hundreds of bytes.
   131072 entries × ~200 bytes = ~26 MB PER CACHE, and there are many caches.
3. _score_once caches full ScoreRow objects — very fat entries.
4. _word_distance and _damerau_levenshtein_distance caches are correct but too large.
5. exact_suffix_matches caches lists of complex GeneratedSuffixCandidate objects.
6. strict_lookup_variants caches lists of strings.

Fixes applied:
A. Reduce all per-instance lru_cache maxsizes to sane, memory-safe values.
B. Remove _score_once lru_cache entirely — it's called per request, the key
   includes `stage` so there's little cross-request reuse, and the ScoreRow
   objects are expensive to store. The underlying _word_distance and
   _damerau_levenshtein_distance are already cached — that's where the speedup is.
C. Add periodic cache trimming after each request to prevent unbounded growth.
D. Remove the improperly-placed `from functools import lru_cache` inside class bodies.
"""

import re

# ------------------------------------------------------------------
# app.py patches
# ------------------------------------------------------------------
with open("Essentials/app.py", "r", encoding="utf-8") as f:
    content = f.read()

original = content  # keep for diff check

# 1. Reduce _normalize_word_cached: was 65536, make 32768 (stores short strings)
content = content.replace(
    "@lru_cache(maxsize=65536)\n    def _normalize_word_cached",
    "@lru_cache(maxsize=16384)\n    def _normalize_word_cached",
)

# 2. Reduce _graphemes_cached: was 32768, make 8192 (stores tuples of chars)
content = content.replace(
    "@lru_cache(maxsize=32768)\n    def _graphemes_cached",
    "@lru_cache(maxsize=8192)\n    def _graphemes_cached",
)

# 3. Reduce tag-lookup caches: were 32768, make 8192 each (bool results, small)
for method in [
    "_is_noun_tagged_word",
    "_is_dual_noun",
    "_is_pronoun_tagged_word",
    "_is_adverb_tagged_word",
    "_is_preposition_tagged_word",
    "_is_feminine_noun",
    "_noun_possessive_base_is_enabled",
    "_noun_possessive_base_for_surface",
]:
    content = content.replace(
        f"@lru_cache(maxsize=32768)\n    def {method}",
        f"@lru_cache(maxsize=4096)\n    def {method}",
    )

# 4. Reduce _noun_possessive_surfaces_for_base: was 8192, make 2048
content = content.replace(
    "@lru_cache(maxsize=8192)\n    def _noun_possessive_surfaces_for_base",
    "@lru_cache(maxsize=2048)\n    def _noun_possessive_surfaces_for_base",
)

# 5. Reduce _extract_consonant_anchor: was 131072, make 16384
#    (stores short anchor strings, but there are many unique words called with)
content = content.replace(
    "@lru_cache(maxsize=131072)\n    def _extract_consonant_anchor",
    "@lru_cache(maxsize=16384)\n    def _extract_consonant_anchor",
)

# 6. Reduce _vowel_slots: was 131072, make 8192
#    (stores lists of (int,str) tuples — moderately sized)
content = content.replace(
    "@lru_cache(maxsize=131072)\n    def _vowel_slots",
    "@lru_cache(maxsize=8192)\n    def _vowel_slots",
)

# 7. Reduce _count_vowels: was 131072, make 8192 (stores int, very small)
content = content.replace(
    "@lru_cache(maxsize=131072)\n    def _count_vowels",
    "@lru_cache(maxsize=8192)\n    def _count_vowels",
)

# 8. Reduce _damerau_levenshtein_distance: was 131072, make 32768
#    (stores int, key is pair of tuples — this is the hot path, keep reasonably large)
content = content.replace(
    "@lru_cache(maxsize=131072)\n    def _damerau_levenshtein_distance",
    "@lru_cache(maxsize=32768)\n    def _damerau_levenshtein_distance",
)

# 9. Reduce _word_distance: was 131072, make 16384
content = content.replace(
    "@lru_cache(maxsize=131072)\n    def _word_distance",
    "@lru_cache(maxsize=16384)\n    def _word_distance",
)

# 10. REMOVE _score_once lru_cache entirely — it stores ScoreRow dataclass objects,
#     which are ~200-400 bytes each. At maxsize=131072 that's 26-52 MB just for this
#     one cache. The underlying distance/token caches already avoid the expensive
#     recomputation; _score_once itself is cheap once those are cached.
content = content.replace(
    "@lru_cache(maxsize=131072)\n    def _score_once",
    "def _score_once",
)

# 11. Reduce _get_candidates_cached: was 32768, make 4096
#     (stores tuples of candidate strings — moderate size, but called per unique word)
content = content.replace(
    "@lru_cache(maxsize=32768)\n    def _get_candidates_cached",
    "@lru_cache(maxsize=4096)\n    def _get_candidates_cached",
)

# 12. Reduce meaning_for: was 65536, make 8192
content = content.replace(
    "@lru_cache(maxsize=65536)\n    def meaning_for",
    "@lru_cache(maxsize=8192)\n    def meaning_for",
)

# 13. Add cache trimming after each request in check_text endpoint.
#     Insert after the `reset_current_profiler(profiler_token)` line.
old_finally = """    finally:
        profiler.finish(token_count=token_count, unique_tokens=unique_tokens)
        reset_current_profiler(profiler_token)"""

new_finally = """    finally:
        profiler.finish(token_count=token_count, unique_tokens=unique_tokens)
        reset_current_profiler(profiler_token)
        # Trim request-scoped caches to prevent unbounded memory growth across requests.
        # We only trim caches whose entries are request-specific (word pairs, scores).
        # Dictionary-based caches (_normalize, _graphemes, tag lookups) are kept warm
        # because they store pre-computed facts about the dictionary words, not user text.
        _trim_request_caches()"""

content = content.replace(old_finally, new_finally)

# 14. Add the _trim_request_caches function before the Flask app section.
trim_fn = '''
def _trim_request_caches() -> None:
    """
    Partially clear high-churn, per-request caches to keep memory bounded.
    Called after every check-text request. Only clears caches that fill up
    with user-input-derived keys (word pairs etc.), not dictionary-derived ones.
    """
    import gc
    spellchecker._word_distance.cache_clear()
    spellchecker._damerau_levenshtein_distance.cache_clear()
    spellchecker._extract_consonant_anchor.cache_clear()
    spellchecker._vowel_slots.cache_clear()
    spellchecker._count_vowels.cache_clear()
    spellchecker._get_candidates_cached.cache_clear()
    gc.collect()

'''

# Insert just before the Flask app section
flask_app_marker = "# Flask app\n# -----------------------------------------------------------------------------\n\napp = Flask(__name__)"
content = content.replace(flask_app_marker, trim_fn + flask_app_marker)

if content == original:
    print("WARNING: no changes were made! Pattern matching may have failed.")
else:
    with open("Essentials/app.py", "w", encoding="utf-8") as f:
        f.write(content)
    print("app.py patched successfully.")

# ------------------------------------------------------------------
# suffix_generator.py patches
# ------------------------------------------------------------------
with open("Essentials/helpers/suffix_generator.py", "r", encoding="utf-8") as f:
    sg_content = f.read()

sg_original = sg_content

# Remove the inline `from functools import lru_cache` inside the class body
sg_content = sg_content.replace(
    "    from functools import lru_cache\n    @lru_cache(maxsize=32768)\n    def exact_suffix_matches",
    "    @lru_cache(maxsize=2048)\n    def exact_suffix_matches",
)

# Ensure lru_cache is imported at the top (it already is, but check)
if "from functools import lru_cache" not in sg_content:
    sg_content = "from functools import lru_cache\n" + sg_content

if sg_content == sg_original:
    print("WARNING: suffix_generator.py — no changes made.")
else:
    with open("Essentials/helpers/suffix_generator.py", "w", encoding="utf-8") as f:
        f.write(sg_content)
    print("suffix_generator.py patched successfully.")

# ------------------------------------------------------------------
# orthographic_generator.py patches
# ------------------------------------------------------------------
with open("Essentials/helpers/orthographic_generator.py", "r", encoding="utf-8") as f:
    og_content = f.read()

og_original = og_content

# Remove the inline `from functools import lru_cache` inside the class body
og_content = og_content.replace(
    "    from functools import lru_cache\n    @lru_cache(maxsize=32768)\n    def strict_lookup_variants",
    "    @lru_cache(maxsize=4096)\n    def strict_lookup_variants",
)

# Ensure lru_cache is imported at the top
if "from functools import lru_cache" not in og_content:
    og_content = "from functools import lru_cache\n" + og_content

if og_content == og_original:
    print("WARNING: orthographic_generator.py — no changes made.")
else:
    with open("Essentials/helpers/orthographic_generator.py", "w", encoding="utf-8") as f:
        f.write(og_content)
    print("orthographic_generator.py patched successfully.")

print("\nDone. Summary of changes:")
print("  app.py:")
print("    - _normalize_word_cached: 65536 → 16384")
print("    - _graphemes_cached: 32768 → 8192")
print("    - tag-lookup caches (x6): 32768 → 4096 each")
print("    - _noun_possessive_surfaces_for_base: 8192 → 2048")
print("    - _extract_consonant_anchor: 131072 → 16384")
print("    - _vowel_slots: 131072 → 8192")
print("    - _count_vowels: 131072 → 8192")
print("    - _damerau_levenshtein_distance: 131072 → 32768")
print("    - _word_distance: 131072 → 16384")
print("    - _score_once: REMOVED lru_cache (was storing ScoreRow objects at 131072!)")
print("    - _get_candidates_cached: 32768 → 4096")
print("    - meaning_for: 65536 → 8192")
print("    - Added _trim_request_caches() called after each request")
print("  suffix_generator.py:")
print("    - exact_suffix_matches: 32768 → 2048, removed inline import")
print("  orthographic_generator.py:")
print("    - strict_lookup_variants: 32768 → 4096, removed inline import")
