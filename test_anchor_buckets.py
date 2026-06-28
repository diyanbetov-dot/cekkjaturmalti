import sys
from Essentials.app import spellchecker

# Build the length buckets
spellchecker.anchor_length_buckets = {}
for anchor in spellchecker.anchor_map:
    spellchecker.anchor_length_buckets.setdefault(len(anchor), []).append(anchor)
for k in list(spellchecker.anchor_length_buckets.keys()):
    spellchecker.anchor_length_buckets[k] = tuple(spellchecker.anchor_length_buckets[k])

def near_anchor_candidates_edits(anchor: str) -> set[str]:
    letters = spellchecker.anchor_letters
    splits = [(anchor[:i], anchor[i:]) for i in range(len(anchor) + 1)]
    deletes = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
    inserts = [L + c + R for L, R in splits for c in letters]
    edits = set([anchor] + deletes + transposes + replaces + inserts)
    candidates = set()
    for edit in edits:
        if edit in spellchecker.anchor_map:
            candidates.update(spellchecker.anchor_map[edit])
    return candidates

def near_anchor_candidates_buckets(anchor: str) -> set[str]:
    candidates = set()
    target_lengths = [len(anchor) - 1, len(anchor), len(anchor) + 1]
    for length in target_lengths:
        known_anchors = spellchecker.anchor_length_buckets.get(length, ())
        for known_anchor in known_anchors:
            dist = spellchecker._damerau_levenshtein_distance(
                tuple(anchor), tuple(known_anchor)
            )
            if dist <= 1:
                candidates.update(spellchecker.anchor_map[known_anchor])
    return candidates

# Test a few anchors
test_anchors = ["bhr", "gw", "nvt", "mrt", "iddeċidew", "tajru", "baħar"]
mismatches = 0
for a in test_anchors:
    res_edits = near_anchor_candidates_edits(a)
    res_buckets = near_anchor_candidates_buckets(a)
    if res_edits != res_buckets:
        print(f"Mismatch for anchor {a}:")
        print("  Edits only:", res_edits - res_buckets)
        print("  Buckets only:", res_buckets - res_edits)
        mismatches += 1
    else:
        print(f"Anchor {a} matches! ({len(res_edits)} candidates)")

if mismatches == 0:
    print("All tests passed! Edits and Buckets return identical candidates.")
else:
    sys.exit(1)
