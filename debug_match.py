import sys
from Essentials.app import spellchecker, WordToken

rules = spellchecker.article_phrase_rules
words = [
    WordToken("mort", 0, 4),
    WordToken("il", 5, 7),
    WordToken("bahar", 8, 13)
]
index = 1

article = rules.normalize(words[index].text).rstrip("-")
noun = rules.normalize(words[index + 1].text)
corrected_noun = noun

print("article:", article)
print("noun:", noun)
print("corrected_noun_init:", corrected_noun)
print("is_article_target_init:", rules._is_article_target(corrected_noun))

if not rules._is_article_target(corrected_noun):
    candidate = rules.normalize(spellchecker.correct_word(noun))
    print("candidate:", candidate)
    print("is_article_target_candidate:", rules._is_article_target(candidate))
    if candidate != noun and rules._is_article_target(candidate):
        corrected_noun = candidate

print("corrected_noun_after:", corrected_noun)
is_target = rules._is_article_target(corrected_noun)
print("is_target:", is_target)

if not is_target:
    print("Failed: not article target")
    sys.exit(0)

previous = words[index - 1].text if index > 0 else None
print("previous:", previous)

if article in {
    "il", "l", "ic", "iċ", "id", "in", "ir", "is", "it",
    "ix", "iz", "iż",
} or article in rules.SUN_LETTERS:
    print("In first if block!")
    is_adj = rules.is_adjective(corrected_noun)
    print("is_adjective:", is_adj)
    corrected = (
        f"l-{corrected_noun}"
        if is_adj
        else rules.corrected_article_phrase(
            article,
            corrected_noun,
            previous,
        )
    )
    print("corrected phrase:", corrected)
    choices = rules.phrase_choices(corrected_noun, previous)
    print("choices:", choices)
    res = (index, index + 2, corrected, choices)
    print("Result:", res)
else:
    print("Not in first if block!")
