import importlib.machinery
import importlib.util
import sys
from pathlib import Path

from helpers.article_phrase_rules import ArticlePhraseSuggestion, MalteseArticlePhraseRules, WordToken, SUN_LETTERS

BASE_DIR = Path(__file__).resolve().parent
_PYC_PATH = BASE_DIR / "app_original.pyc"
_loader = importlib.machinery.SourcelessFileLoader("_compiled_app", str(_PYC_PATH))
_spec = importlib.util.spec_from_loader("_compiled_app", _loader)
_compiled = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _compiled
_loader.exec_module(_compiled)

UniversalMalteseSpellchecker = _compiled.UniversalMalteseSpellchecker

_orig_match_split_article = MalteseArticlePhraseRules.match_split_article
_orig_match_hyphenated_article_after = MalteseArticlePhraseRules.match_hyphenated_article_after
_orig_match_preposition_article_contraction = MalteseArticlePhraseRules.match_preposition_article_contraction
_orig_match_compact_preposition_article = MalteseArticlePhraseRules.match_compact_preposition_article
_orig_preposition_article_form = MalteseArticlePhraseRules.preposition_article_form
_orig_correct_word = UniversalMalteseSpellchecker.correct_word
_orig_suggest = UniversalMalteseSpellchecker.suggest
_orig_ambiguity_choices = UniversalMalteseSpellchecker.ambiguity_choices

_COMPACT_XI_PREFIXES = (
    "x'l-", "x'l", "xl", "xil-", "xir-", "xis-", "xiż-", "xiz-", "xit-", "xid-", "xin-", "xiċ-", "xix-",
)


def _patched_preposition_article_form(self, prefix, noun):
    prefix_norm = self.normalize(prefix).rstrip("-")
    if prefix_norm in {"xi", "xil"}:
        return None
    return _orig_preposition_article_form(self, prefix, noun)


def _patched_match_preposition_article_contraction(self, words, index):
    if index + 1 < len(words) and self.normalize(words[index].text) == "xi":
        return None
    return _orig_match_preposition_article_contraction(self, words, index)


def _patched_match_compact_preposition_article(self, word):
    normalized = self.normalize(word)
    if normalized.startswith(_COMPACT_XI_PREFIXES):
        return None
    return _orig_match_compact_preposition_article(self, word)


def _patched_match_split_article(self, words, index):
    if index + 1 < len(words):
        article = self.normalize(words[index].text).rstrip("-")
        noun = words[index + 1].text
        noun_norm = self.normalize(noun)
        if article in {"ghar", "għar"}:
            is_noun = self.is_noun(noun_norm)
            is_num = self.is_num(noun_norm)
            spellchecker = getattr(self, "spellchecker", None)
            if not is_noun and not is_num and spellchecker is not None:
                corrected_noun = self.normalize(spellchecker.correct_word(noun))
                if corrected_noun and corrected_noun != noun_norm:
                    noun_norm = corrected_noun
                    is_noun = self.is_noun(noun_norm)
                    is_num = self.is_num(noun_norm)
            if (is_noun or is_num) and noun_norm.startswith("r"):
                return ArticlePhraseSuggestion(start=index, end=index + 2, corrected=f"għar-{noun_norm}", choices=[])
    return _orig_match_split_article(self, words, index)


def _patched_match_hyphenated_article_after(self, word, *, previous):
    normalized = self.normalize(word)
    if "-" in normalized:
        prefix, noun = normalized.split("-", 1)
        if prefix in SUN_LETTERS and noun:
            spellchecker = getattr(self, "spellchecker", None)
            corrected_noun = self.normalize(spellchecker.correct_word(noun)) if spellchecker is not None else noun
            if corrected_noun and self._is_article_target(corrected_noun):
                corrected = f"{prefix}-{corrected_noun}"
                return ArticlePhraseSuggestion(start=0, end=1, corrected=corrected, choices=[])

    result = _orig_match_hyphenated_article_after(self, word, previous=previous)
    if result is not None:
        return result

    if "-" not in normalized:
        return None

    prefix, noun = normalized.split("-", 1)
    if prefix not in {"il", "l", "din", "dan", "iċ", "id", "in", "ir", "is", "it", "ix", "iz", "iż"}:
        return None

    spellchecker = getattr(self, "spellchecker", None)
    if spellchecker is None or not noun:
        return None

    corrected_noun = self.normalize(spellchecker.correct_word(noun))
    if not corrected_noun or corrected_noun == noun:
        return None

    candidate = f"{prefix}-{corrected_noun}"
    return _orig_match_hyphenated_article_after(self, candidate, previous=previous)


def _patched_correct_word(self, word):
    normalized = self._normalize_word(word)
    if normalized.startswith(_COMPACT_XI_PREFIXES):
        article_rules = getattr(self, "article_phrase_rules", None)
        if article_rules is not None:
            article_word = normalized[len("xi"):]
            if article_word.startswith("'"):
                article_word = article_word[1:]
            article_match = article_rules.match_hyphenated_article_after(article_word, previous=None)
            if article_match is not None:
                return self._match_capitalisation(word, f"xi {article_match.corrected}")
    return _orig_correct_word(self, word)


def _keep_close_vowel_shape(self, original_norm, candidate_norm):
    return self._count_vowels(candidate_norm) == self._count_vowels(original_norm)


def _patched_suggest(self, word, limit=8):
    suggestions = _orig_suggest(self, word, limit=limit + 4)
    normalized = self._normalize_word(word)
    filtered = []
    for suggestion in suggestions:
        suggestion_norm = self._normalize_word(suggestion)
        if suggestion_norm in {"xil-", "xil", "x'l", "x'l-"}:
            continue
        if normalized and not _keep_close_vowel_shape(self, normalized, suggestion_norm):
            continue
        filtered.append(suggestion)
        if len(filtered) >= limit:
            break
    return filtered[:limit]


def _patched_ambiguity_choices(self, original_word, corrected_word, limit=2):
    choices = _orig_ambiguity_choices(self, original_word, corrected_word, limit=limit + 2)
    original_norm = self._normalize_word(original_word)
    filtered = []
    for choice in choices:
        choice_norm = self._normalize_word(choice.get("word", ""))
        if not choice_norm:
            continue
        if original_norm and not _keep_close_vowel_shape(self, original_norm, choice_norm):
            continue
        if choice_norm in {"xil-", "xil", "x'l", "x'l-"}:
            continue
        filtered.append(choice)
        if len(filtered) >= limit:
            break
    return filtered[:limit]


MalteseArticlePhraseRules.preposition_article_form = _patched_preposition_article_form
MalteseArticlePhraseRules.match_preposition_article_contraction = _patched_match_preposition_article_contraction
MalteseArticlePhraseRules.match_compact_preposition_article = _patched_match_compact_preposition_article
MalteseArticlePhraseRules.match_split_article = _patched_match_split_article
MalteseArticlePhraseRules.match_hyphenated_article_after = _patched_match_hyphenated_article_after
UniversalMalteseSpellchecker.correct_word = _patched_correct_word
UniversalMalteseSpellchecker.suggest = _patched_suggest
UniversalMalteseSpellchecker.ambiguity_choices = _patched_ambiguity_choices

for _name, _value in _compiled.__dict__.items():
    if _name.startswith("__") and _name not in {"__doc__", "__all__"}:
        continue
    globals()[_name] = _value

app = _compiled.app
spellchecker = _compiled.spellchecker

if __name__ == "__main__":
    debug = _compiled.os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(host="127.0.0.1", port=5000, debug=debug)
