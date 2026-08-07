from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher

MALTESE_FOLD = str.maketrans(
    {
        "ċ": "c",
        "Ċ": "C",
        "ġ": "g",
        "Ġ": "G",
        "ħ": "h",
        "Ħ": "H",
        "ż": "z",
        "Ż": "Z",
    }
)
WORD_RE = re.compile(r"[^\W\d_]+(?:['’-][^\W\d_]+)*", re.UNICODE)
PUNCT_RE = re.compile(r"[.!?,;:]")
ENGLISH_HINTS = {
    "air",
    "baby",
    "cashier",
    "carer",
    "container",
    "full",
    "grocery",
    "mobile",
    "owner",
    "partner",
    "please",
    "police",
    "security",
    "seat",
    "support",
    "teacher",
    "time",
    "toilet",
    "transport",
    "update",
}


def normalize_for_analysis(text: str) -> str:
    return unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")


def fold_maltese(text: str) -> str:
    text = normalize_for_analysis(text)
    text = text.replace("GĦ", "GH").replace("Għ", "Gh").replace("għ", "gh")
    return text.translate(MALTESE_FOLD)


def _has_double_change(noisy: str, clean: str) -> bool:
    doubled = re.compile(r"([bcdfgġgħhħjklmnpqrstvwxxżzċ])\1", re.IGNORECASE)
    return bool(doubled.search(noisy)) != bool(doubled.search(clean))


def infer_error_tags(noisy: str, clean: str) -> list[str]:
    noisy_n = normalize_for_analysis(noisy)
    clean_n = normalize_for_analysis(clean)
    if noisy_n == clean_n:
        return ["unchanged_clean_text"]

    tags: set[str] = set()
    noisy_words = WORD_RE.findall(noisy_n)
    clean_words = WORD_RE.findall(clean_n)

    if noisy_n.casefold() == clean_n.casefold():
        tags.add("capitalization")
    if fold_maltese(noisy_n).casefold() == fold_maltese(clean_n).casefold():
        tags.add("missing_maltese_diacritics")
    if clean_n.casefold().count("għ") > noisy_n.casefold().count("għ"):
        tags.add("għ_restoration")
    if any(ch in noisy_n + clean_n for ch in "hħHĦ") and noisy_n != clean_n:
        if fold_maltese(noisy_n).casefold() == fold_maltese(clean_n).casefold():
            tags.add("h_ħ_confusion")
    for plain, marked, label in (
        ("g", "ġ", "g_ġ"),
        ("c", "ċ", "c_ċ"),
        ("z", "ż", "z_ż"),
    ):
        if plain in noisy_n.casefold() and marked in clean_n.casefold():
            tags.add(label)
    if _has_double_change(noisy_n, clean_n):
        tags.add("consonant_doubling")
    if noisy_n.count("'") + noisy_n.count("’") != clean_n.count("'") + clean_n.count("’"):
        tags.add("apostrophe")
    if noisy_n.count("-") != clean_n.count("-"):
        tags.add("hyphenation")
    if len(noisy_words) != len(clean_words):
        tags.add("word_merge" if len(clean_words) < len(noisy_words) else "word_split")
    if re.sub(r"\s+", "", noisy_n) == re.sub(r"\s+", "", clean_n) and noisy_n != clean_n:
        tags.add("spacing")
    if PUNCT_RE.sub("", noisy_n) == PUNCT_RE.sub("", clean_n) and noisy_n != clean_n:
        tags.add("punctuation")
    if any(word.casefold() in ENGLISH_HINTS for word in noisy_words + clean_words):
        tags.add("english_maltese_mixed")
    if re.search(r"\d", noisy_n + clean_n):
        tags.add("numbers")
    if any(ch.isupper() for ch in noisy_n + clean_n):
        tags.add("name_or_place_handling")

    matcher = SequenceMatcher(None, noisy_n, clean_n, autojunk=False)
    changed_blocks = [op for op in matcher.get_opcodes() if op[0] != "equal"]
    if any(
        i2 - i1 >= 2 and j2 - j1 >= 2
        for _, i1, i2, j1, j2 in changed_blocks
    ):
        tags.add("multiple_character_edit")
    if len(changed_blocks) > 1:
        tags.add("multiple_simultaneous_errors")
    if len(noisy_words) == len(clean_words) and noisy_words != clean_words:
        tags.add("spelling_or_morphology")
    if any(
        noisy_word.casefold() != clean_word.casefold()
        and noisy_word[:2].casefold() == clean_word[:2].casefold()
        for noisy_word, clean_word in zip(noisy_words, clean_words)
    ):
        tags.add("suffix_or_inflection")
    return sorted(tags or {"contextual_or_lexical"})


def suspicious_reasons(noisy: str, clean: str) -> list[str]:
    reasons: list[str] = []
    if not clean:
        reasons.append("empty_output")
    if "\ufffd" in noisy + clean or re.search(r"(?:Ã.|Â.|â€)", noisy + clean):
        reasons.append("malformed_unicode")
    if re.search(r"-\n\w", clean):
        reasons.append("possible_visual_line_wrap_inside_token")
    if "*" in clean or re.search(r"^\s*\[.*\]\s*$", clean, re.DOTALL):
        reasons.append("review_placeholder_in_output")
    if "\n" not in clean and re.search(r"\b\w+/\w+\b", clean):
        reasons.append("multiple_alternatives_in_expected_output")
    if len(noisy) > 0:
        ratio = len(clean) / len(noisy)
        if ratio > 1.8 or ratio < 0.55:
            reasons.append("large_length_ratio")
    if clean.endswith(" .") or clean.endswith(" ?") or clean.endswith(" !"):
        reasons.append("space_before_terminal_punctuation")
    return reasons
