import unicodedata

VOWELS = set("aeiouàèìòùáéíóúâêîôû")
SUN_CONSONANTS = {"ċ", "d", "n", "r", "s", "t", "x", "z", "ż"}


def normalize_word(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text).casefold()
    text = text.replace("\u2019", "'").replace("\u2018", "'").replace("\u02bc", "'")
    return text


def get_maltese_graphemes(text: str) -> list[str]:
    norm = normalize_word(text)
    graphemes = []
    i = 0
    while i < len(norm):
        if i + 1 < len(norm) and norm[i : i + 2] in ("għ", "ie"):
            graphemes.append(norm[i : i + 2])
            i += 2
        else:
            graphemes.append(norm[i])
            i += 1
    return graphemes


def determine_casing(text: str) -> str:
    if not text:
        return "lower"
    if text.isupper():
        return "upper"
    if text[0].isupper() and (len(text) == 1 or text[1:].islower()):
        return "title"
    return "lower"


def apply_casing(target: str, casing: str) -> str:
    if not target:
        return target
    if casing == "upper":
        return target.upper()
    if casing == "title":
        if target.startswith(("il-", "l-", "iċ-", "id-", "iġ-", "in-", "ir-", "is-", "it-", "ix-", "iż-")):
            parts = target.split("-", 1)
            return parts[0] + "-" + parts[1].capitalize()
        return target.capitalize()
    return target.lower()
