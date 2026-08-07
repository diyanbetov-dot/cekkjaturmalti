from spellchecker.normalization import normalize_word

KNOWN_ENTITIES = {
    "john", "jack", "amy", "rabat", "mdina", "valletta", "malta", "gozo",
    "san pawl", "san ġiljan", "tas-sliema", "birgu", "isla", "bormla"
}


class EntityLexicon:
    def __init__(self, names_set=None) -> None:
        self.entities = set(KNOWN_ENTITIES)
        if names_set:
            self.entities.update(names_set)

    def is_entity(self, word: str) -> bool:
        return normalize_word(word) in self.entities

    def get_casing_candidate(self, word: str) -> str:
        norm = normalize_word(word)
        if norm in self.entities:
            if "-" in norm:
                parts = norm.split("-")
                return "-".join(p.capitalize() for p in parts)
            if " " in norm:
                parts = norm.split(" ")
                return " ".join(p.capitalize() for p in parts)
            return norm.capitalize()
        return word
