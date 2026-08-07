import re
from typing import List
from .schema import Token
from .normalization import normalize_word, determine_casing

TOKEN_RE = re.compile(
    r"(?P<word>[A-Za-zàèìòùáéíóúâêîôûÀÈÌÒÙÁÉÍÓÚÂÊÎÔÛċġħżĊĠĦŻ]+(?:[''-][A-Za-zàèìòùáéíóúâêîôûÀÈÌÒÙÁÉÍÓÚÂÊÎÔÛċġħżĊĠĦŻ]+)*)"
    r"|(?P<number>\d+(?:\.\d+)?)"
    r"|(?P<punct>[^\w\s])"
    r"|(?P<space>\s+)"
)


def tokenize_text(text: str) -> List[Token]:
    tokens: List[Token] = []
    for m in TOKEN_RE.finditer(text):
        raw = m.group(0)
        start, end = m.span()
        if m.group("word"):
            t_type = "word"
        elif m.group("number"):
            t_type = "number"
        elif m.group("punct"):
            t_type = "punct"
        else:
            t_type = "space"

        norm = normalize_word(raw)
        casing = determine_casing(raw)
        tokens.append(
            Token(
                text=raw,
                normalized=norm,
                start=start,
                end=end,
                token_type=t_type,
                casing=casing,
            )
        )
    return tokens
