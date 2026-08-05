from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Iterator

from .schema import LatticeToken, TextSpan, TokenKind


def _is_extend(character: str) -> bool:
    codepoint = ord(character)
    return (
        bool(unicodedata.combining(character))
        or 0xFE00 <= codepoint <= 0xFE0F
        or 0x1F3FB <= codepoint <= 0x1F3FF
    )


def iter_graphemes(text: str) -> Iterator[tuple[int, int, str]]:
    """Yield practical extended grapheme clusters without changing offsets."""
    index = 0
    while index < len(text):
        start = index
        index += 1
        while index < len(text) and _is_extend(text[index]):
            index += 1
        while index < len(text) and text[index] == "\u200d":
            index += 1
            if index < len(text):
                index += 1
                while index < len(text) and _is_extend(text[index]):
                    index += 1
        yield start, index, text[start:index]


def grapheme_edit_distance(left: str, right: str) -> int:
    """Return Levenshtein distance over graphemes, not code-point offsets."""
    left_units = [cluster for _, _, cluster in iter_graphemes(unicodedata.normalize("NFC", left))]
    right_units = [cluster for _, _, cluster in iter_graphemes(unicodedata.normalize("NFC", right))]
    previous = list(range(len(right_units) + 1))
    for left_index, left_unit in enumerate(left_units, start=1):
        current = [left_index]
        for right_index, right_unit in enumerate(right_units, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_unit != right_unit),
                )
            )
        previous = current
    return previous[-1]


@dataclass(frozen=True, slots=True)
class NormalizedText:
    original: str
    normalized: str
    original_to_normalized: tuple[int, ...]
    normalized_to_original: tuple[int, ...]

    def original_span(self, normalized_start: int, normalized_end: int) -> tuple[int, int]:
        if not 0 <= normalized_start <= normalized_end <= len(self.normalized):
            raise ValueError("Normalized span is outside the text.")
        return (
            self.normalized_to_original[normalized_start],
            self.normalized_to_original[normalized_end],
        )

    def normalized_span(self, original_start: int, original_end: int) -> tuple[int, int]:
        if not 0 <= original_start <= original_end <= len(self.original):
            raise ValueError("Original span is outside the text.")
        return (
            self.original_to_normalized[original_start],
            self.original_to_normalized[original_end],
        )


def normalize_for_lattice(text: str) -> NormalizedText:
    """Create the NFC internal text and reversible UI boundary mappings.

    Only Unicode NFC normalization is performed. Apostrophes, quotation marks,
    hyphens, punctuation, whitespace, and case remain untouched.
    """
    normalized = unicodedata.normalize("NFC", text)
    original_to_normalized = tuple(
        len(unicodedata.normalize("NFC", text[:offset]))
        for offset in range(len(text) + 1)
    )
    normalized_to_original: list[int] = []
    for normalized_offset in range(len(normalized) + 1):
        eligible = [
            original_offset
            for original_offset, mapped in enumerate(original_to_normalized)
            if mapped <= normalized_offset
        ]
        normalized_to_original.append(max(eligible, default=0))
    return NormalizedText(
        original=text,
        normalized=normalized,
        original_to_normalized=original_to_normalized,
        normalized_to_original=tuple(normalized_to_original),
    )


def _cluster_kind(cluster: str) -> TokenKind | None:
    first = cluster[0]
    if first.isspace():
        return None
    category = unicodedata.category(first)
    if category.startswith("L") or category.startswith("M"):
        return TokenKind.WORD
    if category.startswith("N"):
        return TokenKind.NUMBER
    if category.startswith("P"):
        return TokenKind.PUNCTUATION
    return TokenKind.SYMBOL


def tokenize_lattice(text: NormalizedText | str) -> tuple[LatticeToken, ...]:
    """Tokenize NFC text while retaining punctuation and symbols as tokens.

    Whitespace is preserved as gaps between token character offsets. BERTu
    wordpieces are deliberately not involved in this tokenization contract.
    """
    normalized = text.normalized if isinstance(text, NormalizedText) else text
    tokens: list[LatticeToken] = []
    pending_start: int | None = None
    pending_end = 0
    pending_text: list[str] = []
    pending_kind: TokenKind | None = None

    def flush() -> None:
        nonlocal pending_start, pending_end, pending_text, pending_kind
        if pending_start is None or pending_kind is None:
            return
        token_text = "".join(pending_text)
        token = LatticeToken(
            index=len(tokens),
            kind=pending_kind,
            char_start=pending_start,
            char_end=pending_end,
            text=token_text,
        )
        if normalized[token.char_start : token.char_end] != token.text:
            raise AssertionError("Lattice token offsets do not match normalized text.")
        tokens.append(token)
        pending_start = None
        pending_text = []
        pending_kind = None

    for start, end, cluster in iter_graphemes(normalized):
        kind = _cluster_kind(cluster)
        if kind is None:
            flush()
            continue
        merge = kind in {TokenKind.WORD, TokenKind.NUMBER} and kind == pending_kind
        if not merge:
            flush()
            pending_start = start
            pending_kind = kind
        pending_end = end
        pending_text.append(cluster)
        if kind in {TokenKind.PUNCTUATION, TokenKind.SYMBOL}:
            flush()
    flush()
    return tuple(tokens)


def span_for_token_range(
    normalized_text: str,
    tokens: tuple[LatticeToken, ...],
    token_start: int,
    token_end: int,
) -> TextSpan:
    if not 0 <= token_start < token_end <= len(tokens):
        raise ValueError("Non-boundary token ranges must contain at least one token.")
    char_start = tokens[token_start].char_start
    char_end = tokens[token_end - 1].char_end
    span = TextSpan(
        char_start=char_start,
        char_end=char_end,
        token_start=token_start,
        token_end=token_end,
        text=normalized_text[char_start:char_end],
    )
    if normalized_text[span.char_start : span.char_end] != span.text:
        raise AssertionError("Span offsets do not match normalized text.")
    return span


def boundary_span(
    normalized_text: str,
    tokens: tuple[LatticeToken, ...],
    boundary_index: int,
) -> TextSpan:
    if not 0 <= boundary_index <= len(tokens):
        raise ValueError("Boundary index is outside the lattice.")
    if boundary_index == 0 and tokens:
        char_offset = tokens[boundary_index].char_start
    elif boundary_index > 0:
        char_offset = tokens[boundary_index - 1].char_end
    else:
        char_offset = len(normalized_text)
    return TextSpan(
        char_start=char_offset,
        char_end=char_offset,
        token_start=boundary_index,
        token_end=boundary_index,
        text="",
        boundary_index=boundary_index,
    )
