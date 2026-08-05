from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher

from .schema import AlignmentOperation, AlignmentRecord, TextSpan
from .text import NormalizedText, boundary_span, iter_graphemes, span_for_token_range, tokenize_lattice


def _grapheme_ratio(left: str, right: str) -> float:
    left_units = [cluster for _, _, cluster in iter_graphemes(left)]
    right_units = [cluster for _, _, cluster in iter_graphemes(right)]
    return SequenceMatcher(None, left_units, right_units, autojunk=False).ratio()


@dataclass(frozen=True, slots=True)
class AlignmentMap:
    raw: NormalizedText
    s1: NormalizedText
    records: tuple[AlignmentRecord, ...]

    def records_for_raw_span(self, span: TextSpan) -> tuple[AlignmentRecord, ...]:
        if span.is_boundary:
            return tuple(
                record
                for record in self.records
                if record.raw_span.is_boundary
                and record.raw_span.boundary_index == span.boundary_index
            )
        return tuple(
            record
            for record in self.records
            if not record.raw_span.is_boundary
            and record.raw_span.token_start < span.token_end
            and record.raw_span.token_end > span.token_start
        )

    def alignment_for_raw_span(self, span: TextSpan) -> AlignmentRecord | None:
        records = self.records_for_raw_span(span)
        if not records:
            return None
        s1_spans = tuple(s1_span for record in records for s1_span in record.s1_spans)
        exact = (
            len(records) == 1
            and records[0].raw_span.token_start == span.token_start
            and records[0].raw_span.token_end == span.token_end
        )
        return AlignmentRecord(
            raw_span=span,
            s1_spans=s1_spans,
            operation=records[0].operation if exact else AlignmentOperation.MANY_TO_MANY,
            confidence=min(record.confidence for record in records),
            ambiguous=not exact or any(record.ambiguous for record in records),
        )


def align_texts(raw: NormalizedText, s1: NormalizedText) -> AlignmentMap:
    raw_tokens = tokenize_lattice(raw)
    s1_tokens = tokenize_lattice(s1)
    matcher = SequenceMatcher(
        None,
        [token.text.casefold() for token in raw_tokens],
        [token.text.casefold() for token in s1_tokens],
        autojunk=False,
    )
    records: list[AlignmentRecord] = []
    for tag, raw_start, raw_end, s1_start, s1_end in matcher.get_opcodes():
        if raw_start == raw_end:
            raw_span = boundary_span(raw.normalized, raw_tokens, raw_start)
        else:
            raw_span = span_for_token_range(raw.normalized, raw_tokens, raw_start, raw_end)
        s1_spans = (
            ()
            if s1_start == s1_end
            else (span_for_token_range(s1.normalized, s1_tokens, s1_start, s1_end),)
        )

        if tag == "equal":
            operation = AlignmentOperation.EQUAL
            spacing_equal = raw_span.text == s1_spans[0].text
            confidence = 1.0 if spacing_equal else 0.97
            ambiguous = False
        elif tag == "insert":
            operation = AlignmentOperation.INSERT
            confidence = 0.75
            ambiguous = s1_end - s1_start > 1
        elif tag == "delete":
            operation = AlignmentOperation.DELETE
            confidence = 0.75
            ambiguous = raw_end - raw_start > 1
        else:
            raw_count = raw_end - raw_start
            s1_count = s1_end - s1_start
            operation = (
                AlignmentOperation.REPLACE
                if raw_count == s1_count == 1
                else AlignmentOperation.MANY_TO_MANY
            )
            confidence = max(0.05, _grapheme_ratio(raw_span.text, s1_spans[0].text))
            ambiguous = operation == AlignmentOperation.MANY_TO_MANY or confidence < 0.5
        records.append(
            AlignmentRecord(
                raw_span=raw_span,
                s1_spans=s1_spans,
                operation=operation,
                confidence=round(confidence, 4),
                ambiguous=ambiguous,
            )
        )
    return AlignmentMap(raw=raw, s1=s1, records=tuple(records))
