from typing import List, Tuple
from spellchecker.schema import SelectedEdit


class PureRenderer:
    def render(self, raw_text: str, edits: List[SelectedEdit]) -> Tuple[str, List[dict]]:
        if not edits:
            return raw_text, []

        sorted_edits = sorted(edits, key=lambda e: e.source_span[0])
        result_chars = []
        last_idx = 0
        formatted_edits = []

        for edit in sorted_edits:
            start, end = edit.source_span
            result_chars.append(raw_text[last_idx:start])
            result_chars.append(edit.replacement)
            last_idx = end
            formatted_edits.append(
                {
                    "start": start,
                    "end": end,
                    "original": edit.original,
                    "replacement": edit.replacement,
                    "reason": edit.reason,
                    "confidence": edit.confidence,
                }
            )

        result_chars.append(raw_text[last_idx:])
        return "".join(result_chars), formatted_edits
