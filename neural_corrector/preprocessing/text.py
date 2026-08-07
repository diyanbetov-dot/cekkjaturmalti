from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

PROTECTED_RE = re.compile(
    r"https?://\S+|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|\b\d+(?:[.,]\d+)*\b|"
    r"[\U0001F300-\U0001FAFF\u2600-\u27BF]+"
)


@dataclass(frozen=True)
class ProtectedText:
    text: str
    values: tuple[str, ...]

    def restore(self, value: str) -> str:
        for index, original in enumerate(self.values):
            value = value.replace(f"<P{index}>", original)
        return value


def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")


def protect_neutral_spans(text: str) -> ProtectedText:
    values: list[str] = []

    def replace(match: re.Match[str]) -> str:
        values.append(match.group(0))
        return f"<P{len(values) - 1}>"

    return ProtectedText(PROTECTED_RE.sub(replace, normalize_unicode(text)), tuple(values))

