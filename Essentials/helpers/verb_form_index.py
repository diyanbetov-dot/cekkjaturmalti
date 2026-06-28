from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import re
import unicodedata


VOWELS = set("aeiouàèìòù")
MEDIAL_LIQUID_OR_GUTTURAL = {"l", "m", "n", "r", "għ"}
WEAK_RADICALS = {"w", "j"}


@dataclass(frozen=True, slots=True)
class VerbFormRecord:
    word: str
    raw_tag: str
    tag_prefix: str
    root: str
    form_class: str
    tense: str
    person: str
    extra: tuple[str, ...]
    line_number: int
    stem_kind: str = ""

    @property
    def form_number(self) -> int | None:
        match = re.match(r"F(\d+)", self.form_class)
        return int(match.group(1)) if match else None

    @property
    def is_f1(self) -> bool:
        return self.form_class == "F1"

    @property
    def is_stem(self) -> bool:
        return bool(self.stem_kind)

    @property
    def is_perf(self) -> bool:
        return self.tense == "PERF"

    @property
    def is_mperf(self) -> bool:
        return self.tense == "MPERF"

    @property
    def is_imp(self) -> bool:
        return self.tense == "IMP"

    @property
    def short_tag(self) -> str:
        if self.is_stem:
            return f"{self.tag_prefix}-{self.stem_kind}-{self.root}-{self.form_class}"
        return f"{self.tag_prefix}-{self.root}-{self.form_class}"

    @property
    def full_key(self) -> str:
        bits = [
            self.tag_prefix,
        ]

        if self.is_stem:
            bits.append(self.stem_kind)

        bits.extend([
            self.root,
            self.form_class,
        ])

        if self.tense:
            bits.append(self.tense)

        if self.person:
            bits.append(self.person)

        bits.extend(self.extra)

        return "-".join(bits)


class MalteseVerbFormIndex:
    """
    Direct reader/index for verbs.dic-style entries.

    app.py shortens paradigm tags, but suffix generation needs the full
    PERF/MPERF/IMP + person data.  This class therefore reads the verb file
    directly.
    """

    TAG_PATTERN = re.compile(r"^(?:AS|T|Q|S)-")

    def __init__(
        self,
        verbs_file: Path | list[Path],
        *,
        normalizer=None,
        grapheme_splitter=None,
    ) -> None:
        if isinstance(verbs_file, (list, tuple)):
            self.verbs_files = [Path(path) for path in verbs_file]
        else:
            self.verbs_files = [Path(verbs_file)]
        self.verbs_file = self.verbs_files[0]
        self._external_normalizer = normalizer
        self._external_grapheme_splitter = grapheme_splitter

        self.by_word: dict[str, list[VerbFormRecord]] = defaultdict(list)
        self.by_short_tag: dict[str, list[VerbFormRecord]] = defaultdict(list)
        self.by_anchor: dict[str, list[VerbFormRecord]] = defaultdict(list)
        self.by_anchor_length: dict[int, list[str]] = defaultdict(list)

        self.load()

    # ------------------------------------------------------------------
    # Normalisation / graphemes
    # ------------------------------------------------------------------

    def normalize(self, word: str) -> str:
        if self._external_normalizer is not None:
            return self._external_normalizer(word)

        return (
            unicodedata.normalize("NFC", str(word).strip().lower())
            .replace("\u2019", "'")
            .replace("\u02bc", "'")
        )

    def graphemes(self, word: str) -> list[str]:
        if self._external_grapheme_splitter is not None:
            return self._external_grapheme_splitter(word)

        word = self.normalize(word)
        out: list[str] = []
        i = 0

        while i < len(word):
            if word.startswith("għ", i):
                out.append("għ")
                i += 2
            else:
                out.append(word[i])
                i += 1

        return out

    def is_vowel(self, grapheme: str) -> bool:
        return grapheme in VOWELS

    def is_consonant(self, grapheme: str) -> bool:
        return grapheme.isalpha() and not self.is_vowel(grapheme)

    def root_radicals(self, record_or_root: VerbFormRecord | str) -> list[str]:
        if isinstance(record_or_root, VerbFormRecord) and record_or_root.is_stem:
            return []

        root = (
            record_or_root.root
            if isinstance(record_or_root, VerbFormRecord)
            else record_or_root
        )

        return [
            g
            for g in self.graphemes(root)
            if g.isalpha() or g == "għ"
        ]

    def consonant_anchor(self, word: str) -> str:
        """
        Consonant skeleton with doubled consonants collapsed.
        """
        consonants = [
            token
            for token in self.graphemes(word)
            if self.is_consonant(token)
        ]

        collapsed: list[str] = []

        for token in consonants:
            if not collapsed or collapsed[-1] != token:
                collapsed.append(token)

        return "".join(collapsed)

    # ------------------------------------------------------------------
    # Root classification helpers
    # ------------------------------------------------------------------

    def second_radical(self, record: VerbFormRecord) -> str | None:
        if record.is_stem:
            return None
        radicals = self.root_radicals(record)
        return radicals[1] if len(radicals) >= 2 else None

    def third_radical(self, record: VerbFormRecord) -> str | None:
        if record.is_stem:
            return None
        radicals = self.root_radicals(record)
        return radicals[2] if len(radicals) >= 3 else None

    def is_medial_liquid_or_guttural_f1(
        self,
        record: VerbFormRecord,
    ) -> bool:
        return (
            record.is_f1
            and self.second_radical(record)
            in MEDIAL_LIQUID_OR_GUTTURAL
        )

    def is_final_weak(self, record: VerbFormRecord) -> bool:
        return self.third_radical(record) in WEAK_RADICALS

    def is_hollow(self, record: VerbFormRecord) -> bool:
        return self.second_radical(record) in WEAK_RADICALS

    def is_blocked_for_direct_object(
        self,
        record: VerbFormRecord,
    ) -> bool:
        """
        From your rule document:
            F7 (1,2,3) and F9 cannot have DO suffixes.
        """
        if record.form_class.startswith("F7"):
            return True

        if record.form_class == "F9":
            return True

        return False

    def root_class(self, record: VerbFormRecord) -> str:
        if record.is_stem:
            return f"STEM_{record.stem_kind or 'S'}"

        if self.is_medial_liquid_or_guttural_f1(record):
            return "F1_MEDIAL_LIQUID_OR_GUTTURAL"

        if self.is_final_weak(record):
            return "FINAL_WEAK"

        if self.is_hollow(record):
            return "HOLLOW"

        if record.is_f1:
            return "F1_NORMAL"

        return "OTHER"

    # ------------------------------------------------------------------
    # Loading / parsing
    # ------------------------------------------------------------------

    def load(self) -> None:
        self.by_word.clear()
        self.by_short_tag.clear()
        self.by_anchor.clear()
        self.by_anchor_length.clear()

        for verbs_file in self.verbs_files:
            if not verbs_file.exists():
                print(f"Warning: verbs file not found: {verbs_file}")
                continue

            lines = verbs_file.read_text(
                encoding="utf-8",
            ).splitlines()

            if lines and lines[0].strip().isdigit():
                lines = lines[1:]

            for line_number, line in enumerate(lines, start=1):
                record = self.parse_line(line, line_number)

                if record is not None:
                    self.add_record(record)

    def parse_line(
        self,
        line: str,
        line_number: int,
    ) -> VerbFormRecord | None:
        line = line.strip()

        if not line or line.startswith("#"):
            return None

        if "/" not in line:
            return None

        word, raw_tag = line.split("/", 1)
        word = self.normalize(word)
        raw_tag = raw_tag.strip()

        if not self.TAG_PATTERN.match(raw_tag):
            return None

        parts = raw_tag.split("-")

        if len(parts) < 3:
            return None

        tag_prefix = parts[0]

        stem_kind = ""

        if tag_prefix in {"AS", "IS"}:
            if len(parts) < 4:
                return None
            stem_kind = tag_prefix
            root = parts[1]
            form_class = "S"
            tense = parts[2]
            person = parts[3]
            extra = tuple(parts[4:]) if len(parts) >= 5 else ()
        else:
            root = parts[1]
            form_class = parts[2]
            tense = parts[3] if len(parts) >= 4 else ""
            person = parts[4] if len(parts) >= 5 else ""
            extra = tuple(parts[5:]) if len(parts) >= 6 else ()

        return VerbFormRecord(
            word=word,
            raw_tag=raw_tag,
            tag_prefix=tag_prefix,
            root=root,
            form_class=form_class,
            tense=tense,
            person=person,
            extra=extra,
            line_number=line_number,
            stem_kind=stem_kind,
        )

    def add_record(self, record: VerbFormRecord) -> None:
        self.by_word[record.word].append(record)
        self.by_short_tag[record.short_tag].append(record)

        anchor = self.consonant_anchor(record.word)
        if anchor not in self.by_anchor:
            self.by_anchor_length[len(anchor)].append(anchor)
        self.by_anchor[anchor].append(record)

    def iter_records(self):
        for records in self.by_word.values():
            for record in records:
                yield record

    def record_count(self) -> int:
        return sum(len(records) for records in self.by_word.values())

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    @staticmethod
    def _deduplicate_records(
        records: list[VerbFormRecord],
    ) -> list[VerbFormRecord]:
        seen: set[tuple[str, str]] = set()
        out: list[VerbFormRecord] = []

        for record in records:
            key = (record.word, record.raw_tag)

            if key not in seen:
                out.append(record)
                seen.add(key)

        return out

    def word_records(self, word: str) -> list[VerbFormRecord]:
        return list(self.by_word.get(self.normalize(word), []))

    def _anchor_distance(self, first: str, second: str) -> int:
        """
        Small Levenshtein distance for consonant anchors.
        """
        a = list(first)
        b = list(second)

        if not a:
            return len(b)

        if not b:
            return len(a)

        dp = [
            [0] * (len(b) + 1)
            for _ in range(len(a) + 1)
        ]

        for i in range(len(a) + 1):
            dp[i][0] = i

        for j in range(len(b) + 1):
            dp[0][j] = j

        for i in range(1, len(a) + 1):
            for j in range(1, len(b) + 1):
                cost = 0 if a[i - 1] == b[j - 1] else 1

                dp[i][j] = min(
                    dp[i - 1][j] + 1,
                    dp[i][j - 1] + 1,
                    dp[i - 1][j - 1] + cost,
                )

        return dp[-1][-1]

    def near_anchor_records(
        self,
        anchor: str,
        *,
        max_distance: int = 1,
        max_records: int = 80,
    ) -> list[VerbFormRecord]:
        anchor = self.normalize(anchor)

        if not anchor:
            return []

        matches: list[VerbFormRecord] = []

        exact = self.by_anchor.get(anchor, [])

        if exact:
            return self._deduplicate_records(exact[:max_records])

        candidate_anchors: list[str] = []
        for length in range(len(anchor) - max_distance, len(anchor) + max_distance + 1):
            if length >= 0:
                candidate_anchors.extend(self.by_anchor_length.get(length, []))

        for known_anchor in candidate_anchors:
            records = self.by_anchor[known_anchor]
            if max_distance == 1:
                s1, s2 = anchor, known_anchor
                len1, len2 = len(s1), len(s2)
                is_le_1 = False
                if len1 == len2:
                    diff = 0
                    for c1, c2 in zip(s1, s2):
                        if c1 != c2:
                            diff += 1
                            if diff > 1:
                                break
                    else:
                        is_le_1 = True
                else:
                    if len1 > len2:
                        s1, s2 = s2, s1
                        len1, len2 = len2, len1
                    i = 0
                    j = 0
                    diff = 0
                    while i < len1 and j < len2:
                        if s1[i] != s2[j]:
                            diff += 1
                            if diff > 1:
                                break
                            j += 1
                        else:
                            i += 1
                            j += 1
                    else:
                        is_le_1 = True

                if is_le_1:
                    matches.extend(records)
                    if len(matches) >= max_records:
                        break
            else:
                if self._anchor_distance(anchor, known_anchor) <= max_distance:
                    matches.extend(records)
                    if len(matches) >= max_records:
                        break

        return self._deduplicate_records(matches[:max_records])
