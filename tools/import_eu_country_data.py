"""Build country names and demonyms from the EU style-guide Annex A5 tables."""

from __future__ import annotations

import argparse
import html
import json
import re
import unicodedata
from html.parser import HTMLParser
from pathlib import Path


SOURCE_MT = (
    "https://style-guide.europa.eu/o/opportal-service/isg"
    "?resource=mt/annex-a5-list-countries-territories-currencies.html"
)
SOURCE_EN = (
    "https://style-guide.europa.eu/o/opportal-service/isg"
    "?resource=en/annex-a5-list-countries-territories-currencies.html"
)
ARTICLE_RE = re.compile(r"^(?:l|il|i[cdgnrstxzċż])-", re.IGNORECASE)
ISO_RE = re.compile(r"^[A-Z]{2}$")


def clean_text(value: str) -> str:
    value = html.unescape(value)
    value = unicodedata.normalize("NFC", value)
    value = value.replace("\u00a0", " ").replace("\u00ad", "")
    value = value.replace("\u2011", "-").replace("\u2010", "-")
    return re.sub(r"\s+", " ", value).strip()


def without_article(name: str) -> str:
    return ARTICLE_RE.sub("", clean_text(name), count=1)


class AnnexTableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.table_depth = 0
        self.in_target_table = False
        self.in_row = False
        self.in_cell = False
        self.skip_depth = 0
        self.cell_parts: list[str] = []
        self.row: list[str] = []
        self.rows: list[list[str]] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        attrs = dict(attrs)
        if tag == "table":
            classes = set(attrs.get("class", "").split())
            if not self.in_target_table and "annex-5-desktop-table" in classes:
                self.in_target_table = True
                self.table_depth = 1
                return
            if self.in_target_table:
                self.table_depth += 1
        if not self.in_target_table:
            return
        classes = set(attrs.get("class", "").split())
        if self.in_cell and ("fn-marker" in classes or tag == "sup"):
            self.skip_depth += 1
            return
        if self.skip_depth:
            self.skip_depth += 1
            return
        if tag == "tr":
            self.in_row = True
            self.row = []
        elif tag in {"td", "th"} and self.in_row:
            self.in_cell = True
            self.cell_parts = []

    def handle_endtag(self, tag: str) -> None:
        if not self.in_target_table:
            return
        if self.skip_depth:
            self.skip_depth -= 1
            return
        if tag in {"td", "th"} and self.in_cell:
            self.row.append(clean_text("".join(self.cell_parts)))
            self.in_cell = False
        elif tag == "tr" and self.in_row:
            if self.row:
                self.rows.append(self.row)
            self.in_row = False
        elif tag == "table":
            self.table_depth -= 1
            if self.table_depth == 0:
                self.in_target_table = False

    def handle_data(self, data: str) -> None:
        if self.in_target_table and self.in_cell and not self.skip_depth:
            self.cell_parts.append(data)


def parse_rows(path: Path) -> list[list[str]]:
    parser = AnnexTableParser()
    parser.feed(path.read_text(encoding="utf-8-sig", errors="replace"))
    return parser.rows


def rows_by_iso(rows: list[list[str]]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for row in rows:
        iso_index = next(
            (index for index, cell in enumerate(row) if ISO_RE.fullmatch(cell)),
            None,
        )
        if iso_index is not None:
            result[row[iso_index]] = row
    return result


def build_records(mt_rows: list[list[str]], en_rows: list[list[str]]) -> list[dict]:
    mt_by_iso = rows_by_iso(mt_rows)
    en_by_iso = rows_by_iso(en_rows)
    records: list[dict] = []

    missing = sorted(set(mt_by_iso) ^ set(en_by_iso))
    if missing:
        raise ValueError(f"Country-code mismatch between tables: {', '.join(missing)}")

    for iso, mt_row in mt_by_iso.items():
        mt_iso_index = mt_row.index(iso)
        en_row = en_by_iso[iso]
        en_iso_index = en_row.index(iso)
        official_mt = clean_text(mt_row[mt_iso_index - 2])
        english = clean_text(en_row[en_iso_index - 2])
        demonym_text = clean_text(mt_row[mt_iso_index + 2])
        demonyms = [
            clean_text(part)
            for part in demonym_text.split(",")
            if clean_text(part) and clean_text(part) not in {"-", "\u2014"}
        ]
        records.append(
            {
                "iso": iso,
                "english": english,
                "maltese": without_article(official_mt),
                "maltese_official": official_mt,
                "demonyms": demonyms,
            }
        )

    return sorted(records, key=lambda item: item["english"].casefold())


def dictionary_lines(records: list[dict]) -> list[str]:
    lines = [
        "# Generated from the official EU Interinstitutional Style Guide Annex A5.",
        f"# Maltese source: {SOURCE_MT}",
        f"# English source: {SOURCE_EN}",
    ]
    seen: set[str] = set()
    demonym_tags = ("DNYMM", "DYNMF", "DYNMPL")

    def add(line: str) -> None:
        if line not in seen:
            seen.add(line)
            lines.append(line)

    for record in records:
        english = record["english"]
        for name in (record["maltese"], record["maltese_official"]):
            if name and "/" not in name:
                add(f"{name}/MLT-PLACE-{english}")
        for index, demonym in enumerate(record["demonyms"][:3]):
            if "/" not in demonym:
                add(
                    f"{demonym}/MLT-{demonym_tags[index]}"
                    f"-someone from {english}"
                )

    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mt-html", type=Path, required=True)
    parser.add_argument("--en-html", type=Path, required=True)
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("finaldics/eu_countries.json"),
    )
    parser.add_argument(
        "--dic-output",
        type=Path,
        default=Path("finaldics/eu_countries.dic"),
    )
    args = parser.parse_args()

    records = build_records(parse_rows(args.mt_html), parse_rows(args.en_html))
    if len(records) < 200:
        raise ValueError(f"Expected at least 200 country/territory rows, got {len(records)}")

    args.json_output.write_text(
        json.dumps(
            {
                "sources": {"mt": SOURCE_MT, "en": SOURCE_EN},
                "records": records,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    args.dic_output.write_text(
        "\n".join(dictionary_lines(records)) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(records)} official country/territory records.")


if __name__ == "__main__":
    main()
