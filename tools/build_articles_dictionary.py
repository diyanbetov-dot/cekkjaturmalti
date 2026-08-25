from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_DICS = ROOT / "dics" / "basedics"
OUTPUT = BASE_DICS / "articles.dic"

ARTICLE_ENDINGS = ("l", "ċ", "d", "n", "r", "s", "t", "x", "z", "ż")
PREPOSITIONS = (
    ("bi", "bi"),
    ("fi", "fi"),
    ("ma’", "ma"),
    ("minn", "mi"),
    ("ta’", "ta"),
    ("għal", "għa"),
    ("bħal", "bħa"),
    ("ġo", "ġo"),
    ("sa", "sa"),
    ("lil", "li"),
)


def surface(line: str) -> str:
    return line.split("/", 1)[0].strip()


def tag(line: str) -> str:
    return line.split("/", 1)[1] if "/" in line else ""


def build_entries() -> list[str]:
    entries: list[str] = []

    for bare, stem in PREPOSITIONS:
        entries.append(f"{bare}/PREP-NO")
        for ending in ARTICLE_ENDINGS:
            article_stem = "lil" if bare == "lil" and ending == "l" else stem
            entries.append(f"{article_stem}{ending}-/PREP-YES-{ending}")

    entries.extend(("l-/DET-YES-l", "il-/DET-YES-l"))
    entries.extend(
        f"i{ending}-/DET-YES-{ending}" for ending in ARTICLE_ENDINGS[1:]
    )

    for bare, stem in (("dan", "da"), ("din", "di")):
        entries.append(f"{bare}/DET-NO")
        entries.extend(
            f"{stem}{ending}-/DET-YES-{ending}" for ending in ARTICLE_ENDINGS
        )

    return entries


def should_move(line: str, owned: set[str]) -> bool:
    word = surface(line)
    if word not in owned:
        return False

    entry_tag = tag(line)
    if word == "din" and entry_tag.startswith("SINGNOUN"):
        return False
    if word == "sa" and entry_tag.startswith("FPART"):
        return False
    return entry_tag.startswith(("PREP", "DEFPREP", "SHORTPREP", "IART", "ART", "DET", "ADVERB", "SINGNOUN"))


def main() -> None:
    entries = build_entries()
    owned = {surface(line) for line in entries}

    for path in sorted(BASE_DICS.glob("*.dic")):
        if path == OUTPUT:
            continue
        lines = path.read_text(encoding="utf-8-sig").splitlines()
        retained = [line for line in lines if not should_move(line, owned)]
        if retained != lines:
            path.write_text("\n".join(retained).rstrip() + "\n", encoding="utf-8")

    OUTPUT.write_text("\n".join(entries) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
