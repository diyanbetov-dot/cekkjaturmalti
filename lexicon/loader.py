import unicodedata
from pathlib import Path
from typing import Dict, List, Set, Tuple
from spellchecker.schema import MorphAnalysis
from spellchecker.normalization import normalize_word


def parse_dic_line(line: str, file_name: str) -> Tuple[str, List[MorphAnalysis]]:
    line = line.strip()
    if not line or line.startswith("#") or "/" not in line:
        return "", []

    parts = line.split("/", 1)
    surface = unicodedata.normalize("NFC", parts[0].strip())
    payload = parts[1].strip()

    if not surface:
        return "", []

    analyses: List[MorphAnalysis] = []
    # Payload can have multiple slash-separated tags or meanings
    tag_parts = payload.split("-")
    main_tag = tag_parts[0].upper() if tag_parts else ""

    root = ""
    tam_person = ""
    gender_num = ""

    if "T-" in payload or "I-" in payload or "PERF" in payload or "IMPERF" in payload:
        for p in tag_parts:
            if p.startswith("għml") or p.startswith("b") or len(p) == 3 or p.startswith("F"):
                root = p
            elif any(k in p for k in ("PERF", "IMPERF", "IMP", "1S", "2S", "3SM", "3SF", "1P", "2P", "3P")):
                tam_person += p + "-"

    analyses.append(
        MorphAnalysis(
            surface=surface,
            lemma=surface,
            root=root,
            upos=main_tag,
            morph_tags=payload,
            source_dictionary=file_name,
            tam_person=tam_person.strip("-"),
            gender_number_type=main_tag,
            language="MT",
        )
    )
    return surface, analyses


def load_all_finaldics(finaldics_dir: Path) -> Tuple[Dict[str, List[MorphAnalysis]], Set[str], Set[str]]:
    word_map: Dict[str, List[MorphAnalysis]] = {}
    names_set: Set[str] = set()
    no_possession_set: Set[str] = set()

    if not finaldics_dir.exists():
        return word_map, names_set, no_possession_set

    for dic_path in finaldics_dir.glob("*.dic"):
        file_name = dic_path.name
        try:
            content = dic_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        for line in content.splitlines():
            surface, analyses = parse_dic_line(line, file_name)
            if not surface or not analyses:
                continue
            norm = normalize_word(surface)
            if norm not in word_map:
                word_map[norm] = []
            word_map[norm].extend(analyses)

            if "name" in file_name.lower() or any("NAME" in a.upos for a in analyses):
                names_set.add(norm)
            if "nopossession" in file_name.lower() or "nopossession" in line.lower():
                no_possession_set.add(norm)

    return word_map, names_set, no_possession_set
