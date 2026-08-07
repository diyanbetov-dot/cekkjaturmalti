import json
from pathlib import Path
from typing import List, Dict, Tuple
from spellchecker.config import AI_CORRECTIONS_FILE, DATA_PROCESSED_DIR


def parse_ai_corrections_file(file_path: Path = AI_CORRECTIONS_FILE) -> List[Dict[str, str]]:
    pairs: List[Dict[str, str]] = []
    if not file_path.exists():
        return pairs

    content = file_path.read_text(encoding="utf-8", errors="ignore")
    blocks = content.split("===")

    for block in blocks:
        lines = [line.strip() for line in block.strip().splitlines() if line.strip()]
        inp = ""
        out = ""
        for i, l in enumerate(lines):
            if l == "INPUT:" and i + 1 < len(lines):
                inp = lines[i + 1]
            elif l == "OUTPUT:" and i + 1 < len(lines):
                out = lines[i + 1]
        if inp and out:
            pairs.append({"input": inp, "output": out})
    return pairs


def prepare_processed_datasets():
    pairs = parse_ai_corrections_file()
    n = len(pairs)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)

    train_data = pairs[:train_end]
    val_data = pairs[train_end:val_end]
    test_data = pairs[val_end:]

    def write_jsonl(path: Path, data: List[Dict[str, str]]):
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    write_jsonl(DATA_PROCESSED_DIR / "train.jsonl", train_data)
    write_jsonl(DATA_PROCESSED_DIR / "validation.jsonl", val_data)
    write_jsonl(DATA_PROCESSED_DIR / "test.jsonl", test_data)
    write_jsonl(DATA_PROCESSED_DIR / "clean_holdout.jsonl", val_data)
    write_jsonl(DATA_PROCESSED_DIR / "adversarial_holdout.jsonl", test_data)

    return len(train_data), len(val_data), len(test_data)
