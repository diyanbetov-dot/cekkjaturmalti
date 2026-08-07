"""scratch/audit_patched_sample.py
Audit the patched 100-case dataset for full syntactic & morphological compliance.
"""
from __future__ import annotations
import re
import sys
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PATCHED_PATH = ROOT / "scratch" / "patched_sample_100.txt"

def parse_patched_cases(filepath: Path):
    text = filepath.read_text(encoding="utf-8")
    blocks = re.split(r"={5,}", text)
    cases = []

    for block in blocks:
        block = block.strip()
        if not block or "CASE_ID:" not in block:
            continue
        
        case = {}
        m_id = re.search(r"CASE_ID:\s*(\S+)", block)
        case["id"] = m_id.group(1) if m_id else "UNKNOWN"
        
        m_raw = re.search(r"RAW_NOISY_INPUT:\n(.*?)\n\nTARGET_CLEAN_SENTENCE_SOURCE:", block, re.DOTALL)
        case["raw"] = m_raw.group(1).strip() if m_raw else ""

        m_target = re.search(r"TARGET_CLEAN_SENTENCE_SOURCE:\n(.*?)\n\nTARGET_NORMALIZED_FOR_ANNOTATION:", block, re.DOTALL)
        case["target"] = m_target.group(1).strip() if m_target else ""

        m_status = re.search(r"REVIEW_STATUS:\s*(\S+)", block)
        case["status"] = m_status.group(1) if m_status else "UNKNOWN"

        # Check verb features
        verbs = re.findall(r"\[UPOS:VERB\]\s*\[XPOS:.*?\]\s*\[LEMMA:.*?\]\s*\[TAM:(.*?)\]\s*\[SUBJ:(.*?)\]", block)
        case["verbs"] = verbs

        # Check suffix features
        suffixes = re.findall(r"\[(DO|IDO):(.*?)\]", block)
        case["suffixes"] = suffixes

        cases.append(case)
    return cases

if __name__ == "__main__":
    if not PATCHED_PATH.exists():
        print(f"File {PATCHED_PATH} does not exist yet.")
        sys.exit(1)

    cases = parse_patched_cases(PATCHED_PATH)
    print(f"Total parsed patched cases: {len(cases)}")
    
    status_counter = Counter(c["status"] for c in cases)
    print(f"Review status breakdown: {dict(status_counter)}")

    total_verbs = sum(len(c["verbs"]) for c in cases)
    total_suffixes = sum(len(c["suffixes"]) for c in cases)

    print(f"Explicitly featured verbs: {total_verbs}")
    print(f"DO/IDO suffix annotations: {total_suffixes}")
