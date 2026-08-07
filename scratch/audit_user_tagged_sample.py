"""scratch/audit_user_tagged_sample.py
Audit all 100 cases provided in the user's tagged dataset sample.
Checks:
- Case count and structural integrity
- Tag distribution (UNKNOWN_MORPH, NOUN_OR_OTHER:REVIEW, UNRESOLVED_AGREEMENT)
- Consistency in target spelling, punctuation, hyphenation
- Verb suffix annotations (IDO, DO, STEM)
- Potential issues/hiccups for training
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

# Read the file content from the user's prompt text
# We will create a local copy of the prompt's content in scratch/tagged_sample_100.txt
SAMPLE_PATH = ROOT / "scratch" / "tagged_sample_100.txt"

def parse_cases(filepath: Path):
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
        
        m_raw = re.search(r"RAW_NOISY_INPUT:\n(.*?)\n\nNOISY_TOKEN_ANALYSIS:", block, re.DOTALL)
        case["raw"] = m_raw.group(1).strip() if m_raw else ""

        m_target = re.search(r"TARGET_CLEAN_SENTENCE:\n(.*?)\n\nTARGET_TOKEN_ANALYSIS:", block, re.DOTALL)
        case["target"] = m_target.group(1).strip() if m_target else ""

        # Extract tags
        noisy_tags = re.findall(r"\[(.*?)\]", re.search(r"NOISY_TOKEN_ANALYSIS:(.*?)\n\nTARGET_CLEAN_SENTENCE:", block, re.DOTALL).group(1) if "NOISY_TOKEN_ANALYSIS:" in block else "")
        target_tags = re.findall(r"\[(.*?)\]", re.search(r"TARGET_TOKEN_ANALYSIS:(.*?)\n\nRELATIONS:", block, re.DOTALL).group(1) if "TARGET_TOKEN_ANALYSIS:" in block else "")
        
        case["noisy_tags"] = noisy_tags
        case["target_tags"] = target_tags

        # Extract errors
        m_errs = re.search(r"ERROR_TRANSITIONS:(.*?)\n\nREVIEW_STATUS:", block, re.DOTALL)
        case["errors"] = m_errs.group(1).strip() if m_errs else ""

        cases.append(case)
    return cases

if __name__ == "__main__":
    if not SAMPLE_PATH.exists():
        print(f"File {SAMPLE_PATH} does not exist yet. Please write it first.")
        sys.exit(1)

    cases = parse_cases(SAMPLE_PATH)
    print(f"Total parsed cases: {len(cases)}")
    
    tag_counter = Counter()
    unresolved_count = 0
    review_needed_count = 0
    verb_suffixed_count = 0
    identity_count = 0

    hiccups = []

    for c in cases:
        for tag in c["target_tags"]:
            tag_counter[tag] += 1
            if "UNRESOLVED_AGREEMENT" in tag:
                unresolved_count += 1
            if "NOUN_OR_OTHER:REVIEW" in tag:
                review_needed_count += 1
            if "VERB:SUFFIXED" in tag:
                verb_suffixed_count += 1
        
        if "[ERROR:NONE_IDENTITY]" in c["errors"]:
            identity_count += 1

        # Check for obvious typos or weirdness in targets
        target_text = c["target"]
        if "  " in target_text:
            hiccups.append((c["id"], "Double space in target_text", target_text))
        if re.search(r"\b(ikun|tista|tibqa|baqa)\s*'", target_text):
            # Checking apostrophe spacing like "tista ' " vs "tista'"
            hiccups.append((c["id"], "Space before apostrophe in verb", target_text))

    print("\n--- Tag Statistics ---")
    print(f"Total target tags: {sum(tag_counter.values())}")
    print(f"NOUN_OR_OTHER:REVIEW count: {review_needed_count}")
    print(f"UNRESOLVED_AGREEMENT count: {unresolved_count}")
    print(f"VERB:SUFFIXED count: {verb_suffixed_count}")
    print(f"Identity cases (no errors): {identity_count}")

    print("\nMost common tags:")
    for tag, count in tag_counter.most_common(15):
        print(f"  {tag:<35}: {count}")

    print(f"\n--- Potential Hiccups / Inconsistencies ({len(hiccups)}) ---")
    for case_id, issue, snippet in hiccups[:20]:
        print(f"  [{case_id}] {issue}: {snippet!r}")
