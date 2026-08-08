import json
import sys
import time
import difflib
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from spellchecker.config import DATA_PROCESSED_DIR, DATA_ARTIFACTS_DIR
from spellchecker.pipeline import SpellcheckerPipeline
from spellchecker.tokenizer import tokenize_text


def align_tokens_finely(inp_text: str, out_text: str):
    inp_tokens = tokenize_text(inp_text)
    out_tokens = tokenize_text(out_text)

    inp_words = [t for t in inp_tokens if t.token_type == "word"]
    out_words = [t for t in out_tokens if t.token_type == "word"]

    matcher = difflib.SequenceMatcher(
        None,
        [w.normalized for w in inp_words],
        [w.normalized for w in out_words]
    )
    opcodes = matcher.get_opcodes()

    gold_edits = []

    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            for k in range(i2 - i1):
                w_in = inp_words[i1 + k]
                w_out = out_words[j1 + k]
                if w_in.text != w_out.text:
                    gold_edits.append({
                        "span": (w_in.start, w_in.end),
                        "gold_src": w_in.text,
                        "gold_tgt": w_out.text
                    })
            continue

        # If 1-to-1 word replacement:
        if (i2 - i1) == 1 and (j2 - j1) == 1:
            w_in = inp_words[i1]
            w_out = out_words[j1]
            gold_edits.append({
                "span": (w_in.start, w_in.end),
                "gold_src": w_in.text,
                "gold_tgt": w_out.text
            })
            continue

        # If N-to-M word block (e.g. 2 words to 1 word or 1 word to 2 words):
        src_start = inp_words[i1].start if i1 < len(inp_words) else (inp_words[i1 - 1].end if i1 > 0 else 0)
        src_end = inp_words[i2 - 1].end if i2 > 0 else (inp_words[i1].start if i1 < len(inp_words) else len(inp_text))

        tgt_text = out_text[out_words[j1].start:out_words[j2 - 1].end] if j1 < j2 else ""

        # If i2 - i1 > 1 and j2 - j1 > 1, attempt pairwise alignment if word counts match or can be split
        if (i2 - i1) == (j2 - j1):
            for k in range(i2 - i1):
                w_in = inp_words[i1 + k]
                w_out = out_words[j1 + k]
                if w_in.text != w_out.text:
                    gold_edits.append({
                        "span": (w_in.start, w_in.end),
                        "gold_src": w_in.text,
                        "gold_tgt": w_out.text
                    })
        else:
            gold_edits.append({
                "span": (src_start, src_end),
                "gold_src": inp_text[src_start:src_end],
                "gold_tgt": tgt_text
            })

    return gold_edits


def classify_miss_reason(src: str, tgt: str) -> str:
    s_norm = src.casefold()
    t_norm = tgt.casefold()

    if "għ" in t_norm and "għ" not in s_norm:
        return "MISSING_GH_H_CHANNEL"
    if (" " in s_norm and " " not in t_norm) or (" " not in s_norm and " " in t_norm):
        return "MISSING_SPLIT_JOIN_SPAN"
    if "-" in t_norm or "'" in t_norm or "-" in s_norm or "'" in s_norm:
        return "MISSING_ARTICLE_OR_PUNCTUATION_SPAN"
    if s_norm == t_norm and src != tgt:
        return "MISSING_CAPITALIZATION"
    if len(src) >= 15 or len(tgt) >= 20:
        return "TOO_FAR_EDIT"
    return "MISSING_MORPH_OR_OTHER"


def evaluate_train_oracle():
    p = SpellcheckerPipeline()
    train_file = DATA_PROCESSED_DIR / "train.jsonl"
    if not train_file.exists():
        print(f"Error: {train_file} does not exist.")
        return

    train_records = []
    with open(train_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                train_records.append(json.loads(line.strip()))

    total_gold_edits = 0
    matched_gold_edits = 0
    missing_records = []

    t0 = time.perf_counter()

    for idx, rec in enumerate(train_records):
        inp = rec.get("input", "")
        out = rec.get("output", "")
        if inp == out:
            continue

        tokens = tokenize_text(inp)
        candidates = p.candidate_generator.generate_candidates(tokens)

        cand_map = {}
        for c in candidates:
            span = (c.source_start, c.source_end)
            if span not in cand_map:
                cand_map[span] = set()
            cand_map[span].add(c.replacement)

        gold_edits = align_tokens_finely(inp, out)

        for ge in gold_edits:
            total_gold_edits += 1
            span = ge["span"]
            gold_tgt = ge["gold_tgt"]
            gen_repls = cand_map.get(span, set())

            if gold_tgt in gen_repls or gold_tgt.casefold() in [r.casefold() for r in gen_repls]:
                matched_gold_edits += 1
            else:
                missing_records.append({
                    "record_id": idx,
                    "source_sentence": inp,
                    "target_sentence": out,
                    "span": list(span),
                    "gold_src": ge["gold_src"],
                    "gold_tgt": gold_tgt,
                    "generated_replacements": list(gen_repls),
                    "reason": classify_miss_reason(ge["gold_src"], gold_tgt)
                })

    t1 = time.perf_counter()
    recall = matched_gold_edits / total_gold_edits if total_gold_edits > 0 else 1.0

    print(f"train_gold_edits: {total_gold_edits}")
    print(f"ordinary_generated: {matched_gold_edits}")
    print(f"oracle_recall: {recall:.4f}")
    print(f"missing: {len(missing_records)}")
    print(f"evaluation_time: {t1 - t0:.2f}s")

    reasons = {}
    for mr in missing_records:
        r = mr["reason"]
        reasons[r] = reasons.get(r, 0) + 1
    print("\nmissing_by_reason:")
    for r, count in sorted(reasons.items(), key=lambda x: -x[1]):
        print(f"  {r}: {count}")

    out_missing = DATA_ARTIFACTS_DIR / "oracle_missing_train.jsonl"
    with open(out_missing, "w", encoding="utf-8") as f:
        for rec in missing_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    evaluate_train_oracle()
