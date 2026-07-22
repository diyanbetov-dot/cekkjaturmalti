import sys
import os
import json
import time
from collections import Counter
from typing import List, Dict, Any

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.getcwd())

from Essentials.app import spellchecker

def benchmark_text(text: str) -> Dict[str, Any]:
    """
    Benchmarks Maltese text against the spellchecker engine.
    Measures token coverage %, total vs unique tokens, unrecognized word counts,
    and returns top unrecognized words.
    """
    if not text:
        return {"total_tokens": 0, "coverage_pct": 100.0, "unrecognized": []}

    t0 = time.time()
    res = spellchecker.correct_text_rich(text)
    elapsed = time.time() - t0

    tokens = res.get("tokens", [])
    total_tokens = len(tokens)
    if total_tokens == 0:
        return {"total_tokens": 0, "coverage_pct": 100.0, "unrecognized": []}

    recognized = 0
    unrecognized_words: List[str] = []

    for t in tokens:
        original = t.get("original", "")
        norm = spellchecker._normalize_word(original)
        is_unrecognized = t.get("unrecognized", False) or (
            not spellchecker._is_recognized_surface(norm)
            and t.get("corrected") != original
        )
        if is_unrecognized and norm:
            unrecognized_words.append(norm)
        else:
            recognized += 1

    coverage_pct = round((recognized / total_tokens) * 100, 2)
    unrecognized_counts = Counter(unrecognized_words).most_common(50)

    return {
        "elapsed_sec": round(elapsed, 3),
        "total_tokens": total_tokens,
        "recognized_tokens": recognized,
        "coverage_pct": coverage_pct,
        "unique_unrecognized_count": len(set(unrecognized_words)),
        "top_unrecognized": [
            {"word": word, "frequency": freq}
            for word, freq in unrecognized_counts
        ]
    }

if __name__ == "__main__":
    sample_corpus = """
    Nixtieq naqsam magħkom xi ħaġa li esperjenzajt meta mort ma' xi ħadd li ried jixtri karozza.
    Wara li rajna u għamilna test drive tal-karozza, tkellimna mal-bejjiegħ u kien aċċertana li il-karozza kienet ikklasifikata 4.5.
    Il-karozza kienet ikklasifikata R li tfisser li kienet involuta f'inċident u ġiet imsewwija.
    Mudguard ta' quddiem, bieba tad-driver, bieba ta' wara naħa tax-xufier mibdula, pilastru ta' bejn il-biebien u running board irrranġati u sprejjati.
    Isseqqu minni u toqogħdux fuq li jurukom imma iċċekkjaw intom ħalli ma tingidmux.
    """
    report = benchmark_text(sample_corpus)
    print("=== Lexical Coverage Benchmark Report ===")
    print(json.dumps(report, indent=2, ensure_ascii=False))

    output_path = os.path.join(os.getcwd(), "scratch", "lexical_coverage_report.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Report saved to: {output_path}")
