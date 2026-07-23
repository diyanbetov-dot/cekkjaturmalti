# -*- coding: utf-8 -*-
"""
Detailed Candidate Pipeline Diagnostic Script
==============================================
Runs the text through the spellchecker on the experimental branch and generates
a comprehensive diagnostic report for EVERY token where candidates were generated or scored.
"""

import sys, os, json
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.abspath("."))

# Enable corpus scoring environment variables
os.environ["SPELLCHECK_CORPUS_SCORING"] = "true"
os.environ["SPELLCHECK_CORPUS_UNIGRAM"] = "true"
os.environ["SPELLCHECK_CORPUS_BIGRAM"] = "true"
os.environ["SPELLCHECK_CORPUS_MAX_SCORE"] = "0.25"

from Essentials.app import spellchecker

INPUT_TEXT = """Gheziez hbieb,

Naf li bhalissa ghaddejin minn zmien difficcli u iebes minhabba nuqqas ta' dawl ghall hinijiet twal u eccessivi. Shana, twahhil, tidlik u hassazinijat. Naf ukoll li hawn hafna minn qed ibaghti minhabba eta, dizabilta' jew kundizzjonijiet medici. U ta' dan jiddispjacini hafna, ghaliex hadd ma haqqu jghaddi minn dan kollu!

Jin kulma nixtieq huwa haga wahda biss: forsi nkun ftit biased ukoll izda nhoss li ghandi nghidha. Jekk jghogbokom tihduwiex mal-haddiema tal-Enemalta jew il-customer care taghhom. Naf u nifhem li mhux sitwazzjoni sabiha u pjacevoli, izda dawn mghandhom l-ebda tort! Il-haddiema kollha hilom jahdmu fix-xemx u s-shana granet twal hafna u bla waqfien. U ghalhekk nixtieq li tipruvaw tifmu li huma mghandhomx tort u qed jipruvaw bil-kapacita' kollha li ghandhom jreggghu kollox lura ghan-normal. It-tort tuh lin-nies ta' fuq; dawk li jikkmandaw il-ligi!

Dawn il-haddiema li qed jahdmu lejl u nhar bix ituna l-lura l-aktar haga bazika li tezisti: il-kumdita', ma jahtu xejn!

Miniex nistenna li ma ccemplux jew ma tistaqsux ghax ghandkom dritt- izda li titfu htija fuq min mghandux huwa ingust u jwegga' hafna. Missieri flimkien mal-haddiema kollha tal-enemalta hilhom granet u ijlieli shah jahdmu bla nifs u bilkemm jistriehu jew narawhom. Ejjew ma nhallux is-sahna u r-rabja tal-mument tirkibna billi nghajjru jew imqaddru lil haddiema li jahdmu mill-qalb f'dan it-temp kifer!

Min hawn nghid grazzi lil haddiema kollha tal-enemalta u l-customer care li qed jaghmlu hilithom kollha bix isolvu problema li mghandhomx tort fiha!"""


def run_diagnostics():
    words = [w.strip(".,!?:;\"'()[]") for w in INPUT_TEXT.split() if w.strip(".,!?:;\"'()[]")]
    # Dedup preserving order
    seen = set()
    unique_words = []
    for w in words:
        norm = spellchecker._normalize_word(w)
        if norm and norm not in seen:
            seen.add(norm)
            unique_words.append(w)

    scorer = getattr(spellchecker, "corpus_scorer", None)
    reranker = getattr(spellchecker, "bertu_reranker", None)
    bertu_avail = reranker is not None and reranker.is_available()
    corpus_avail = scorer is not None and scorer.is_available()

    report_lines = []
    report_lines.append("# Candidate Pipeline Diagnostic Report\n")
    report_lines.append(f"- **Corpus Scorer Available**: {corpus_avail}")
    report_lines.append(f"- **BERTu Reranker Available**: {bertu_avail}\n")

    for orig_word in unique_words:
        norm_word = spellchecker._normalize_word(orig_word)
        if not norm_word:
            continue

        # Check if recognized dictionary word directly
        is_recognized = spellchecker._is_recognized_surface(orig_word)
        
        # Get candidates
        candidates = spellchecker.suggest(orig_word, limit=12)
        corrected = spellchecker.correct_word(orig_word)
        
        # Diagnostics report block
        report_lines.append(f"## Token: `{orig_word}` (normalized: `{norm_word}`)")
        report_lines.append(f"- **Is Directly Recognized**: {is_recognized}")
        report_lines.append(f"- **Final System Correction**: `{corrected}`")
        report_lines.append(f"- **Generated Candidates Pool**: `{candidates}`")
        
        if not candidates:
            report_lines.append("- *Note: No candidates generated (empty candidate pool).*")
            report_lines.append("\n" + "-" * 50 + "\n")
            continue

        report_lines.append("\n| Candidate | Rule / Base Confidence | Corpus Bonus | BERTu Score | Final Score | Hard Guard Status | Threshold Decision |")
        report_lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")

        for cand in candidates:
            cand_norm = spellchecker._normalize_word(cand)
            
            # 1. Rule confidence
            # Check dictionary or generator confidence
            if cand_norm in spellchecker.dictionary_set:
                rule_conf = 0.82 # high
            else:
                rule_conf = 0.62 # medium
                
            # 2. Corpus score
            corpus_bonus = scorer.score_candidate(cand) if corpus_avail else 0.0
            
            # 3. BERTu score
            bertu_score = 0.0
            if bertu_avail:
                b_scores = reranker.score_candidates(sentence=orig_word, token_index=0, candidates=[cand])
                bertu_score = b_scores.get(cand, 0.0)

            # 4. Final score
            final_score = min(1.0, round(rule_conf + corpus_bonus, 4))

            # 5. Hard guard status
            hard_guard = "PASSED"
            if cand_norm.endswith("ħek") and not any(m in norm_word for m in ("h", "ħ", "għ", "gh")):
                hard_guard = "REJECTED (Guttural Guard)"
            elif spellchecker._is_implausible_vowel_swap(norm_word, cand_norm):
                hard_guard = "REJECTED (Implausible Vowel Swap)"

            # 6. Threshold decision
            threshold_status = "ACCEPTED" if cand == corrected else "REJECTED / LOWER RANK"

            report_lines.append(
                f"| `{cand}` | {rule_conf:.2f} | +{corpus_bonus:.4f} | {bertu_score:.4f} | {final_score:.4f} | {hard_guard} | {threshold_status} |"
            )

        report_lines.append("\n" + "-" * 50 + "\n")

    output_path = Path("Other/docs/detailed_candidate_diagnostic_report.md")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"Diagnostic report written successfully to: {output_path}")

if __name__ == "__main__":
    run_diagnostics()
