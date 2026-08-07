"""neural_corrector/dataset/generate_agreement_synthetic.py

Generates synthetic training pairs for:
1. Subject-Verb Agreement (3SM, 3SF, 3PL subject nouns with matching verb forms).
2. Object Suffix Transitive Preservation (Nouns + pre-verbs + verbs with object suffixes).
3. 1st-Person Speaker Context (1S pronouns/auxiliaries + 1S verbs).
4. Dual/Plural Numeral Agreement (e.g. sebat ijiem, tliet gżejjer, erba' bozoz).
5. Preposition-Article Contractions (fil-, mill-, għall-, tal-).

Saves augmented dataset to neural_corrector/data/processed/synthetic_agreement_train.jsonl
"""
from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUTPUT_PATH = ROOT / "neural_corrector" / "data" / "processed" / "synthetic_agreement_train.jsonl"
ALL_PAIRS_PATH = ROOT / "neural_corrector" / "data" / "processed" / "all_pairs.jsonl"

# ---------------------------------------------------------------------------
# Lexical pools for systematic combinatorial generation
# ---------------------------------------------------------------------------

SUBJECT_NOUNS_3SM = [
    ("ir-raġel", ["ir-ragel", "ir ragel", "ir-ragel"]),
    ("it-tifel", ["it-tifel", "it tifel", "it-tifel"]),
    ("it-tabib", ["it-tabib", "it tabib", "it-tabib"]),
    ("l-għalliem", ["l-ghalliem", "l ghalliem", "l-ghalliem"]),
    ("l-istudent", ["l-istudent", "l istudent", "l-istudent"]),
    ("il-missier", ["il-missier", "il missier", "il-missier"]),
    ("il-kelb", ["il-kelb", "il kelb", "il-kelb"]),
    ("il-ħabib", ["il-habib", "il habib", "il-habib"]),
]

SUBJECT_NOUNS_3SF = [
    ("il-mara", ["il-mara", "il mara", "il-mara"]),
    ("it-tifla", ["it-tifla", "it tifla", "it-tifla"]),
    ("it-tabiba", ["it-tabiba", "it tabiba", "it-tabiba"]),
    ("l-għalliema", ["l-ghalliema", "l ghalliema", "l-ghalliema"]),
    ("l-istudenta", ["l-istudenta", "l istudenta", "l-istudenta"]),
    ("il-omm", ["il-omm", "il omm", "il-omm"]),
    ("il-qattusa", ["il-qattusa", "il qattusa", "il-qattusa"]),
    ("il-ħabiba", ["il-habiba", "il habiba", "il-habiba"]),
]

SUBJECT_NOUNS_3PL = [
    ("ir-rġiel", ["ir-rgiel", "ir rgiel"]),
    ("it-tfal", ["it-tfal", "it tfal"]),
    ("it-tobba", ["it-tobba", "it tobba"]),
    ("l-għalliema", ["l-ghalliema", "l ghalliema"]),
    ("l-istudenti", ["l-istudenti", "l istudenti"]),
    ("il-ġenituri", ["il-genituri", "il genituri"]),
    ("il-klieb", ["il-klieb", "il klieb"]),
    ("il-ħbieb", ["il-hbieb", "il hbieb"]),
    ("in-nies", ["in-nies", "in nies"]),
]

PRE_VERBS = [
    ("ħa", ["ha", "ħa"]),
    ("se", ["se"]),
    ("ser", ["ser"]),
]

# Intransitive Verbs by Person
VERBS_INTRANS_3SM = [
    ("jiġġennen", ["jiggennen", "jigennen", "jiġġennen"]),
    ("jaqbeż", ["jaqbez", "jaqbeż"]),
    ("jinkwieta", ["jinkwieta"]),
    ("joqgħod", ["joqghod", "joqod", "joqgħod"]),
    ("jistrieħ", ["jistrieh", "jistrieħ"]),
    ("jimxi", ["jimxi"]),
]

VERBS_INTRANS_3SF = [
    ("tiġġennen", ["tiggennen", "tigennen", "tiġġennen"]),
    ("taqbeż", ["taqbez", "taqbeż"]),
    ("tinkwieta", ["tinkwieta"]),
    ("toqgħod", ["toqghod", "toqod", "toqgħod"]),
    ("tistrieħ", ["tistrieh", "tistrieħ"]),
    ("timxi", ["timxi"]),
]

VERBS_INTRANS_3PL = [
    ("jiġġennnu", ["jiggennnu", "jigennnu", "jiġġennnu"]),
    ("jaqbżu", ["jaqbzu", "jaqbżu"]),
    ("jinkwetaw", ["jinkwetaw"]),
    ("joqogħdu", ["joqoghdu", "joqodu", "joqogħdu"]),
    ("jistrieħu", ["jistriehu", "jistrieħu"]),
    ("jimxu", ["jimxu"]),
]

VERBS_INTRANS_1S = [
    ("niġġennen", ["niggennen", "nigennen", "niġġennen"]),
    ("naqbeż", ["naqbez", "naqbeż"]),
    ("ninkwieta", ["ninkwieta"]),
    ("noqgħod", ["noqghod", "noqod", "noqgħod"]),
    ("nistrieħ", ["nistrieh", "nistrieħ"]),
    ("nimxi", ["nimxi"]),
]

# Transitive Verbs with Object Suffixes
TRANSITIVE_SUFFIX_PAIRS = [
    # (Clean Verb + Suffix, [Noisy Variants], Person/Suffix Note)
    ("tħawdu", ["thawdu", "tħawdu"], "2S/3SF verb + 3SM suffix -u"),
    ("tħawdha", ["thawdha", "tħawdha"], "2S/3SF verb + 3SF suffix -ha"),
    ("tgħinu", ["tgeinu", "tgejnu", "tgħinu"], "2S/3SF verb + 3SM suffix -u"),
    ("tgħinha", ["tgeinha", "tgejha", "tgħinha"], "2S/3SF verb + 3SF suffix -ha"),
    ("tgħinhom", ["tgeinhom", "tgħinhom"], "2S/3SF verb + 3PL suffix -hom"),
    ("tagħtihomli", ["tagtihomli", "tagħtihomli"], "2S/3SF verb + 3PL DO + 1S IDO"),
    ("tippostjahom", ["tippostjahom"], "3SF verb + 3PL DO"),
]

# 1S Speaker Prefixes
FIRST_PERSON_PREFIXES = [
    ("Jien", ["jien", "Jien"]),
    ("Kont", ["kont", "Kont"]),
    ("Kważi", ["kwazi", "Kważi"]),
    ("Meta rajtu", ["meta rajtu", "Meta rajtu"]),
    ("Manakx x'ħa naqbad nagħmel,", ["manafx xha naqbad namel", "manafx x'ha naqbad namel"]),
]

# Numeral & Dual pairs
NUMERAL_DUAL_PAIRS = [
    ("sebat ijiem", ["seba jiem", "seba ijiem", "sebat jiem"]),
    ("tliet gżejjer", ["tlett gzejjer", "tliet gzejjer", "tlett gżejjer"]),
    ("erba' bozoz", ["erba bozza", "erba bozoz", "erba' bozza"]),
    ("żewġ ulied", ["zewg ulied", "zewġ ulied"]),
    ("erba' kuġini", ["erba kugini", "erba kuġini"]),
    ("ħames snin", ["hames snin"]),
    ("ittra", ["ittra"]),
]

# Preposition-Article Contractions
CONTRACTIONS = [
    ("fil-karozza", ["fil karozza", "fil-karozza"]),
    ("mill-festa", ["mill festa", "mill-festa"]),
    ("għall-ġid", ["ghall gid", "ghall-gid", "għall-ġid"]),
    ("tal-vjaġġ", ["tal vjagg", "tal-vjagg"]),
    ("b'idejk", ["bidejk", "b idejk"]),
    ("m'għamilt", ["mghamilt", "m għamilt"]),
]


def generate_all_agreement_pairs() -> list[dict]:
    pairs = []
    pair_id = 10000

    # 1. Pattern A: Subject-Verb Agreement (3SM)
    for subj_c, subj_n_list in SUBJECT_NOUNS_3SM:
        for pv_c, pv_n_list in PRE_VERBS:
            for v_c, v_n_list in VERBS_INTRANS_3SM:
                clean = f"{subj_c.capitalize()} {pv_c} {v_c}."
                for s_n in subj_n_list:
                    for p_n in pv_n_list:
                        for v_n in v_n_list:
                            noisy = f"{s_n} {p_n} {v_n}"
                            pairs.append({
                                "id": f"syn_aggr_{pair_id}",
                                "noisy": noisy,
                                "clean": clean,
                                "source": "synthetic_subject_agreement_3sm",
                                "group": f"group_{pair_id % 500}",
                            })
                            pair_id += 1
                            # Also test incorrect 1S typo corruption (e.g. ir-ragel ha niggennen -> Ir-raġel ħa jiġġennen.)
                            for v_1s_c, v_1s_n_list in VERBS_INTRANS_1S:
                                for v_1s_n in v_1s_n_list:
                                    pairs.append({
                                        "id": f"syn_aggr_{pair_id}",
                                        "noisy": f"{s_n} {p_n} {v_1s_n}",
                                        "clean": clean,
                                        "source": "synthetic_subject_override_3sm",
                                        "group": f"group_{pair_id % 500}",
                                    })
                                    pair_id += 1

    # 2. Pattern A: Subject-Verb Agreement (3SF)
    for subj_c, subj_n_list in SUBJECT_NOUNS_3SF:
        for pv_c, pv_n_list in PRE_VERBS:
            for v_c, v_n_list in VERBS_INTRANS_3SF:
                clean = f"{subj_c.capitalize()} {pv_c} {v_c}."
                for s_n in subj_n_list:
                    for p_n in pv_n_list:
                        for v_n in v_n_list:
                            noisy = f"{s_n} {p_n} {v_n}"
                            pairs.append({
                                "id": f"syn_aggr_{pair_id}",
                                "noisy": noisy,
                                "clean": clean,
                                "source": "synthetic_subject_agreement_3sf",
                                "group": f"group_{pair_id % 500}",
                            })
                            pair_id += 1

    # 3. Pattern B: Object Suffix Preservation (e.g. ir-raġel ħa tħawdu)
    for subj_c, subj_n_list in SUBJECT_NOUNS_3SM:
        for pv_c, pv_n_list in PRE_VERBS:
            for v_c, v_n_list, note in TRANSITIVE_SUFFIX_PAIRS:
                clean = f"{subj_c.capitalize()} {pv_c} {v_c}."
                for s_n in subj_n_list:
                    for p_n in pv_n_list:
                        for v_n in v_n_list:
                            noisy = f"{s_n} {p_n} {v_n}"
                            pairs.append({
                                "id": f"syn_aggr_{pair_id}",
                                "noisy": noisy,
                                "clean": clean,
                                "source": "synthetic_object_suffix_preservation",
                                "group": f"group_{pair_id % 500}",
                            })
                            pair_id += 1

    # 4. Pattern C: First-Person Speaker Preservation (e.g. jien ħa niġġennen / ħa niġġennen)
    for prefix_c, prefix_n_list in FIRST_PERSON_PREFIXES:
        for pv_c, pv_n_list in PRE_VERBS:
            for v_c, v_n_list in VERBS_INTRANS_1S:
                clean = f"{prefix_c} {pv_c} {v_c}."
                for p_n in prefix_n_list:
                    for pv_n in pv_n_list:
                        for v_n in v_n_list:
                            noisy = f"{p_n} {pv_n} {v_n}"
                            pairs.append({
                                "id": f"syn_aggr_{pair_id}",
                                "noisy": noisy,
                                "clean": clean,
                                "source": "synthetic_first_person_preservation",
                                "group": f"group_{pair_id % 500}",
                            })
                            pair_id += 1

    # Standalone pre-verb + 1S verb (e.g. ħa niġġennen)
    for pv_c, pv_n_list in PRE_VERBS:
        for v_c, v_n_list in VERBS_INTRANS_1S:
            clean = f"{pv_c.capitalize()} {v_c}."
            for pv_n in pv_n_list:
                for v_n in v_n_list:
                    noisy = f"{pv_n} {v_n}"
                    pairs.append({
                        "id": f"syn_aggr_{pair_id}",
                        "noisy": noisy,
                        "clean": clean,
                        "source": "synthetic_standalone_1s",
                        "group": f"group_{pair_id % 500}",
                    })
                    pair_id += 1

    # 5. Pattern D: Dual & Numeral Pairs
    for clean_phrase, noisy_list in NUMERAL_DUAL_PAIRS:
        clean = f"{clean_phrase.capitalize()}."
        for noisy in noisy_list:
            pairs.append({
                "id": f"syn_aggr_{pair_id}",
                "noisy": noisy,
                "clean": clean,
                "source": "synthetic_numeral_dual",
                "group": f"group_{pair_id % 500}",
            })
            pair_id += 1

    # 6. Pattern E: Preposition-Article Contractions
    for clean_phrase, noisy_list in CONTRACTIONS:
        clean = f"{clean_phrase.capitalize()}."
        for noisy in noisy_list:
            pairs.append({
                "id": f"syn_aggr_{pair_id}",
                "noisy": noisy,
                "clean": clean,
                "source": "synthetic_contractions",
                "group": f"group_{pair_id % 500}",
            })
            pair_id += 1

    return pairs


if __name__ == "__main__":
    print("Generating synthetic agreement pairs...")
    new_pairs = generate_all_agreement_pairs()
    print(f"Generated {len(new_pairs)} synthetic agreement training pairs.")

    # Deduplicate and save
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for item in new_pairs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Saved synthetic agreement dataset to {OUTPUT_PATH}")
