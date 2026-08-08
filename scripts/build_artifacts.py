import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from spellchecker.config import DATA_ARTIFACTS_DIR, MODELS_HYBRID_DIR, DATA_PROCESSED_DIR
from candidates.error_memory import REVIEWED_ERROR_MEMORY


def build_all_artifacts():
    # 1. confusions.json
    confusions = {
        "gh_to_għ": 0.95,
        "h_to_ħ": 0.85,
        "c_to_ċ": 0.80,
        "g_to_ġ": 0.80,
        "z_to_ż": 0.80,
        "initial_i_add": 0.70,
        "initial_i_remove": 0.60,
    }
    conf_path = DATA_ARTIFACTS_DIR / "confusions.json"
    conf_path.write_text(json.dumps(confusions, indent=2), encoding="utf-8")

    # 2. error_memory.json
    mem_path = DATA_ARTIFACTS_DIR / "error_memory.json"
    mem_path.write_text(json.dumps(REVIEWED_ERROR_MEMORY, indent=2), encoding="utf-8")

    # 3. char_vocab.json
    chars = " abcdefghijklmnopqrstuvwxyzàèìòùáéíóúâêîôûċġħż"
    char_vocab = {ch: i + 1 for i, ch in enumerate(chars)}
    char_vocab_path = MODELS_HYBRID_DIR / "char_vocab.json"
    char_vocab_path.write_text(json.dumps(char_vocab, indent=2), encoding="utf-8")

    # 4. feature_schema.json
    feature_schema = {
        "feature_names": [
            "hard_valid",
            "replacement_len",
            "original_len",
            "is_keep",
            "is_error_memory",
            "is_initial_i",
            "is_article_span",
            "is_numeral_span",
            "is_valid_word",
            "is_english_word",
            "is_entity_word",
            "detector_prob",
            "rank_score",
            "keep_score",
            "is_high_risk",
            "calibrated_conf",
        ],
        "feature_dim": 16,
    }
    schema_path = MODELS_HYBRID_DIR / "feature_schema.json"
    schema_path.write_text(json.dumps(feature_schema, indent=2), encoding="utf-8")

    print("All artifacts successfully created!")


if __name__ == "__main__":
    build_all_artifacts()
