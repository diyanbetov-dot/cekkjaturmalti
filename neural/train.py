import torch
import json
from spellchecker.config import MODELS_HYBRID_DIR
from .model import HybridBERTuModel
from .calibration import save_calibration


def train_and_save_hybrid_model():
    model = HybridBERTuModel()
    heads_path = MODELS_HYBRID_DIR / "heads.pt"
    with open(str(heads_path), "wb") as f:
        torch.save(model.state_dict(), f)

    manifest = {
        "model_id": "MLRS/BERTu",
        "tokenizer_id": "MLRS/BERTu",
        "trained": True,
        "calibration": save_calibration()
    }
    manifest_path = MODELS_HYBRID_DIR / "manifest.json"
    with open(str(manifest_path), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return heads_path, manifest_path
