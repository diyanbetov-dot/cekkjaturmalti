import json
from spellchecker.config import MODELS_HYBRID_DIR


def save_calibration(manifest: dict = None):
    if manifest is None:
        manifest = {
            "low_risk_threshold": 0.5,
            "medium_risk_threshold": 0.75,
            "high_risk_threshold": 0.95,
            "precision_target": 0.99,
        }
    path = MODELS_HYBRID_DIR / "calibration.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_calibration() -> dict:
    path = MODELS_HYBRID_DIR / "calibration.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return save_calibration()
