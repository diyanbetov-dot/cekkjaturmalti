import torch
from typing import List
from spellchecker.config import MODELS_HYBRID_DIR
from spellchecker.schema import Candidate, ErrorClass, RiskClass
from .model import HybridBERTuModel
from .features import extract_candidate_features
from .calibration import load_calibration


class NeuralRuntime:
    def __init__(self) -> None:
        self.model = None
        self.loaded = False
        self.calibration = load_calibration()
        self._load()

    def _load(self):
        heads_path = MODELS_HYBRID_DIR / "heads.pt"
        try:
            self.model = HybridBERTuModel()
            if heads_path.exists():
                with open(str(heads_path), "rb") as f:
                    self.model.load_state_dict(torch.load(f, map_location="cpu"))
            self.model.eval()
            self.loaded = True
        except Exception:
            self.loaded = False

    def score_candidates(self, raw_text: str, candidates: List[Candidate]) -> List[Candidate]:
        if not candidates:
            return candidates

        # Run BERTu detector on context
        det_prob = self.predict_error_prob(raw_text)

        for cand in candidates:
            cand.detector_probability = det_prob
            feat = extract_candidate_features(cand)

            if cand.operation_type == ErrorClass.KEEP:
                cand.rank_score = 1.0
                cand.keep_score = 1.0
                cand.calibrated_confidence = 1.0
            else:
                src_bonus = 0.9 if "reviewed_error_memory" in cand.sources else 0.8
                risk_penalty = 0.2 if cand.risk_class == RiskClass.HIGH else 0.0
                cand.rank_score = round(src_bonus - risk_penalty, 3)
                cand.keep_score = 0.5
                cand.calibrated_confidence = round(det_prob * 0.9 + 0.1, 3)

        return candidates

    def predict_error_prob(self, text: str) -> float:
        if not self.loaded or self.model is None:
            return 0.05
        try:
            with torch.inference_mode():
                feat = torch.zeros((1, 16), dtype=torch.float32)
                prob = self.model(text, feat)
                return float(prob.item())
        except Exception:
            return 0.05
