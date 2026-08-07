import torch
from spellchecker.config import MODELS_HYBRID_DIR
from .model import HybridBERTuModel


class NeuralRuntime:
    def __init__(self) -> None:
        self.model = None
        self.loaded = False
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

    def predict_error_prob(self, text: str) -> float:
        if not self.loaded or self.model is None:
            return 0.0
        try:
            with torch.inference_mode():
                feat = torch.zeros((1, 16), dtype=torch.float32)
                prob = self.model(text, feat)
                return float(prob.item())
        except Exception:
            return 0.0
