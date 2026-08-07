import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from spellchecker.config import BERTU_MODEL_ID


class HybridBERTuModel(nn.Module):
    def __init__(self, bertu_model_id: str = BERTU_MODEL_ID, feature_dim: int = 16) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(bertu_model_id)
        self.bertu = AutoModel.from_pretrained(bertu_model_id)

        # Freeze BERTu by default
        for param in self.bertu.parameters():
            param.requires_grad = False

        hidden_dim = self.bertu.config.hidden_size

        # Error Detector Head
        self.detector = nn.Sequential(
            nn.Linear(hidden_dim + feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        # Char BiGRU Encoder
        self.char_embed = nn.Embedding(256, 32)
        self.char_gru = nn.GRU(32, 32, batch_first=True, bidirectional=True)

        # Ranker Head
        self.ranker = nn.Sequential(
            nn.Linear(hidden_dim + 64 + feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def encode_text(self, text: str):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=384)
        with torch.no_grad():
            outputs = self.bertu(**inputs)
        return outputs.last_hidden_state.mean(dim=1)

    def forward(self, text: str, feature_vec: torch.Tensor):
        ctx_emb = self.encode_text(text)
        detector_in = torch.cat([ctx_emb, feature_vec], dim=-1)
        prob = self.detector(detector_in)
        return prob
