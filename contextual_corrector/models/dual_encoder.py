from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import nn


@dataclass(slots=True)
class DualEncoderOutput:
    raw_text: str
    s1_text: str
    raw_hidden: torch.Tensor  # shape: (seq_len_raw, hidden_dim)
    s1_hidden: torch.Tensor   # shape: (seq_len_s1, hidden_dim)
    raw_sentence_embedding: torch.Tensor  # shape: (hidden_dim,)
    s1_sentence_embedding: torch.Tensor   # shape: (hidden_dim,)
    hidden_dim: int

    def get_raw_span_embedding(self, char_start: int, char_end: int) -> torch.Tensor:
        """Extract pooled span embedding from RAW text based on character offsets."""
        text_len = len(self.raw_text)
        if text_len == 0 or self.raw_hidden.size(0) == 0:
            return torch.zeros(self.hidden_dim, device=self.raw_hidden.device)

        # Character to token index approximation
        # Token 0 is [CLS], Token -1 is [SEP]
        num_tokens = self.raw_hidden.size(0)
        if num_tokens <= 2:
            return self.raw_sentence_embedding

        content_tokens = num_tokens - 2
        tok_start = 1 + int((char_start / max(1, text_len)) * content_tokens)
        tok_end = 1 + max(tok_start + 1, int((char_end / max(1, text_len)) * content_tokens))
        tok_start = max(1, min(num_tokens - 1, tok_start))
        tok_end = max(tok_start + 1, min(num_tokens, tok_end))

        span_tokens = self.raw_hidden[tok_start:tok_end]
        if span_tokens.size(0) == 0:
            return self.raw_sentence_embedding
        return span_tokens.mean(dim=0)

    def get_s1_span_embedding(self, char_start: int, char_end: int) -> torch.Tensor:
        """Extract pooled span embedding from S1 text based on character offsets."""
        text_len = len(self.s1_text)
        if text_len == 0 or self.s1_hidden.size(0) == 0:
            return torch.zeros(self.hidden_dim, device=self.s1_hidden.device)

        num_tokens = self.s1_hidden.size(0)
        if num_tokens <= 2:
            return self.s1_sentence_embedding

        content_tokens = num_tokens - 2
        tok_start = 1 + int((char_start / max(1, text_len)) * content_tokens)
        tok_end = 1 + max(tok_start + 1, int((char_end / max(1, text_len)) * content_tokens))
        tok_start = max(1, min(num_tokens - 1, tok_start))
        tok_end = max(tok_start + 1, min(num_tokens, tok_end))

        span_tokens = self.s1_hidden[tok_start:tok_end]
        if span_tokens.size(0) == 0:
            return self.s1_sentence_embedding
        return span_tokens.mean(dim=0)


class BERTuDualEncoder(nn.Module):
    """Frozen BERTu dual encoder for RAW and S1 text contexts."""

    def __init__(
        self,
        model_name_or_path: str = "MLRS/BERTu",
        *,
        device: str | torch.device = "cpu",
        use_mock_encoder: bool = False,
        hidden_dim: int = 768,
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.hidden_dim = hidden_dim
        self.use_mock = use_mock_encoder
        self.model_name = model_name_or_path

        if not self.use_mock:
            try:
                from transformers import AutoModel, AutoTokenizer

                self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
                self.bert = AutoModel.from_pretrained(model_name_or_path)
                self.bert.eval()
                # Freeze all parameters
                for param in self.bert.parameters():
                    param.requires_grad = False
                self.hidden_dim = self.bert.config.hidden_size
            except Exception:
                # Fall back to lightweight mock mode if network/checkpoint is absent
                self.use_mock = True

        if self.use_mock:
            self.tokenizer = None
            self.bert = None
            # Dummy embedding layer for mock mode
            self.dummy_emb = nn.Embedding(256, self.hidden_dim)
            for param in self.dummy_emb.parameters():
                param.requires_grad = False

        self.to(self.device)

    def verify_frozen(self) -> bool:
        """Verify that all encoder parameters have requires_grad=False."""
        if self.use_mock:
            return not any(p.requires_grad for p in self.dummy_emb.parameters())
        return not any(p.requires_grad for p in self.bert.parameters())

    def _encode_single_text(self, text: str) -> tuple[torch.Tensor, torch.Tensor]:
        if self.use_mock or self.bert is None:
            chars = [ord(c) % 256 for c in text] or [0]
            input_ids = torch.tensor(chars, dtype=torch.long, device=self.device)
            with torch.no_grad():
                seq_emb = self.dummy_emb(input_ids)
                cls_emb = seq_emb.mean(dim=0)
            return seq_emb, cls_emb

        with torch.no_grad():
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            ).to(self.device)
            outputs = self.bert(**inputs)
            hidden = outputs.last_hidden_state.squeeze(0)  # (seq_len, hidden_dim)
            cls_pooled = outputs.pooler_output.squeeze(0) if outputs.pooler_output is not None else hidden[0]
            return hidden, cls_pooled

    def encode_contexts(self, raw_text: str, s1_text: str) -> DualEncoderOutput:
        raw_hidden, raw_cls = self._encode_single_text(raw_text)
        s1_hidden, s1_cls = self._encode_single_text(s1_text)

        return DualEncoderOutput(
            raw_text=raw_text,
            s1_text=s1_text,
            raw_hidden=raw_hidden,
            s1_hidden=s1_hidden,
            raw_sentence_embedding=raw_cls,
            s1_sentence_embedding=s1_cls,
            hidden_dim=self.hidden_dim,
        )
