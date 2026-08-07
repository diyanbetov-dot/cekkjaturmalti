from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class CharEditTagger(nn.Module):
    def __init__(
        self,
        character_count: int,
        action_count: int,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(character_count, embedding_dim, padding_idx=0)
        self.encoder = nn.GRU(
            embedding_dim,
            hidden_dim,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, action_count)

    def forward(self, inputs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.dropout(self.embedding(inputs))
        packed = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        encoded, _ = self.encoder(packed)
        encoded, _ = pad_packed_sequence(
            encoded, batch_first=True, total_length=inputs.shape[1]
        )
        return self.classifier(self.dropout(encoded))

