from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import nn

from ..schema import SpanCandidate, CandidateOperation, TokenKind
from .dual_encoder import DualEncoderOutput


STATIC_FEATURE_DIM = 18


class CandidateFeatureExtractor:
    """Extracts a 17-dimensional static feature vector from a SpanCandidate."""

    SOURCE_NAMES = (
        "stage1",
        "neural",
        "dictionary",
        "suffix",
        "phrase_orthographic",
        "keep",
    )

    OPERATIONS = (
        CandidateOperation.KEEP,
        CandidateOperation.REPLACE,
        CandidateOperation.MERGE,
        CandidateOperation.SPLIT,
        CandidateOperation.BOUNDARY,
    )

    @classmethod
    def extract_features(cls, candidate: SpanCandidate) -> torch.Tensor:
        orig = candidate.raw_span.text
        repl = candidate.replacement
        orig_len = max(1, len(orig))
        repl_len = len(repl)

        # 1. Edit distance (normalized)
        edit_dist = len(candidate.edit_operations) if candidate.edit_operations else abs(len(orig) - len(repl))
        norm_edit_dist = float(edit_dist) / max(orig_len, repl_len)

        # 2. Is KEEP
        is_keep = 1.0 if candidate.operation == CandidateOperation.KEEP else 0.0

        # 3. Operation one-hot (5 dims)
        op_one_hot = [
            1.0 if candidate.operation == op else 0.0 for op in cls.OPERATIONS
        ]

        # 4. Source multi-hot (6 dims)
        candidate_sources = candidate.sources or ()
        source_multi_hot = [
            1.0 if src in candidate_sources else 0.0 for src in cls.SOURCE_NAMES
        ]

        # 5. Dictionary validity
        dict_valid = (
            1.0
            if (candidate.dictionary_evidence and any(d.exact for d in candidate.dictionary_evidence))
            else 0.0
        )

        # 6. Suffix validity
        suffix_valid = (
            1.0
            if (candidate.suffix_evidence and any(s.surface_valid for s in candidate.suffix_evidence))
            else 0.0
        )

        # 7. Unsupported clitic insertion
        unsupported_clitic = (
            1.0 if candidate.unsupported_clitic_insertion else 0.0
        )

        # 8. Name/place preservation
        is_name = 1.0 if (candidate.metadata and candidate.metadata.get("name_like")) else 0.0

        # 9. Length ratio
        length_ratio = float(repl_len) / orig_len

        feat_list = [
            norm_edit_dist,
            is_keep,
            *op_one_hot,
            *source_multi_hot,
            dict_valid,
            suffix_valid,
            unsupported_clitic,
            is_name,
            length_ratio,
        ]

        return torch.tensor(feat_list, dtype=torch.float32)


@dataclass(slots=True)
class RankerOutput:
    score: torch.Tensor          # scalar Tensor: candidate score (logit)
    gate_weights: torch.Tensor   # Tensor of shape (2,): [raw_gate, s1_gate]
    features: torch.Tensor       # Tensor of shape (17,): static candidate features


class GatedCandidateRanker(nn.Module):
    """Gated contextual candidate ranker combining RAW & S1 representations with static candidate features."""

    def __init__(
        self,
        hidden_dim: int = 768,
        feature_dim: int = STATIC_FEATURE_DIM,
        mlp_hidden_dim: int = 128,
        dropout_prob: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim

        # Gating network to compute attention weights over RAW and S1 contextual span embeddings
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim * 2 + feature_dim, mlp_hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout_prob),
            nn.Linear(mlp_hidden_dim, 2),
            nn.Softmax(dim=-1),
        )

        # Candidate scoring MLP combining gated context representation + static features
        self.score_mlp = nn.Sequential(
            nn.Linear(hidden_dim + feature_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim // 2),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim // 2, 1),
        )

    def forward(
        self,
        raw_span_emb: torch.Tensor,
        s1_span_emb: torch.Tensor,
        features: torch.Tensor,
    ) -> RankerOutput:
        """
        Forward pass for a single candidate or batch.
        raw_span_emb: (hidden_dim,) or (batch, hidden_dim)
        s1_span_emb:  (hidden_dim,) or (batch, hidden_dim)
        features:     (feature_dim,) or (batch, feature_dim)
        """
        is_unbatched = raw_span_emb.dim() == 1
        if is_unbatched:
            raw_span_emb = raw_span_emb.unsqueeze(0)
            s1_span_emb = s1_span_emb.unsqueeze(0)
            features = features.unsqueeze(0)

        # Concatenate inputs for gate calculation
        gate_input = torch.cat([raw_span_emb, s1_span_emb, features], dim=-1)
        gate_weights = self.gate_net(gate_input)  # (batch, 2)

        raw_weight = gate_weights[:, 0:1]
        s1_weight = gate_weights[:, 1:2]

        fused_context = raw_weight * raw_span_emb + s1_weight * s1_span_emb  # (batch, hidden_dim)

        score_input = torch.cat([fused_context, features], dim=-1)
        score = self.score_mlp(score_input).squeeze(-1)  # (batch,)

        if is_unbatched:
            return RankerOutput(
                score=score.squeeze(0),
                gate_weights=gate_weights.squeeze(0),
                features=features.squeeze(0),
            )

        return RankerOutput(
            score=score,
            gate_weights=gate_weights,
            features=features,
        )

    def score_candidate(
        self,
        candidate: SpanCandidate,
        dual_output: DualEncoderOutput,
    ) -> RankerOutput:
        """Extract embeddings and static features for a candidate and score it."""
        raw_span = candidate.raw_span
        s1_char_start = raw_span.char_start
        s1_char_end = raw_span.char_end
        if candidate.s1_alignment is not None and candidate.s1_alignment.s1_spans:
            s1_char_start = candidate.s1_alignment.s1_spans[0].char_start
            s1_char_end = candidate.s1_alignment.s1_spans[-1].char_end

        raw_emb = dual_output.get_raw_span_embedding(raw_span.char_start, raw_span.char_end)
        s1_emb = dual_output.get_s1_span_embedding(s1_char_start, s1_char_end)
        features = CandidateFeatureExtractor.extract_features(candidate).to(raw_emb.device)

        return self.forward(raw_emb, s1_emb, features)
