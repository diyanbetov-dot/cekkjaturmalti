import pytest
import torch

from contextual_corrector.models.dual_encoder import BERTuDualEncoder, DualEncoderOutput
from contextual_corrector.models.gated_ranker import (
    STATIC_FEATURE_DIM,
    CandidateFeatureExtractor,
    GatedCandidateRanker,
)
from contextual_corrector.lattice import CandidateLattice
from contextual_corrector.text import normalize_for_lattice
from contextual_corrector.schema import (
    CandidateOperation,
    SpanCandidate,
    TextSpan,
)


def test_dual_encoder_is_frozen() -> None:
    encoder = BERTuDualEncoder(use_mock_encoder=True)
    assert encoder.verify_frozen()
    out = encoder.encode_contexts("Censu hareg", "Ċensu ħareġ")
    assert isinstance(out, DualEncoderOutput)
    assert out.raw_sentence_embedding.shape[0] == encoder.hidden_dim
    assert out.s1_sentence_embedding.shape[0] == encoder.hidden_dim
    assert out.raw_sentence_embedding.requires_grad is False


def test_dual_encoder_span_embedding_extraction() -> None:
    encoder = BERTuDualEncoder(use_mock_encoder=True, hidden_dim=64)
    out = encoder.encode_contexts("Censu hareg minn gol vann", "Ċensu ħareġ minn ġol-vann")

    raw_span_emb = out.get_raw_span_embedding(0, 5)
    assert raw_span_emb.shape == (64,)

    s1_span_emb = out.get_s1_span_embedding(0, 5)
    assert s1_span_emb.shape == (64,)


def test_candidate_feature_extractor() -> None:
    lattice = CandidateLattice(
        sentence_id="raw_test",
        raw=normalize_for_lattice("Censu hareg"),
    )
    candidate = lattice.make_candidate(
        span=TextSpan(token_start=0, token_end=1, char_start=0, char_end=5, text="Censu"),
        replacement="Ċensu",
        operation=CandidateOperation.REPLACE,
        sources=("stage1", "dictionary"),
    )
    features = CandidateFeatureExtractor.extract_features(candidate)
    assert features.shape == (STATIC_FEATURE_DIM,)
    assert features.dtype == torch.float32


def test_gated_ranker_forward_and_gradient_isolation() -> None:
    encoder = BERTuDualEncoder(use_mock_encoder=True, hidden_dim=64)
    ranker = GatedCandidateRanker(hidden_dim=64, mlp_hidden_dim=32)

    raw_text = "Censu hareg minn gol vann"
    s1_text = "Ċensu ħareġ minn ġol-vann"
    dual_out = encoder.encode_contexts(raw_text, s1_text)

    lattice = CandidateLattice(
        sentence_id="raw_test2",
        raw=normalize_for_lattice(raw_text),
    )
    candidate = lattice.make_candidate(
        span=TextSpan(token_start=0, token_end=1, char_start=0, char_end=5, text="Censu"),
        replacement="Ċensu",
        operation=CandidateOperation.REPLACE,
        sources=("stage1",),
    )

    ranker_out = ranker.score_candidate(candidate, dual_out)
    assert ranker_out.score.dim() == 0  # scalar
    assert ranker_out.gate_weights.shape == (2,)
    assert torch.isclose(ranker_out.gate_weights.sum(), torch.tensor(1.0), atol=1e-5)

    # Compute loss & backward pass
    loss = ranker_out.score ** 2
    loss.backward()

    # Ranker parameters MUST receive gradients
    ranker_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in ranker.parameters())
    assert ranker_has_grad

    # Encoder parameters MUST NOT receive gradients
    encoder_has_grad = any(p.grad is not None for p in encoder.parameters())
    assert not encoder_has_grad
