import torch
from spellchecker.schema import Candidate


def extract_candidate_features(cand: Candidate) -> torch.Tensor:
    vec = [
        1.0 if cand.hard_valid else 0.0,
        float(len(cand.replacement)),
        float(len(cand.original_text)),
        1.0 if cand.sources and "KEEP" in cand.sources else 0.0,
        1.0 if cand.sources and "reviewed_error_memory" in cand.sources else 0.0,
        1.0 if cand.sources and "initial_i" in cand.sources[0] else 0.0,
        1.0 if cand.sources and "sun_article" in cand.sources[0] else 0.0,
        1.0 if cand.sources and "numeral" in cand.sources[0] else 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    ]
    return torch.tensor([vec], dtype=torch.float32)
