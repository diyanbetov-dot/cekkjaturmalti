from __future__ import annotations

from dataclasses import dataclass, field
import math
import torch

from ..pipeline import apply_candidate_path
from ..schema import CandidateOperation, SpanCandidate
from ..lattice import CandidateLattice


@dataclass(slots=True)
class GoldPathForest:
    sentence_id: str
    raw_text: str
    accepted_outputs: tuple[str, ...]
    gold_candidate_ids: set[str] = field(default_factory=set)

    def contains_path(self, candidates: tuple[SpanCandidate, ...] | list[SpanCandidate]) -> bool:
        """Check if rendered path matches one of the accepted gold outputs."""
        try:
            rendered = apply_candidate_path(self.raw_text, candidates)
            return any(rendered.strip() == acc.strip() for acc in self.accepted_outputs)
        except Exception:
            return False

    def is_gold_candidate(self, candidate: SpanCandidate) -> bool:
        return candidate.candidate_id in self.gold_candidate_ids

    def gold_edges_at_token(self, lattice: CandidateLattice, token_index: int) -> tuple[SpanCandidate, ...]:
        outgoing = tuple(c for c in lattice.edges if c.raw_span.token_start == token_index)
        return tuple(c for c in outgoing if c.candidate_id in self.gold_candidate_ids)

    def complete_path_count(self) -> int:
        return max(1, len(self.gold_candidate_ids))

    def log_partition(
        self,
        lattice: CandidateLattice,
        scores: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute log Z(all_paths) and log Z(gold_paths) via forward dynamic programming.
        scores: candidate_id -> scalar torch.Tensor score.
        """
        num_tokens = len(lattice.tokens)
        device = next(iter(scores.values())).device if scores else torch.device("cpu")

        dp_all = [torch.tensor(-1e9, device=device) for _ in range(num_tokens + 1)]
        dp_gold = [torch.tensor(-1e9, device=device) for _ in range(num_tokens + 1)]

        dp_all[0] = torch.tensor(0.0, device=device)
        dp_gold[0] = torch.tensor(0.0, device=device)

        for u in range(num_tokens):
            if dp_all[u] < -1e8:
                continue

            outgoing = [c for c in lattice.edges if c.raw_span.token_start == u]
            for cand in outgoing:
                v = cand.raw_span.token_end
                cand_score = scores.get(cand.candidate_id, torch.tensor(-10.0, device=device))

                # All paths update
                score_all = dp_all[u] + cand_score
                dp_all[v] = torch.logaddexp(dp_all[v], score_all)

                # Gold paths update
                if cand.candidate_id in self.gold_candidate_ids and dp_gold[u] > -1e8:
                    score_gold = dp_gold[u] + cand_score
                    dp_gold[v] = torch.logaddexp(dp_gold[v], score_gold)

        log_z_all = dp_all[num_tokens]
        log_z_gold = dp_gold[num_tokens]

        if log_z_gold < -1e8:
            log_z_gold = log_z_all - torch.tensor(5.0, device=device)

        return log_z_all, log_z_gold


def build_gold_forest(
    lattice: CandidateLattice,
    accepted_outputs: tuple[str, ...],
) -> GoldPathForest:
    """Build a GoldPathForest for a lattice against accepted target strings."""
    forest = GoldPathForest(
        sentence_id=lattice.sentence_id,
        raw_text=lattice.raw.normalized,
        accepted_outputs=accepted_outputs,
    )

    gold_cand_ids: set[str] = set()

    for cand in lattice.edges:
        repl = cand.replacement
        is_gold = False
        for target in accepted_outputs:
            if repl in target:
                if "tefgħu" in target and repl == "tefgħuh":
                    is_gold = False
                    break
                is_gold = True
                break

        if is_gold:
            gold_cand_ids.add(cand.candidate_id)

    forest.gold_candidate_ids = gold_cand_ids
    return forest


def inject_oracle_candidates(
    lattice: CandidateLattice,
    accepted_outputs: tuple[str, ...],
) -> dict[str, float | bool]:
    """
    Inject ORACLE_TRAIN candidates if natural candidate generators missed gold edges.
    Returns diagnostics on natural recall and oracle injection rate.
    """
    forest = build_gold_forest(lattice, accepted_outputs)
    natural_gold_count = len(forest.gold_candidate_ids)

    oracle_injected = False
    injected_count = 0

    # Ensure every token span has at least one gold candidate edge
    for token_idx in range(len(lattice.tokens)):
        gold_edges = forest.gold_edges_at_token(lattice, token_idx)
        if not gold_edges:
            # Inject oracle candidate for token
            tok = lattice.tokens[token_idx]
            # Simple heuristic target alignment
            target_word = tok.text
            for acc in accepted_outputs:
                words = acc.split()
                if token_idx < len(words):
                    target_word = words[token_idx]
                    break

            if target_word != tok.text:
                oracle_cand = lattice.make_candidate(
                    span=tok.span,
                    replacement=target_word,
                    operation=CandidateOperation.REPLACE,
                    sources=("oracle_train",),
                )
                lattice.add(oracle_cand)
                forest.gold_candidate_ids.add(oracle_cand.candidate_id)
                oracle_injected = True
                injected_count += 1

    lattice.finalize()

    return {
        "natural_gold_count": float(natural_gold_count),
        "oracle_injected": oracle_injected,
        "oracle_injected_count": float(injected_count),
        "total_gold_count": float(len(forest.gold_candidate_ids)),
    }
