from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import nn

from .gold_forest import GoldPathForest
from ..lattice import CandidateLattice
from ..models.gated_ranker import RankerOutput


@dataclass(slots=True)
class LossComponents:
    total_loss: torch.Tensor
    l_path: torch.Tensor
    l_local: torch.Tensor
    l_keep: torch.Tensor
    l_calibration: torch.Tensor
    l_relations: torch.Tensor
    l_margin: torch.Tensor
    l_gate_reg: torch.Tensor

    def as_dict(self) -> dict[str, float]:
        return {
            "total_loss": self.total_loss.item(),
            "l_path": self.l_path.item(),
            "l_local": self.l_local.item(),
            "l_keep": self.l_keep.item(),
            "l_calibration": self.l_calibration.item(),
            "l_relations": self.l_relations.item(),
            "l_margin": self.l_margin.item(),
            "l_gate_reg": self.l_gate_reg.item(),
        }


class ContextualStructuredLoss(nn.Module):
    """Structured loss function for contextual candidate ranker."""

    def __init__(
        self,
        *,
        w_local: float = 0.30,
        w_keep: float = 0.20,
        w_calib: float = 0.10,
        w_rel: float = 0.10,
        w_margin: float = 0.15,
        w_gate_reg: float = 0.05,
        margin: float = 1.0,
    ) -> None:
        super().__init__()
        self.w_local = w_local
        self.w_keep = w_keep
        self.w_calib = w_calib
        self.w_rel = w_rel
        self.w_margin = w_margin
        self.w_gate_reg = w_gate_reg
        self.margin = margin

    def forward(
        self,
        lattice: CandidateLattice,
        gold_forest: GoldPathForest,
        ranker_outputs: dict[str, RankerOutput],
    ) -> LossComponents:
        scores = {cid: out.score for cid, out in ranker_outputs.items()}
        device = next(iter(scores.values())).device if scores else torch.device("cpu")

        # 1. Path Loss: log Z(all) - log Z(gold)
        log_z_all, log_z_gold = gold_forest.log_partition(lattice, scores)
        l_path = log_z_all - log_z_gold

        # 2. Local Candidate Loss & Calibration & Margin
        l_local = torch.tensor(0.0, device=device)
        l_keep = torch.tensor(0.0, device=device)
        l_calib = torch.tensor(0.0, device=device)
        l_rel = torch.tensor(0.0, device=device)
        l_margin = torch.tensor(0.0, device=device)
        l_gate_reg = torch.tensor(0.0, device=device)

        token_count = max(1, len(lattice.tokens))

        for token_idx in range(len(lattice.tokens)):
            outgoing = [c for c in lattice.edges if c.raw_span.token_start == token_idx]
            if not outgoing:
                continue

            edge_scores = torch.stack([scores[c.candidate_id] for c in outgoing])
            gold_flags = torch.tensor(
                [1.0 if gold_forest.is_gold_candidate(c) else 0.0 for c in outgoing],
                device=device,
            )

            # Local cross entropy
            if gold_flags.sum() > 0:
                gold_idx = torch.argmax(gold_flags)
                l_local += nn.functional.cross_entropy(edge_scores.unsqueeze(0), gold_idx.unsqueeze(0))

            # Local KEEP supervision
            for c in outgoing:
                c_out = ranker_outputs[c.candidate_id]
                prob = torch.sigmoid(c_out.score)
                is_gold = gold_forest.is_gold_candidate(c)

                # Calibration Brier score
                target_prob = 1.0 if is_gold else 0.0
                l_calib += (prob - target_prob) ** 2

                # Gate regularization (entropy penalty)
                gates = c_out.gate_weights
                gate_entropy = -torch.sum(gates * torch.log(gates + 1e-8))
                l_gate_reg += gate_entropy

                # Relational penalty for unsupported clitics
                if c.unsupported_clitic_insertion and not is_gold:
                    l_rel += torch.relu(c_out.score + 2.0)

                # Margin loss
                if is_gold:
                    for neg_c in outgoing:
                        if not gold_forest.is_gold_candidate(neg_c):
                            neg_score = ranker_outputs[neg_c.candidate_id].score
                            l_margin += torch.relu(self.margin - (c_out.score - neg_score))

                if c.keep and not is_gold:
                    l_keep += prob ** 2

        l_local = l_local / token_count
        l_keep = l_keep / token_count
        l_calib = l_calib / token_count
        l_rel = l_rel / token_count
        l_margin = l_margin / token_count
        l_gate_reg = l_gate_reg / token_count

        total_loss = (
            l_path
            + self.w_local * l_local
            + self.w_keep * l_keep
            + self.w_calib * l_calib
            + self.w_rel * l_rel
            + self.w_margin * l_margin
            + self.w_gate_reg * l_gate_reg
        )

        return LossComponents(
            total_loss=total_loss,
            l_path=l_path,
            l_local=l_local,
            l_keep=l_keep,
            l_calibration=l_calib,
            l_relations=l_rel,
            l_margin=l_margin,
            l_gate_reg=l_gate_reg,
        )
