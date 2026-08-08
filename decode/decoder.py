from typing import List, Tuple
from spellchecker.schema import Candidate, ErrorClass, RiskClass, SelectedEdit
from .constraints import is_overlapping


class GlobalDecoder:
    def __init__(self, high_risk_threshold: float = 0.95) -> None:
        self.high_risk_threshold = high_risk_threshold

    def decode(self, candidates: List[Candidate]) -> List[SelectedEdit]:
        # Filter changed candidates vs KEEP candidates
        valid_changed: List[Candidate] = []

        for c in candidates:
            if c.operation_type == ErrorClass.KEEP or not c.hard_valid:
                continue

            # Safety Gate: High Risk changes (e.g. valid-word -> valid-word) require high detector probability & calibrated confidence!
            if c.risk_class == RiskClass.HIGH:
                if c.detector_probability < self.high_risk_threshold or c.calibrated_confidence < self.high_risk_threshold:
                    continue  # Reject high-risk change, KEEP wins!

            valid_changed.append(c)

        # Priority: longer span replacements first, then earlier start position
        valid_changed.sort(key=lambda c: (-(c.source_end - c.source_start), c.source_start))

        selected_cands: List[Candidate] = []
        for cand in valid_changed:
            if not any(is_overlapping(cand, sel) for sel in selected_cands):
                selected_cands.append(cand)

        # Sort selected candidates by start position for rendering
        selected_cands.sort(key=lambda c: c.source_start)

        edits: List[SelectedEdit] = []
        for cand in selected_cands:
            edits.append(
                SelectedEdit(
                    source_span=(cand.source_start, cand.source_end),
                    original=cand.original_text,
                    replacement=cand.replacement,
                    reason=cand.operation_type.value,
                    confidence=cand.calibrated_confidence or 0.95,
                    candidate_source=cand.sources[0] if cand.sources else "unknown",
                    diagnostics={
                        "detector_prob": cand.detector_probability,
                        "rank_score": cand.rank_score,
                        "risk_class": cand.risk_class.value,
                    },
                )
            )
        return edits
