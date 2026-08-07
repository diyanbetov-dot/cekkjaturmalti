from typing import List, Tuple
from spellchecker.schema import Candidate, ErrorClass, SelectedEdit
from .constraints import is_overlapping


class GlobalDecoder:
    def decode(self, candidates: List[Candidate]) -> List[SelectedEdit]:
        # Filter changed candidates vs KEEP candidates
        changed = [c for c in candidates if c.operation_type != ErrorClass.KEEP and c.hard_valid]

        # Priority: longer span replacements first, then earlier start position
        changed.sort(key=lambda c: (-(c.source_end - c.source_start), c.source_start))

        selected_cands: List[Candidate] = []
        for cand in changed:
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
                )
            )
        return edits
