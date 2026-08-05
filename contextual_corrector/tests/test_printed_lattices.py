from __future__ import annotations

from contextual_corrector.lattice import CandidateLattice
from contextual_corrector.schema import CandidateOperation, SourceEvidence
from contextual_corrector.text import normalize_for_lattice


EXAMPLES = (
    ("xandek", [(0, 1, "x'għandek", CandidateOperation.REPLACE)]),
    ("ma hawnx", [(0, 2, "m'hawnx", CandidateOperation.MERGE)]),
    ("il lejla", [(0, 2, "illejla", CandidateOperation.MERGE)]),
    ("daqs li kieku", [(0, 3, "daqslikieku", CandidateOperation.MERGE)]),
    (
        "Censu hareg minn gol vann, tefghaw barra.",
        [
            (0, 1, "Ċensu", CandidateOperation.REPLACE),
            (1, 2, "ħareġ", CandidateOperation.REPLACE),
            (3, 5, "ġol-vann", CandidateOperation.REPLACE),
            (6, 7, "tefgħu", CandidateOperation.REPLACE),
        ],
    ),
)


def build_printed_lattices() -> tuple[str, ...]:
    rendered = []
    for example_index, (text, candidates) in enumerate(EXAMPLES):
        lattice = CandidateLattice(
            sentence_id=f"printed-{example_index}", raw=normalize_for_lattice(text)
        )
        for token_start, token_end, replacement, operation in candidates:
            span = lattice.span(token_start, token_end)
            lattice.add(
                lattice.make_candidate(
                    span=span,
                    replacement=replacement,
                    operation=operation,
                    sources={
                        "stub": (
                            SourceEvidence(
                                source="stub",
                                rule_id="printed-example",
                                raw_score=0.9,
                                deterministic=True,
                            ),
                        )
                    },
                )
            )
        rendered.append(lattice.render())
    return tuple(rendered)


def test_requested_lattice_examples_render_with_complete_keep_paths() -> None:
    rendered = build_printed_lattices()

    assert len(rendered) == 5
    assert all("complete_keep_path=True" in output for output in rendered)
    assert "x'għandek" in rendered[0]
    assert "m'hawnx" in rendered[1]
    assert "illejla" in rendered[2]
    assert "daqslikieku" in rendered[3]
    assert "'gol vann' -> 'ġol-vann'" in rendered[4]
    assert "'tefghaw' -> 'tefgħu'" in rendered[4]
    assert "tefgħu" in rendered[4]
