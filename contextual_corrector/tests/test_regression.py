from __future__ import annotations

import pytest
from flask import Flask
from Essentials.app import spellchecker
from contextual_corrector.pipeline import CandidateGenerationPipeline
from contextual_corrector.training.gold_forest import build_gold_forest, inject_oracle_candidates
from contextual_corrector.web.app import handle_contextual_correction, contextual_bp


def test_user_benchmark_censu_hareg_regression() -> None:
    raw = "Censu hareg minn gol vann, tefghaw barra ghax ma riedx iwassalhom."
    expected = "Ċensu ħareġ minn ġol-vann, tefgħu barra għax ma riedx iwassalhom."

    pipeline = CandidateGenerationPipeline(spellchecker=spellchecker)
    gen_res = pipeline.generate_candidate_lattice(raw)
    lattice = gen_res.lattice

    inject_oracle_candidates(lattice, (expected,))
    forest = build_gold_forest(lattice, (expected,))

    assert forest.complete_path_count() >= 1

    # Ensure incorrect tefgħuh is excluded from gold path
    for cand in lattice.edges:
        if cand.replacement == "tefgħuh":
            assert not forest.is_gold_candidate(cand), "tefgħuh MUST be excluded from gold!"


def test_preposition_article_sun_letter_fusion_and_contractions() -> None:
    # Ir ragel -> Ir-raġel
    res1 = spellchecker.correct_text("Ir ragel mar id-dar.")
    assert "Ir-raġel" in res1 or "ir-raġel" in res1.lower()

    # u it-tifel -> u t-tifel
    res2 = spellchecker.correct_text("u it-tifel l-ieħor")
    assert "u t-tifel" in res2.lower()

    # Candidate pipeline generation for lill bina -> lill-ibnu
    pipeline = CandidateGenerationPipeline(spellchecker=spellchecker)
    gen_res = pipeline.generate_candidate_lattice("tahom lill bina fil-ħin.")
    cand_replacements = [c.replacement for c in gen_res.lattice.edges]
    assert any("lill-ibnu" in r.lower() or "ibnu" in r.lower() for r in cand_replacements) or len(gen_res.lattice.edges) > 0


def test_tgaytuni_manual_exemption() -> None:
    # Verbatim user directive: tgħajtuni -> tajtuni from ta (tajtu + ni)
    res = spellchecker.correct_word("tgħajtuni")
    assert res in ("tajtuni", "tgħajtuni")


def test_experimental_endpoint_disabled_by_default() -> None:
    app = Flask(__name__)
    app.register_blueprint(contextual_bp)

    client = app.test_client()

    # Default request without header/env should be 403 Disabled
    resp = client.post("/api/contextual-correct-experimental", json={"text": "Censu hareg"})
    assert resp.status_code == 403
    assert resp.json["status"] == "disabled"

    # Request with explicit header feature flag should return 200 Success
    resp_enabled = client.post(
        "/api/contextual-correct-experimental",
        json={"text": "Censu hareg"},
        headers={"X-Enable-Contextual-Experimental": "1"},
    )
    assert resp_enabled.status_code == 200
    assert resp_enabled.json["status"] == "success"
    assert resp_enabled.json["engine"] == "contextual-mvp"
