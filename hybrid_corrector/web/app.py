"""hybrid_corrector/web/app.py — Flask app for the Hybrid-First experiment (port 5002).

This is experimental branch 2. It combines:
  - Stage 1: UniversalMalteseSpellchecker (word-level precision)
  - Stage 2: NeuralCorrector (conservative context arbiter for unrecognised words)
  - Stage 3: MalteseGrammarRuleEngine (same as the main app)

All three components are *shared* from the already-loaded Essentials.app module,
so startup is fast (the main pipeline is already warm).
"""
from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

from flask import Flask, jsonify, request, send_file

# ── Resolve project root ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

UI_PATH = PROJECT_ROOT / "Essentials" / "index.html"
DEVTOY_PATH = PROJECT_ROOT / "Essentials" / "devtoy.js"
DEFAULT_ARTIFACT = (
    PROJECT_ROOT / "neural_corrector" / "artifacts" / "char_edit_bigru_v6"
)


# ---------------------------------------------------------------------------
# Suggestion helpers — strips noisy neural labels from choices
# ---------------------------------------------------------------------------

_SUPPRESS_MEANINGS = frozenset({
    "Neural whole-word alternative",
    "The neural model predicts this spelling",
    "The neural model restored għ",
    "Missing or incorrect Maltese diacritic",
    "Capitalization changed from contextual evidence",
    "Word spacing was adjusted",
    "Apostrophe usage was adjusted",
    "Hyphenation or article attachment was adjusted",
    "A word boundary or short phrase was corrected",
    "The learned correction changes the word form",
    "Punctuation was adjusted",
})


def _clean_choices(choices: list) -> list:
    """Remove noisy neural meaning labels from suggestion choices."""
    result = []
    for choice in choices:
        if isinstance(choice, dict) and choice.get("meaning") in _SUPPRESS_MEANINGS:
            choice = {**choice, "meaning": ""}
        result.append(choice)
    return result


def _build_pipeline(artifact_dir: Path) -> tuple:
    """
    Initialise a fully-wired spellchecker + neural corrector + grammar engine
    by importing the pre-configured instances from Essentials.app.
    """
    from Essentials.app import spellchecker, grammar_rule_engine
    from Essentials.core.spellchecker import MAX_TEXT_LENGTH
    from neural_corrector.inference.corrector import NeuralCorrector
    from hybrid_corrector.pipeline import HybridFirstCorrector

    print("hybrid-first: loading neural corrector...")
    neural_corrector = NeuralCorrector(artifact_dir)

    pipeline = HybridFirstCorrector(
        spellchecker=spellchecker,
        neural_corrector=neural_corrector,
        grammar_rule_engine=grammar_rule_engine,
        neural_confidence_threshold=0.78,
    )
    print("hybrid-first: pipeline ready.")
    return pipeline, MAX_TEXT_LENGTH


def create_app(artifact_dir: Path = DEFAULT_ARTIFACT) -> Flask:
    app = Flask(__name__)
    pipeline, MAX_TEXT_LENGTH = _build_pipeline(artifact_dir)

    @app.get("/")
    def index():
        return send_file(UI_PATH)

    @app.get("/devtoy.js")
    def devtoy():
        return send_file(DEVTOY_PATH)

    @app.get("/health")
    def health():
        nc = pipeline.neural_corrector
        sc = pipeline.spellchecker
        return jsonify({
            "ok": True,
            "status": "ok",
            "system": "hybrid-first-experiment",
            "neural_model_version": nc.model_version,
            "neural_confidence_threshold": pipeline.threshold,
            "dictionary_words": len(sc.dictionary),
        })

    @app.post("/check-text")
    def check_text():
        payload = request.get_json(silent=True) or {}
        text = payload.get("text", "")
        if not isinstance(text, str):
            return jsonify({"error": "text must be a string."}), 400
        if not text.strip():
            return jsonify({"error": "Please write some Maltese text first."}), 400
        if len(text) > MAX_TEXT_LENGTH:
            return jsonify({
                "error": (
                    f"Text is too long. Maximum length is {MAX_TEXT_LENGTH} characters."
                )
            }), 413

        try:
            include_grammar = bool(payload.get("include_grammar", True))
            result = pipeline.correct(text, include_grammar=include_grammar)
            result["log_id"] = str(uuid.uuid4())
            result["tokens"] = _sanitise_tokens(result.get("tokens", []))
            return jsonify(result)
        except Exception:
            app.logger.exception("hybrid-first /check-text error")
            return jsonify({"error": "Internal spell-checking error."}), 500

    @app.post("/suggest-word")
    def suggest_word():
        payload = request.get_json(silent=True) or {}
        word = payload.get("word", "")
        if not isinstance(word, str) or not word.strip():
            return jsonify({"error": "word must be a non-empty string."}), 400
        try:
            result = pipeline.correct(word, include_grammar=False)
            corrected = result["corrected_text"]
            suggestions: list[str] = []
            for tok in result.get("tokens", []):
                if isinstance(tok, dict) and tok.get("type") == "word":
                    for ch in tok.get("choices", []):
                        w = ch.get("word", "") if isinstance(ch, dict) else ch
                        if w and w.lower() != word.lower() and w not in suggestions:
                            suggestions.append(w)

            sc_suggs = pipeline.spellchecker.suggest(word, limit=4)
            for s in sc_suggs:
                sw = s.get("word", "") if isinstance(s, dict) else str(s)
                if sw and sw.lower() != word.lower() and sw not in suggestions:
                    suggestions.append(sw)

            nc_res = pipeline.neural_corrector.correct(word)
            nc_cand = nc_res.get("corrected_text", "")
            if nc_cand and nc_cand.lower() != word.lower() and nc_cand not in suggestions:
                suggestions.append(nc_cand)

            if not suggestions:
                suggestions = [corrected]
            return jsonify({
                "word": word,
                "corrected": corrected,
                "suggestions": suggestions[:4],
                "system": "hybrid-first-experiment",
            })
        except Exception:
            app.logger.exception("hybrid-first /suggest-word error")
            return jsonify({"error": "Internal error."}), 500

    @app.post("/log-suggestion-choice")
    def log_suggestion_choice():
        return jsonify({"ok": True, "stored": False, "experiment": True})

    @app.post("/submit-feedback")
    def submit_feedback():
        return jsonify({
            "error": "Feedback delivery is disabled in the local hybrid experiment."
        }), 503

    @app.post("/debug-word")
    def debug_word():
        payload = request.get_json(silent=True) or {}
        word = str(payload.get("word", ""))
        result = pipeline.correct(word, include_grammar=False)
        return jsonify(result)

    return app


def _sanitise_tokens(tokens: list) -> list:
    """Walk through the token list and strip noisy neural meaning labels."""
    result = []
    for tok in tokens:
        if isinstance(tok, dict) and tok.get("type") == "word":
            tok = dict(tok)
            if "choices" in tok:
                tok["choices"] = _clean_choices(tok["choices"])
        result.append(tok)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid-First Maltese Spellchecker (port 5002)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5002)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT)
    args = parser.parse_args()
    create_app(args.artifact_dir).run(
        host=args.host, port=args.port, debug=False, use_reloader=False
    )


if __name__ == "__main__":
    main()
