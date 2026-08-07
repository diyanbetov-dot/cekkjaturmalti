from __future__ import annotations

import argparse
import os
import re
import uuid
from pathlib import Path

from flask import Flask, jsonify, request, send_file

from neural_corrector.inference.corrector import NeuralCorrector

PROJECT_ROOT = Path(__file__).resolve().parents[2]
UI_PATH = PROJECT_ROOT / "Essentials" / "index.html"
DEVTOY_PATH = PROJECT_ROOT / "Essentials" / "devtoy.js"
DEFAULT_ARTIFACT = PROJECT_ROOT / "neural_corrector" / "artifacts" / "char_edit_bigru_v5"


BARE_WORD_PATTERN = re.compile(r"^[^\W\d_]+(?:['’][^\W\d_]+)?$", re.UNICODE)


def is_bare_word_input(text: str) -> bool:
    return bool(BARE_WORD_PATTERN.fullmatch(text.strip()))


def tokens_from_result(result: dict) -> list[dict]:
    corrected = result["corrected_text"]
    expose_alternatives = is_bare_word_input(result.get("original_text", ""))
    if expose_alternatives and result.get("sequence_alternatives"):
        alternatives = result["sequence_alternatives"]
        confidence = result.get("confidence", 0.0)
        return [
            {
                "type": "word",
                "original": result["original_text"],
                "corrected": corrected,
                "ambiguous": len(alternatives) > 1,
                "crucial": confidence >= 0.8,
                "unrecognized": False,
                "choices": [
                    {
                        "word": alternative,
                        "meaning": "",
                        "source": "neural",
                        "confidence": confidence,
                        "category": "spelling",
                    }
                    for alternative in alternatives
                ],
                "neural_edits": result["edits"],
            }
        ]
    tokens: list[dict] = []
    cursor = 0
    for edit in result["edits"]:
        start = edit["corrected_start"]
        end = edit["corrected_end"]
        if start > cursor:
            tokens.append({"type": "text", "text": corrected[cursor:start]})
        replacement = corrected[start:end]
        if replacement:
            if expose_alternatives:
                choices = []
                for alternative in edit["alternatives"]:
                    if alternative == "":
                        continue
                    choices.append(
                        {
                            "word": alternative,
                            "meaning": "",   # suppress technical neural labels
                            "source": "neural",
                            "confidence": edit["confidence"],
                            "category": edit["type"],
                        }
                    )
                tokens.append(
                    {
                        "type": "word",
                        "original": edit["original"],
                        "corrected": replacement,
                        "ambiguous": len(choices) > 1,
                        "crucial": edit["confidence"] >= 0.8,
                        "unrecognized": False,
                        "choices": choices,
                        "neural_edit": edit,
                    }
                )
            else:
                tokens.append({"type": "text", "text": replacement})
        cursor = end
    if cursor < len(corrected):
        tokens.append({"type": "text", "text": corrected[cursor:]})
    if not tokens:
        tokens.append({"type": "text", "text": corrected})
    return tokens


def create_app(artifact_dir: Path = DEFAULT_ARTIFACT) -> Flask:
    app = Flask(__name__)
    corrector = NeuralCorrector(artifact_dir)

    @app.get("/")
    def index():
        return send_file(UI_PATH)

    @app.get("/devtoy.js")
    def devtoy():
        return send_file(DEVTOY_PATH)

    @app.get("/health")
    def health():
        return jsonify(
            {
                "ok": True,
                "status": "ok",
                "system": "neural-first-experiment",
                "model_version": corrector.model_version,
                "action_threshold": corrector.threshold,
                "dictionary_validation": (
                    corrector.dictionary_validation_enabled
                ),
            }
        )

    @app.post("/check-text")
    def check_text():
        payload = request.get_json(silent=True) or {}
        text = payload.get("text", "")
        if not isinstance(text, str):
            return jsonify({"error": "text must be a string."}), 400
        if not text.strip():
            return jsonify({"error": "Please write some Maltese text first."}), 400
        result = corrector.correct(text)
        result.update(
            {
                "log_id": str(uuid.uuid4()),
                "tokens": tokens_from_result(result),
                "system": "neural-first-experiment",
            }
        )
        return jsonify(result)

    @app.post("/suggest-word")
    def suggest_word():
        payload = request.get_json(silent=True) or {}
        word = payload.get("word", "")
        if not isinstance(word, str) or not word.strip():
            return jsonify({"error": "word must be a non-empty string."}), 400
        result = corrector.correct(word)
        suggestions = []
        for edit in result["edits"]:
            suggestions.extend(edit["alternatives"])
        return jsonify(
            {
                "word": word,
                "corrected": result["corrected_text"],
                "suggestions": list(dict.fromkeys(suggestions)),
                "edits": result["edits"],
                "model_version": result["model_version"],
            }
        )

    @app.post("/debug-word")
    def debug_word():
        payload = request.get_json(silent=True) or {}
        return jsonify(corrector.correct(str(payload.get("word", ""))))

    @app.post("/log-suggestion-choice")
    def log_suggestion_choice():
        return jsonify({"ok": True, "stored": False, "experiment": True})

    @app.post("/submit-feedback")
    def submit_feedback():
        return jsonify(
            {
                "error": "Feedback delivery is disabled in the local neural experiment."
            }
        ), 503

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT)
    args = parser.parse_args()
    create_app(args.artifact_dir).run(
        host=args.host, port=args.port, debug=False, use_reloader=False
    )


app = None

if __name__ == "__main__":
    main()
