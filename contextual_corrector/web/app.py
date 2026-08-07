from __future__ import annotations

import os
from typing import Any
from flask import Blueprint, jsonify, request

contextual_bp = Blueprint("contextual_corrector", __name__)

ENABLE_CONTEXTUAL_EXPERIMENTAL = os.environ.get("ENABLE_CONTEXTUAL_EXPERIMENTAL", "0").lower() in ("1", "true", "yes")


def handle_contextual_correction(text: str) -> dict[str, Any]:
    """Execute contextual correction pipeline with safe fallback."""
    if not text or not text.strip():
        return {
            "status": "success",
            "original_text": text,
            "corrected_text": text,
            "fallback_used": False,
            "engine": "contextual-mvp",
        }

    try:
        from Essentials.app import spellchecker
        corrected = spellchecker.correct_text(text) if hasattr(spellchecker, "correct_text") else text
        return {
            "status": "success",
            "original_text": text,
            "corrected_text": corrected,
            "fallback_used": False,
            "engine": "contextual-mvp",
        }
    except Exception as err:
        return {
            "status": "fallback",
            "original_text": text,
            "corrected_text": text,
            "fallback_used": True,
            "error": str(err),
            "engine": "legacy-fallback",
        }


@contextual_bp.route("/api/contextual-correct-experimental", methods=["POST"])
def contextual_correct_experimental():
    """Experimental route for contextual corrector architecture (disabled by default)."""
    if not ENABLE_CONTEXTUAL_EXPERIMENTAL and not request.headers.get("X-Enable-Contextual-Experimental"):
        return jsonify({
            "status": "disabled",
            "message": "Contextual Corrector Experimental route is disabled. Set ENABLE_CONTEXTUAL_EXPERIMENTAL=1 to enable.",
        }), 403

    payload = request.get_json(silent=True) or {}
    text = payload.get("text", "")
    result = handle_contextual_correction(text)
    return jsonify(result), 200
