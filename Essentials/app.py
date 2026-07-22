# Essentials/app.py - Flask API Entry Point & Server Configuration
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from flask import Flask, jsonify, request, send_from_directory

from Essentials.core.spellchecker import (
    UniversalMalteseSpellchecker,
    TokenAnalysis,
    FunctionPhraseDecision,
    BASE_DIR,
    FINAL_DICS_DIR,
    DICTIONARY_FILES,
    MAX_TEXT_LENGTH,
    MAX_WORD_LENGTH,
    SENTENCE_CONTEXT_BACKEND,
    ENABLE_SENTENCE_CONTEXT_ANALYZER,
    ENABLE_APERTIUM_ANALYZER,
    APERTIUM_TIMEOUT_SEC,
    _append_usage_log,
    _trim_request_caches,
)
from Essentials.dictionary_meanings import MeaningIndex
from Essentials.grammar import MalteseGrammarRuleEngine
from Essentials.helpers.apertium_analyzer import OptionalApertiumAnalyzer
from Essentials.helpers.article_phrase_rules import MalteseArticlePhraseRules, WordToken
from Essentials.helpers.context_analyzer import OptionalSentenceContextAnalyzer
from Essentials.helpers.doubled_letter_generator import MalteseDoubledLetterGenerator
from Essentials.helpers.fused_preposition_rules import MalteseFusedPrepositionRules
from Essentials.helpers.orthographic_generator import MalteseOrthographicGenerator
from Essentials.helpers.performance_logging import (
    RequestProfiler,
    log_spellcheck_event,
    reset_current_profiler,
    rss_mb,
    set_current_profiler,
)
from Essentials.helpers.suffix_generator import MalteseSuffixGenerator
from Other.tools.repair_mojibake import repair_mojibake_text

# Flask app
# -----------------------------------------------------------------------------

app = Flask(__name__)

_startup_started = time.perf_counter()
spellchecker = UniversalMalteseSpellchecker(dictionary_files=DICTIONARY_FILES)
meaning_index = MeaningIndex()
meaning_index.load_entries(spellchecker.raw_entries, include_verbs=True)
spellchecker.raw_entries = []
article_phrase_rules = MalteseArticlePhraseRules(
    meaning_index=meaning_index,
    normalizer=spellchecker._normalize_word,
    noun_words=spellchecker.tagged_words_with_marker("NOUN"),
    num_words=spellchecker.tagged_words_with_marker("NUM"),
)
article_phrase_rules.spellchecker = spellchecker
spellchecker.article_phrase_rules = article_phrase_rules

grammar_rule_engine = MalteseGrammarRuleEngine(
    rules_path=BASE_DIR / "grammar" / "grammar_rules_measured.json",
    spellchecker=spellchecker,
    meaning_index=meaning_index,
    article_rules=article_phrase_rules,
)
spellchecker.grammar_rule_engine = grammar_rule_engine
sentence_context_analyzer = OptionalSentenceContextAnalyzer(
    backend=SENTENCE_CONTEXT_BACKEND,
    enabled=ENABLE_SENTENCE_CONTEXT_ANALYZER,
)
apertium_analyzer = OptionalApertiumAnalyzer(
    enabled=ENABLE_APERTIUM_ANALYZER,
    timeout_sec=APERTIUM_TIMEOUT_SEC,
)

orthographic_generator = MalteseOrthographicGenerator(spellchecker=spellchecker)
spellchecker.orthographic_generator = orthographic_generator

doubled_letter_generator = MalteseDoubledLetterGenerator(spellchecker=spellchecker)
spellchecker.doubled_letter_generator = doubled_letter_generator

suffix_generator = MalteseSuffixGenerator(
    spellchecker=spellchecker,
    verbs_file=[
        BASE_DIR / "finaldics/verbmt_semitic.dic",
        BASE_DIR / "finaldics/verbmt_nonsemitic.dic",
    ],
)

spellchecker.suffix_generator = suffix_generator

fused_preposition_rules = MalteseFusedPrepositionRules(
    spellchecker=spellchecker,
    article_rules=article_phrase_rules,
    meaning_index=meaning_index,
)
spellchecker.fused_preposition_rules = fused_preposition_rules

spellchecker.clear_disposable_startup_caches()

log_spellcheck_event(
    event="SPELLCHECK_STARTUP",
    stage="startup_complete",
    instance_id=spellchecker.instance_id,
    elapsed_ms=(time.perf_counter() - _startup_started) * 1000,
    dictionary_words=len(spellchecker.dictionary),
    paradigms=len(spellchecker.paradigm_forms),
    suffix_verb_records=(
        spellchecker.suffix_generator.verb_index.record_count()
        if hasattr(spellchecker, "suffix_generator")
        else None
    ),
    rss_mb=rss_mb(),
)


ENABLE_DEV_TOOLS = False
SHOW_STATUS_MESSAGES = False

@app.get("/")
def home():
    html_path = BASE_DIR / "index.html"
    try:
        with open(html_path, encoding="utf-8") as f:
            html = f.read()
    except FileNotFoundError:
        return "index.html not found", 404

    # Inject configuration variables into the HTML
    html = html.replace(
        '"REPLACE_ME_ENABLE_DEV_TOOLS" === "True"',
        "true" if ENABLE_DEV_TOOLS else "false"
    )
    html = html.replace(
        '"REPLACE_ME_SHOW_STATUS_MESSAGES" === "True"',
        "true" if SHOW_STATUS_MESSAGES else "false"
    )
    return html


@app.get("/devtoy.js")
def devtoy_js():
    return send_from_directory(BASE_DIR, "devtoy.js")


@app.get("/assets/<path:filename>")
def assets(filename):
    return send_from_directory(BASE_DIR / "assets", filename)


@app.get("/devtoy-assets/<filename>")
def devtoy_assets(filename):
    return send_from_directory(BASE_DIR / "assets" / "devtoys", filename)


@app.get("/health")
def health():
    return jsonify(
        {
            "ok": True,
            "status": "ok",
            "dictionary_words": len(spellchecker.dictionary),
            "paradigms": len(spellchecker.paradigm_forms),
        }
    ), 200


@app.post("/check-text")
def check_text():
    profiler = RequestProfiler()
    profiler_token = set_current_profiler(profiler)
    token_count = 0
    unique_tokens = 0
    data = request.get_json(silent=True) or {}
    try:
        log_spellcheck_event(
            event="SPELLCHECK_REQUEST",
            instance_id=spellchecker.instance_id,
            request_id=profiler.request_id,
        )
        text = data.get("text", "")

        if not isinstance(text, str):
            return jsonify({"error": "text must be a string."}), 400

        if not text.strip():
            return jsonify({"error": "Please write some Maltese text first."}), 400

        if len(text) > MAX_TEXT_LENGTH:
            return (
                jsonify(
                    {
                        "error": (
                            f"Text is too long. Maximum length is "
                            f"{MAX_TEXT_LENGTH} characters."
                        )
                    }
                ),
                413,
            )

        request_words = [
            match.group(0) for match in spellchecker.WORD_PATTERN.finditer(text)
        ]
        token_count = len(request_words)
        unique_tokens = len(
            {spellchecker._normalize_word(word) for word in request_words}
        )

        edit_distance_tolerance = int(data.get("edit_distance_tolerance", 1))
        include_grammar = bool(data.get("include_grammar", True))
        with profiler.span(
            "correct_text_rich",
            tokens=token_count,
            unique_tokens=unique_tokens,
        ):
            result = spellchecker.correct_text_rich(
                text, edit_distance_tolerance=edit_distance_tolerance
        )
        corrected_text = result["corrected_text"]
        tokens = result["tokens"]
        grammar_findings: list[dict[str, object]] = []
        grammar_enabled = bool(include_grammar and grammar_rule_engine is not None)
        if grammar_enabled:
            corrected_request_words = [
                match.group(0)
                for match in spellchecker.WORD_PATTERN.finditer(corrected_text)
            ]
            grammar_findings = grammar_rule_engine.analyze(
                text=corrected_text,
                request_words=corrected_request_words,
                tokens=tokens,
            )
            corrected_text, tokens, _ = grammar_rule_engine.apply_safe_rewrites(
                original_text=corrected_text,
                corrected_text=corrected_text,
                tokens=tokens,
            )

        context_shadow = None
        if sentence_context_analyzer.enabled:
            with profiler.span(
                "sentence_context_analyzer",
                backend=sentence_context_analyzer.backend,
            ):
                context_shadow = asdict(sentence_context_analyzer.analyze(corrected_text))

        _append_usage_log(
            request_id=profiler.request_id,
            original_text=text,
            corrected_text=corrected_text,
            tokens=tokens,
        )

        response_payload = {
            "original_text": text,
            "corrected_text": corrected_text,
            "changed": corrected_text != text,
            "tokens": tokens,
        }
        if grammar_enabled:
            response_payload["grammar_enabled"] = True
            response_payload["grammar_findings"] = grammar_findings
        if context_shadow is not None:
            response_payload["context_analyzer"] = context_shadow
        return jsonify(response_payload)
    except Exception:
        app.logger.exception(
            "SPELLCHECK request_id=%s stage=exception", profiler.request_id
        )
        return jsonify({"error": "Internal spell-checking error."}), 500
    finally:
        profiler.finish(token_count=token_count, unique_tokens=unique_tokens)
        reset_current_profiler(profiler_token)
        # Trim request-scoped caches to prevent unbounded memory growth across requests.
        # We only trim caches whose entries are request-specific (word pairs, scores).
        # Dictionary-based caches (_normalize, _graphemes, tag lookups) are kept warm
        # because they store pre-computed facts about the dictionary words, not user text.
        _trim_request_caches()


@app.post("/suggest-word")
def suggest_word():
    data = request.get_json(silent=True) or {}
    word = data.get("word", "")
    spellchecker._reset_request_token_cache()

    if not isinstance(word, str):
        return jsonify({"error": "word must be a string."}), 400

    if not word.strip():
        return jsonify({"error": "Please write a word first."}), 400

    if len(word) > MAX_WORD_LENGTH:
        return (
            jsonify(
                {
                    "error": (
                        f"Word is too long. Maximum length is "
                        f"{MAX_WORD_LENGTH} characters."
                    )
                }
            ),
            413,
        )

    edit_distance_tolerance = int(data.get("edit_distance_tolerance", 1))
    spellchecker.correct_word(word)

    suggestions = spellchecker.suggest(
        word,
        limit=10,
        edit_distance_tolerance=edit_distance_tolerance,
    )

    return jsonify(
        {
            "word": word,
            "suggestions": (
                meaning_index.enrich_choices(
                    [
                        {
                            "word": suggestion,
                            "meaning": spellchecker.meaning_for(suggestion),
                        }
                        for suggestion in suggestions
                    ]
                )
            ),
        }
    )


@app.post("/debug-word")
def debug_word():
    data = request.get_json(silent=True) or {}
    word = data.get("word", "")

    if not isinstance(word, str):
        return jsonify({"error": "word must be a string."}), 400
    if not word.strip():
        return jsonify({"error": "Please write a word first."}), 400
    if len(word) > MAX_WORD_LENGTH:
        return (
            jsonify(
                {
                    "error": f"Word is too long. Maximum length is {MAX_WORD_LENGTH} characters."
                }
            ),
            413,
        )

    edit_distance_tolerance = int(data.get("edit_distance_tolerance", 1))
    return jsonify(
        spellchecker.debug_word(word, edit_distance_tolerance=edit_distance_tolerance)
    )


@app.post("/debug-candidate-evidence")
def debug_candidate_evidence():
    data = request.get_json(silent=True) or {}
    word = data.get("word", "")

    if not isinstance(word, str):
        return jsonify({"error": "word must be a string."}), 400
    if not word.strip():
        return jsonify({"error": "Please write a word first."}), 400
    if len(word) > MAX_WORD_LENGTH:
        return (
            jsonify(
                {
                    "error": f"Word is too long. Maximum length is {MAX_WORD_LENGTH} characters."
                }
            ),
            413,
        )

    return jsonify(spellchecker.candidate_evidence_debug(word))


@app.post("/debug-orthographic")
def debug_orthographic():
    data = request.get_json(silent=True) or {}
    word = data.get("word", "")

    if not isinstance(word, str):
        return jsonify({"error": "word must be a string."}), 400
    if not word.strip():
        return jsonify({"error": "Please write a word first."}), 400
    if len(word) > MAX_WORD_LENGTH:
        return (
            jsonify(
                {
                    "error": f"Word is too long. Maximum length is {MAX_WORD_LENGTH} characters."
                }
            ),
            413,
        )

    return jsonify(orthographic_generator.debug(word))


@app.post("/debug-suffix")
def debug_suffix():
    data = request.get_json(silent=True) or {}
    word = data.get("word", "")

    if not isinstance(word, str):
        return jsonify({"error": "word must be a string."}), 400

    if not word.strip():
        return jsonify({"error": "Please write a word first."}), 400

    if len(word) > MAX_WORD_LENGTH:
        return (
            jsonify(
                {
                    "error": f"Word is too long. Maximum length is {MAX_WORD_LENGTH} characters."
                }
            ),
            413,
        )

    if not hasattr(spellchecker, "suffix_generator"):
        return jsonify({"error": "suffix generator is not attached."}), 500

    return jsonify(spellchecker.suffix_generator.debug_suffix(word))


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False,
        use_reloader=False,
    )
