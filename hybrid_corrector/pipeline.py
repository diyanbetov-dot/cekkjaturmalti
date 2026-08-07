"""hybrid_corrector/pipeline.py — HybridFirstCorrector

Runs the main UniversalMalteseSpellchecker pipeline first (Stage 1), then
uses NeuralCorrector as a conservative arbiter (Stage 2) that is only
allowed to act on words that the main pipeline left UNRECOGNISED or DUBIOUS.

Arbitration rules for accepting a neural suggestion:
  1. The word was flagged unrecognised/dubious by the main pipeline OR the
     main pipeline made no correction to it (original == stage1 word).
  2. Neural model confidence >= NEURAL_CONFIDENCE_THRESHOLD.
  3. The neural suggestion exists in the dictionary or suffix index.
  4. The neural suggestion does NOT strip diacritics already inserted by Stage 1.
  5. The neural suggestion is not identical to the original (no-op guard).

Words where the main pipeline *changed* the token (original != stage1 output)
are LOCKED and cannot be overridden by the neural model.
"""
from __future__ import annotations

import re
import time
from typing import Any

# ---------------------------------------------------------------------------
# Default arbitration thresholds (tunable via constructor)
# ---------------------------------------------------------------------------
NEURAL_CONFIDENCE_THRESHOLD = 0.78   # min neural confidence to accept
DIACRITICS = set("ċġħżĊĠĦŻ") | {"għ", "Għ"}

# Regex for Maltese word tokens (mirrors the neural corrector's WORD_RE)
_WORD_RE = re.compile(r"[^\W\d_]+(?:[''][^\W\d_]+)*", re.UNICODE)

# Diacritics that the main pipeline adds; the neural model must not remove them
_DIAC_CHARS = frozenset("ċġħżĊĠĦŻ")


def _has_more_diacritics(stage1_word: str, neural_word: str) -> bool:
    """Return True when stage1 introduced diacritics that neural_word lacks."""
    s1_diac = {c for c in stage1_word if c in _DIAC_CHARS}
    nn_diac = {c for c in neural_word if c in _DIAC_CHARS}
    # Also check għ digraph
    s1_gh = stage1_word.lower().count("għ")
    nn_gh = neural_word.lower().count("għ")
    return bool(s1_diac - nn_diac) or (s1_gh > nn_gh)


class HybridFirstCorrector:
    """Three-stage hybrid corrector combining rule-based and neural approaches."""

    def __init__(
        self,
        spellchecker,
        neural_corrector,
        grammar_rule_engine=None,
        neural_confidence_threshold: float = NEURAL_CONFIDENCE_THRESHOLD,
    ) -> None:
        self.spellchecker = spellchecker
        self.neural_corrector = neural_corrector
        self.grammar_rule_engine = grammar_rule_engine
        self.threshold = neural_confidence_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def correct(self, text: str, include_grammar: bool = True) -> dict:
        started = time.perf_counter()

        # ── Stage 1: Main spellchecker pipeline ───────────────────────
        stage1_result = self.spellchecker.correct_text_rich(text)
        stage1_text = stage1_result["corrected_text"]
        stage1_tokens = stage1_result["tokens"]

        # Track which Stage 1 tokens are already settled. The neural arbiter is
        # allowed to help only where Stage 1 explicitly marked doubt.
        locked_norm: set[str] = set()     # normalised forms the main pipeline changed
        unrecognised_norm: set[str] = set()  # words flagged unrecognised/dubious

        for tok in stage1_tokens:
            if isinstance(tok, dict):
                orig = tok.get("original", "")
                corr = tok.get("corrected", "")
                if orig and corr and orig.lower() != corr.lower():
                    locked_norm.add(corr.lower())   # lock the corrected form
                if tok.get("unrecognized") or tok.get("dubious"):
                    if orig:
                        unrecognised_norm.add(orig.lower())
                    if corr:
                        unrecognised_norm.add(corr.lower())

        # ── Stage 2: Neural arbiter ────────────────────────────────────
        # Once Stage 1 has made deterministic repairs, the neural layer is
        # intentionally limited to doubtful word surfaces. Running it over the
        # whole paragraph made simple exact words vulnerable to unrelated
        # rewrites and made long texts unnecessarily slow.
        neural_edits: list[dict] = []
        stage2_text = self._apply_neural_word_arbitration(
            stage1_text=stage1_text,
            locked_norm=locked_norm,
            unrecognised_norm=unrecognised_norm,
        )

        # ── Stage 3: Grammar engine ────────────────────────────────────
        grammar_findings: list[dict] = []
        final_text = stage2_text

        if include_grammar and self.grammar_rule_engine is not None:
            final_text, stage1_tokens, grammar_findings = self.grammar_rule_engine.apply_safe_rewrites(
                original_text=stage2_text,
                corrected_text=stage2_text,
                tokens=stage1_tokens,
            )

        # ── Build annotated token list ─────────────────────────────────
        tokens = self._build_tokens(
            text,
            final_text,
            stage1_tokens,
            neural_edits,
            stage1_text=stage1_text,
        )

        elapsed = round(time.perf_counter() - started, 6)
        payload: dict[str, Any] = {
            "original_text": text,
            "corrected_text": final_text,
            "changed": final_text != text,
            "tokens": tokens,
            "system": "hybrid-first-experiment",
            "processing_time": elapsed,
            "debug": {
                "stage1_text": stage1_text,
                "stage2_text": stage2_text,
                "neural_confidence_threshold": self.threshold,
            },
        }
        if grammar_findings:
            payload["grammar_enabled"] = True
            payload["grammar_findings"] = grammar_findings
        return payload

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _word_in_index(self, word: str) -> bool:
        """Return True if 'word' is in the neural dictionary or suffix index."""
        nc = self.neural_corrector
        if nc.dictionary_index is not None and nc.dictionary_index.contains_surface_form(word):
            return True
        if nc.suffix_index is not None and nc.suffix_index.contains(word):
            return True
        return False

    def _apply_neural_arbitration(
        self,
        stage1_text: str,
        neural_text: str,
        neural_edits: list[dict],
        locked_norm: set[str],
        unrecognised_norm: set[str],
    ) -> str:
        """Merge neural corrections into stage1_text using arbitration rules."""
        # Align words between stage1 and neural outputs
        s1_matches = list(_WORD_RE.finditer(stage1_text))
        nn_matches = list(_WORD_RE.finditer(neural_text))

        if len(s1_matches) != len(nn_matches):
            # Word count changed — refuse to apply (safe fallback to stage1)
            return stage1_text

        # Build index of neural edits keyed by start position in neural_text
        edit_confidence: dict[tuple[int, int], float] = {}
        for edit in neural_edits:
            cs, ce = edit.get("corrected_start", -1), edit.get("corrected_end", -1)
            if cs >= 0 and ce > cs:
                edit_confidence[(cs, ce)] = float(edit.get("confidence", 0.0))

        # Collect replacements (applied in reverse to preserve offsets)
        replacements: list[tuple[int, int, str]] = []

        for s1_m, nn_m in zip(s1_matches, nn_matches):
            s1_word = s1_m.group(0)
            nn_word = nn_m.group(0)

            # Skip if neural agrees with stage1
            if s1_word.lower() == nn_word.lower():
                continue

            s1_norm = s1_word.lower()

            # Rule: locked words (main pipeline already fixed them) cannot be touched
            if s1_norm in locked_norm:
                continue

            # Rule: only act on words Stage 1 explicitly marked as doubtful.
            # A valid unchanged word is already a strong decision; letting the
            # neural layer rewrite it caused regressions such as valid words
            # being replaced by unrelated valid words.
            if s1_norm not in unrecognised_norm:
                continue

            # Rule: neural suggestion must be in dictionary / suffix index
            if not self._word_in_index(nn_word):
                continue

            # Rule: neural must not strip diacritics already in stage1 output
            if _has_more_diacritics(s1_word, nn_word):
                continue

            # Rule: confidence threshold — find the best edit confidence for this span
            best_confidence = 0.0
            for (cs, ce), conf in edit_confidence.items():
                # Check if this neural word span overlaps the edit span
                if cs < nn_m.end() and ce > nn_m.start():
                    best_confidence = max(best_confidence, conf)

            if best_confidence < self.threshold:
                continue

            # Accept this neural correction
            replacements.append((s1_m.start(), s1_m.end(), nn_word))

        # Apply replacements in reverse order
        result = stage1_text
        for start, end, replacement in reversed(replacements):
            result = result[:start] + replacement + result[end:]

        return result

    def _apply_neural_word_arbitration(
        self,
        *,
        stage1_text: str,
        locked_norm: set[str],
        unrecognised_norm: set[str],
    ) -> str:
        """Apply neural repairs only to word surfaces Stage 1 left doubtful."""
        if not unrecognised_norm:
            return stage1_text

        replacements: list[tuple[int, int, str]] = []
        checked: dict[str, str | None] = {}

        for match in _WORD_RE.finditer(stage1_text):
            word = match.group(0)
            norm = word.lower()
            if norm in locked_norm or norm not in unrecognised_norm:
                continue

            if norm not in checked:
                checked[norm] = self._neural_single_word_candidate(word)
            candidate = checked[norm]
            if candidate and candidate.lower() != norm:
                replacements.append((match.start(), match.end(), candidate))

        result = stage1_text
        for start, end, replacement in reversed(replacements):
            result = result[:start] + replacement + result[end:]
        return result

    def _neural_single_word_candidate(self, word: str) -> str | None:
        result = self.neural_corrector.correct(word)
        candidate = str(result.get("corrected_text", "")).strip()
        if not candidate or " " in candidate:
            return None
        if candidate.lower() == word.lower():
            return None
        if not self._word_in_index(candidate):
            return None
        if _has_more_diacritics(word, candidate):
            return None
        if self._is_bare_final_vowel_flip(word, candidate):
            return None

        confidence = 0.0
        for edit in result.get("edits", []):
            confidence = max(confidence, float(edit.get("confidence", 0.0)))
        if confidence < self.threshold:
            return None
        return candidate

    @staticmethod
    def _is_bare_final_vowel_flip(original: str, candidate: str) -> bool:
        original_lower = original.lower()
        candidate_lower = candidate.lower()
        vowels = "aeiouàèìòùáéíóúâêîôû"
        return (
            len(original_lower) == len(candidate_lower)
            and len(original_lower) > 3
            and original_lower[:-1] == candidate_lower[:-1]
            and original_lower[-1] in vowels
            and candidate_lower[-1] in vowels
            and original_lower[-1] != candidate_lower[-1]
        )

    def _build_tokens(
        self,
        original_text: str,
        final_text: str,
        stage1_tokens: list,
        neural_edits: list[dict],
        *,
        stage1_text: str | None = None,
    ) -> list[dict]:
        """
        Build the token list for the UI. Uses stage1_tokens as the base and
        marks tokens whose final form differs from stage1 as neural-arbitrated.
        Suppresses neural 'meaning' labels so the UI doesn't show technical
        explanations like 'The neural model predicts this spelling'.
        """
        if stage1_text is not None and final_text == stage1_text:
            return _clone_clean_tokens(stage1_tokens)

        # For simple rendering: rebuild from scratch using final_text vs original
        result_tokens: list[dict] = []
        orig_words = list(_WORD_RE.finditer(original_text))
        final_words = list(_WORD_RE.finditer(final_text))

        # Use stage1_tokens as the primary source of UI metadata
        # (they contain meaning, choices, etc from the main pipeline)
        # Map by normalised corrected word for quick lookup
        stage1_by_corr: dict[str, list[dict]] = {}
        for tok in stage1_tokens:
            if isinstance(tok, dict) and tok.get("type") == "word":
                key = (tok.get("corrected") or tok.get("original") or "").lower()
                stage1_by_corr.setdefault(key, []).append(tok)

        cursor = 0
        for i, final_m in enumerate(final_words):
            # Add any non-word text before this word
            if final_m.start() > cursor:
                result_tokens.append({"type": "text", "text": final_text[cursor:final_m.start()]})

            final_word = final_m.group(0)
            orig_word = orig_words[i].group(0) if i < len(orig_words) else final_word

            if orig_word.lower() == final_word.lower():
                result_tokens.append({"type": "text", "text": final_word})
            else:
                # Find the best matching stage1 token for this word
                s1_candidates = stage1_by_corr.get(final_word.lower(), [])
                base_tok = s1_candidates.pop(0) if s1_candidates else None

                if base_tok and base_tok.get("type") == "word":
                    tok = dict(base_tok)
                    tok["corrected"] = final_word
                    # Clean up neural meaning labels from choices
                    clean_choices = _clean_neural_choices(tok.get("choices", []))
                    if clean_choices:
                        tok["choices"] = clean_choices
                    result_tokens.append(tok)
                else:
                    # Fallback: create a minimal word token
                    result_tokens.append({
                        "type": "word",
                        "original": orig_word,
                        "corrected": final_word,
                        "ambiguous": False,
                        "crucial": True,
                        "unrecognized": False,
                        "choices": [{
                            "word": final_word,
                            "meaning": "",
                            "source": "hybrid",
                            "confidence": 1.0,
                            "category": "spelling",
                        }],
                    })

            cursor = final_m.end()

        if cursor < len(final_text):
            result_tokens.append({"type": "text", "text": final_text[cursor:]})

        return result_tokens


def _token_text(token: Any) -> str:
    if not isinstance(token, dict):
        return ""
    if token.get("type") == "text":
        return str(token.get("text", ""))
    if token.get("type") == "word":
        return str(token.get("corrected") or token.get("original") or "")
    return ""


def _clone_clean_tokens(tokens: list) -> list:
    result = []
    for token in tokens:
        if isinstance(token, dict):
            cloned = dict(token)
            if cloned.get("type") == "word":
                choices = _clean_neural_choices(cloned.get("choices", []))
                if choices:
                    cloned["choices"] = choices
                elif "choices" in cloned:
                    cloned["choices"] = []
            result.append(cloned)
        else:
            result.append(token)
    return result


def _clean_neural_choices(choices: list) -> list:
    """
    Remove or suppress choices that have noisy neural 'meaning' labels like
    'The neural model predicts this spelling', 'Neural whole-word alternative', etc.
    These are technical labels that should not be shown to users.
    """
    _SUPPRESS = {
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
    }
    result = []
    for choice in choices:
        if isinstance(choice, dict):
            # Suppress the meaning if it's a known neural label
            meaning = choice.get("meaning", "")
            if meaning in _SUPPRESS:
                choice = {**choice, "meaning": ""}
            result.append(choice)
        else:
            result.append(choice)
    return result
