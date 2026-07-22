# -*- coding: utf-8 -*-
"""
BERTu-based candidate re-ranker for the Maltese spellchecker.

Uses the MLRS/BERTu masked language model to score candidate corrections
in context. Given a sentence and a set of candidate corrections for one
token position, the model scores each candidate by substituting it into
the sentence and querying the masked LM probability at that position.

This module is intentionally decoupled from the core spellchecker: it is
an *additive* scoring layer. If BERTu is unavailable (no transformers,
no model weights, OOM, etc.) the system falls back silently to the
existing evidence-based scoring.

Usage:
    from Essentials.helpers.bertu_reranker import BertuReranker
    reranker = BertuReranker()                  # lazy — loads model on first call
    scores = reranker.score_candidates(
        sentence="Werrinha ir-rapport li qal kien ippreżentat",
        token_index=0,                          # position of the misspelled word
        candidates=["Werriena", "Werriha", "Warrani"],
    )
    # scores: {"Werriena": 0.83, "Werriha": 0.61, "Warrani": 0.04}
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

_MODEL_NAME = "MLRS/BERTu"
_MAX_SEQ_LEN = 512
_BATCH_SIZE = 8  # number of candidates to score in one forward pass


class BertuReranker:
    """
    Lazy-loading BERTu masked LM re-ranker.

    The model and tokenizer are only loaded on the first call to
    score_candidates(), so importing this module has zero startup cost.
    """

    def __init__(self, model_name: str = _MODEL_NAME) -> None:
        self._model_name = model_name
        self._model = None
        self._tokenizer = None
        self._available: bool | None = None  # None = not yet checked

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Return True if BERTu loaded successfully."""
        if self._available is None:
            self._load()
        return bool(self._available)

    def score_candidates(
        self,
        sentence: str,
        token_index: int,
        candidates: list[str],
        *,
        tokens: list[str] | None = None,
    ) -> dict[str, float]:
        """
        Score each candidate by its MLM probability at token_index in sentence.

        Parameters
        ----------
        sentence:    The full sentence string (used to split into tokens if
                     `tokens` is not provided).
        token_index: Index of the word to score within the token list.
        candidates:  Candidate replacement strings for that position.
        tokens:      Optional pre-tokenized word list. If None, a simple
                     whitespace split of `sentence` is used.

        Returns
        -------
        dict mapping each candidate to its log-probability score (higher = better).
        Returns an empty dict if BERTu is unavailable.
        """
        if not self.is_available():
            return {}
        if not candidates:
            return {}

        if tokens is None:
            tokens = sentence.split()

        scores: dict[str, float] = {}
        try:
            for batch_start in range(0, len(candidates), _BATCH_SIZE):
                batch = candidates[batch_start : batch_start + _BATCH_SIZE]
                batch_scores = self._score_batch(tokens, token_index, batch)
                scores.update(batch_scores)
        except Exception as exc:
            logger.warning("BertuReranker.score_candidates failed: %s", exc)
            return {}

        return scores

    def rerank(
        self,
        sentence: str,
        token_index: int,
        candidates: list[str],
        *,
        tokens: list[str] | None = None,
        fallback_order: list[str] | None = None,
    ) -> list[str]:
        """
        Return candidates sorted by BERTu score (best first).

        Falls back to `fallback_order` (or the original `candidates` order)
        if BERTu is unavailable or scoring fails.
        """
        scores = self.score_candidates(
            sentence, token_index, candidates, tokens=tokens
        )
        if not scores:
            return list(fallback_order or candidates)
        return sorted(candidates, key=lambda c: scores.get(c, float("-inf")), reverse=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Attempt to load the BERTu model. Sets self._available."""
        try:
            import torch  # noqa: F401
            from transformers import AutoModelForMaskedLM, AutoTokenizer

            logger.info("BertuReranker: loading %s …", self._model_name)
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            self._model = AutoModelForMaskedLM.from_pretrained(self._model_name)
            self._model.eval()
            self._available = True
            logger.info("BertuReranker: model loaded successfully.")
        except Exception as exc:
            logger.warning(
                "BertuReranker: could not load %s (%s). "
                "Falling back to evidence-only scoring.",
                self._model_name,
                exc,
            )
            self._available = False

    def _score_batch(
        self,
        tokens: list[str],
        token_index: int,
        candidates: list[str],
    ) -> dict[str, float]:
        """
        Score a batch of candidates at `token_index` using masked LM.

        For each candidate we:
          1. Build a sentence with [MASK] at token_index.
          2. Tokenize the sentence.
          3. Find the position of [MASK] in the token ids.
          4. Run the model and read the log-softmax probability of the
             candidate's first subword token at the [MASK] position.

        This is an approximation — multi-subword candidates are scored only
        by their first subword. For most short Maltese words this is fine.
        """
        import torch
        import torch.nn.functional as F

        tokenizer = self._tokenizer
        model = self._model

        # Build masked sentence once
        masked_tokens = list(tokens)
        masked_tokens[token_index] = tokenizer.mask_token
        masked_sentence = " ".join(masked_tokens)

        # Tokenize masked sentence
        encoding = tokenizer(
            masked_sentence,
            return_tensors="pt",
            truncation=True,
            max_length=_MAX_SEQ_LEN,
        )
        input_ids = encoding["input_ids"]  # (1, seq_len)

        # Find [MASK] position in input_ids
        mask_token_id = tokenizer.mask_token_id
        mask_positions = (input_ids[0] == mask_token_id).nonzero(as_tuple=True)[0]
        if len(mask_positions) == 0:
            # Sentence was truncated past the mask — skip
            return {c: 0.0 for c in candidates}
        mask_pos = mask_positions[0].item()

        # Run model
        with torch.no_grad():
            logits = model(**encoding).logits  # (1, seq_len, vocab_size)

        log_probs = F.log_softmax(logits[0, mask_pos, :], dim=-1)  # (vocab_size,)

        scores: dict[str, float] = {}
        for candidate in candidates:
            # Tokenize just the candidate to get its first subword token id
            cand_ids = tokenizer(
                candidate,
                add_special_tokens=False,
            )["input_ids"]
            if not cand_ids:
                scores[candidate] = float("-inf")
                continue
            first_token_id = cand_ids[0]
            scores[candidate] = log_probs[first_token_id].item()

        return scores
