# -*- coding: utf-8 -*-
"""
Runtime Corpus Scorer using preprocessed Korpus Malti indexes.

Loads compact unigram, bigram, and trigram index files. If index files are missing
or invalid, logs a warning instruction and gracefully disables corpus scoring,
allowing the spellchecker and BERTu to proceed normally.
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import math
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_CORPUS_DIR = Path(__file__).resolve().parent.parent / "corpus"


class MalteseCorpusScorer:
    """
    Lightweight runtime scorer based on Korpus Malti frequency statistics.
    """

    def __init__(
        self,
        corpus_dir: Optional[Path] = None,
        enabled: bool = True,
        shadow_mode: bool = False,
        unigram_enabled: bool = True,
        bigram_enabled: bool = True,
        trigram_enabled: bool = False,
        max_score_contribution: float = 0.25,
    ) -> None:
        self.corpus_dir = Path(corpus_dir or DEFAULT_CORPUS_DIR)
        self.enabled = bool(enabled)
        self.shadow_mode = bool(shadow_mode)
        self.unigram_enabled = bool(unigram_enabled)
        self.bigram_enabled = bool(bigram_enabled)
        self.trigram_enabled = bool(trigram_enabled)
        self.max_score_contribution = max(0.0, float(max_score_contribution))

        self.available = False
        self.meta: dict = {}
        self.unigrams: Dict[str, float] = {}
        self.bigrams: Dict[str, Dict[str, float]] = {}
        self.trigrams: Dict[str, Dict[str, float]] = {}
        self.status: str = "CORPUS_SCORER_DISABLED_BY_CONFIG"
        self.status_reason: str = ""
        self.expected_index_paths: Dict[str, Path] = {}

        self._max_unigram_log: float = 1.0

        if self.enabled:
            self._load_indexes()
        else:
            self.status_reason = "Corpus scoring is disabled by configuration."

    def _load_indexes(self) -> None:
        meta_file = self.corpus_dir / "meta.json"
        unigrams_file = self.corpus_dir / "unigrams.json.gz"
        bigrams_file = self.corpus_dir / "bigrams.json.gz"
        trigrams_file = self.corpus_dir / "trigrams.json.gz"
        self.expected_index_paths = {
            "meta": meta_file,
            "unigrams": unigrams_file,
            "bigrams": bigrams_file,
            "trigrams": trigrams_file,
        }

        missing_files = [
            name
            for name, path in self.expected_index_paths.items()
            if name != "trigrams" and not path.exists()
        ]
        if missing_files:
            self.status = "CORPUS_SCORER_MISSING_INDEXES"
            self.status_reason = (
                f"Missing required corpus index files: {', '.join(missing_files)}. "
                f"Expected directory: {self.corpus_dir}."
            )
            logger.warning(
                "\n" + "=" * 70 + "\n"
                "CORPUS_SCORER_MISSING_INDEXES\n"
                f"Expected location: {self.corpus_dir}\n"
                f"Missing files: {', '.join(missing_files)}\n"
                "Corpus candidate evidence scoring will be DISABLED.\n\n"
                "To enable corpus evidence, please run the offline setup script:\n"
                "    python tools/setup_korpus_malti.py\n" + "=" * 70
            )
            self.available = False
            return

        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                self.meta = json.load(f)

            with gzip.open(unigrams_file, "rt", encoding="utf-8") as f:
                self.unigrams = json.load(f)

            with gzip.open(bigrams_file, "rt", encoding="utf-8") as f:
                self.bigrams = json.load(f)

            if trigrams_file.exists() and self.trigram_enabled:
                with gzip.open(trigrams_file, "rt", encoding="utf-8") as f:
                    self.trigrams = json.load(f)

            malformed_keys = [
                key
                for key in list(self.unigrams)[:2000]
                if "\t" in key or "|" in key
            ]
            if malformed_keys:
                self.status = "CORPUS_SCORER_MALFORMED_INDEX"
                self.status_reason = (
                    "Corpus unigram keys contain unsplit vertical-corpus rows. "
                    "Rebuild the indexes with tools/setup_korpus_malti.py."
                )
                logger.warning("%s: examples=%r", self.status, malformed_keys[:3])
                self.available = False
                self.unigrams = {}
                self.bigrams = {}
                self.trigrams = {}
                return

            if self.unigrams:
                self._max_unigram_log = max(self.unigrams.values()) or 1.0

            self.available = True
            self.status = "CORPUS_SCORER_READY"
            self.status_reason = (
                f"Loaded {len(self.unigrams)} unigrams and {len(self.bigrams)} bigram roots "
                f"from {self.corpus_dir}."
            )
            logger.info(
                "CORPUS_SCORER_READY: %s | %s",
                self.status_reason,
                ", ".join(f"{name}={path}" for name, path in self.expected_index_paths.items()),
            )
        except Exception as exc:
            self.status = "CORPUS_SCORER_LOAD_FAILED"
            self.status_reason = f"Failed to load Korpus Malti indexes from {self.corpus_dir}: {exc}."
            logger.warning(
                "CORPUS_SCORER_LOAD_FAILED: %s | expected=%s",
                self.status_reason,
                self.expected_index_paths,
            )
            self.available = False

    def is_available(self) -> bool:
        return self.enabled and self.available

    def get_status_details(self) -> Dict[str, object]:
        return {
            "status": self.status,
            "available": self.is_available(),
            "reason": self.status_reason,
            "corpus_dir": str(self.corpus_dir),
            "expected_paths": {name: str(path) for name, path in self.expected_index_paths.items()},
        }

    def score_candidate(
        self,
        candidate: str,
        *,
        prev_word: Optional[str] = None,
        next_word: Optional[str] = None,
        prev_prev_word: Optional[str] = None,
    ) -> float:
        """
        Calculate a bounded bonus in range [0.0, max_score_contribution].
        Absence from corpus yields 0.0 (neutral), never a penalty.
        """
        components = self.score_candidate_components(
            candidate,
            prev_word=prev_word,
            next_word=next_word,
            prev_prev_word=prev_prev_word,
        )
        final_bonus = float(components["total_after_cap"])

        if self.shadow_mode:
            logger.debug(f"[CorpusShadow] candidate={candidate} bonus={final_bonus}")
            return 0.0

        return final_bonus

    @staticmethod
    def _normalize_surface_token(value: Optional[str]) -> str:
        return str(value or "").strip().lower()

    @staticmethod
    def _surface_tokens(candidate: str) -> List[str]:
        """
        Convert an application surface such as ``l-iktar`` or ``'l ommi``
        into the tokenization used by the MLRS vertical corpus.
        """
        surface = str(candidate or "").strip().lower()
        if not surface:
            return []

        result: List[str] = []
        prefixes = (
            "għall-", "bħall-", "mill-", "mall-", "tal-", "mal-", "fil-",
            "bil-", "lill-", "għad-", "għan-", "għar-", "għas-", "għat-",
            "għax-", "għaż-", "għaċ-", "għaż-", "mid-", "min-", "mir-",
            "mis-", "mit-", "mix-", "miż-", "miċ-", "fid-", "fin-",
            "fir-", "fis-", "fit-", "fix-", "fiż-", "fiċ-", "bid-", "bin-",
            "bir-", "bis-", "bit-", "bix-", "biż-", "biċ-", "tad-", "tan-",
            "tar-", "tas-", "tat-", "tax-", "taż-", "taċ-", "id-", "in-",
            "ir-", "is-", "it-", "ix-", "iż-", "iċ-", "il-", "l-",
        )
        for chunk in surface.split():
            if chunk.startswith("'l-") and len(chunk) > 3:
                result.extend(["'l", chunk[3:]])
                continue
            if chunk.startswith("'il-") and len(chunk) > 4:
                result.extend(["'il", chunk[4:]])
                continue
            matched = False
            for prefix in prefixes:
                if chunk.startswith(prefix) and len(chunk) > len(prefix):
                    result.extend([prefix, chunk[len(prefix):]])
                    matched = True
                    break
            if not matched:
                result.append(chunk)
        return [token for token in result if token]

    def _ratio(self, value: float) -> float:
        return max(0.0, min(1.0, float(value) / self._max_unigram_log))

    def _bigram_ratio(self, left: str, right: str) -> float:
        value = self.bigrams.get(left, {}).get(right)
        return self._ratio(value) if value is not None else 0.0

    def score_candidate_components(
        self,
        candidate: str,
        *,
        prev_word: Optional[str] = None,
        next_word: Optional[str] = None,
        prev_prev_word: Optional[str] = None,
    ) -> Dict[str, object]:
        """
        Return inspectable corpus evidence. Missing n-grams are neutral.

        Multi-token surfaces are averaged internally so a longer article or
        preposition candidate does not win merely because it contains more
        scoreable pieces.
        """
        empty = {
            "candidate": candidate,
            "tokens": [],
            "unigram": 0.0,
            "left_bigram": 0.0,
            "internal_bigram": 0.0,
            "right_bigram": 0.0,
            "trigram": 0.0,
            "evidence_count": 0,
            "total_before_cap": 0.0,
            "total_after_cap": 0.0,
        }
        if not self.is_available():
            return empty

        tokens = self._surface_tokens(candidate)
        if not tokens:
            return empty

        uni_ratios = [
            self._ratio(self.unigrams[token])
            for token in tokens
            if token in self.unigrams
        ]
        unigram = sum(uni_ratios) / len(uni_ratios) if uni_ratios else 0.0

        internal_ratios = [
            self._bigram_ratio(left, right)
            for left, right in zip(tokens, tokens[1:])
        ]
        internal_ratios = [value for value in internal_ratios if value > 0.0]
        internal = (
            sum(internal_ratios) / len(internal_ratios)
            if internal_ratios
            else 0.0
        )

        prev_norm = self._normalize_surface_token(prev_word)
        next_norm = self._normalize_surface_token(next_word)
        prev_prev_norm = self._normalize_surface_token(prev_prev_word)
        left_bigram = (
            self._bigram_ratio(prev_norm, tokens[0])
            if self.bigram_enabled and prev_norm
            else 0.0
        )
        right_bigram = (
            self._bigram_ratio(tokens[-1], next_norm)
            if self.bigram_enabled and next_norm
            else 0.0
        )

        trigram = 0.0
        if self.trigram_enabled and prev_prev_norm and prev_norm:
            tri_key = f"{prev_prev_norm} {prev_norm}"
            tri_value = self.trigrams.get(tri_key, {}).get(tokens[0])
            if tri_value is not None:
                trigram = self._ratio(tri_value)

        weighted = 0.0
        if self.unigram_enabled:
            weighted += unigram * (self.max_score_contribution * 0.30)
        if self.bigram_enabled:
            weighted += left_bigram * (self.max_score_contribution * 0.30)
            weighted += internal * (self.max_score_contribution * 0.20)
            weighted += right_bigram * (self.max_score_contribution * 0.15)
        if self.trigram_enabled:
            weighted += trigram * (self.max_score_contribution * 0.20)

        evidence_count = sum(
            value > 0.0
            for value in (unigram, left_bigram, internal, right_bigram, trigram)
        )
        result = dict(empty)
        result.update(
            {
                "tokens": tokens,
                "unigram": round(unigram, 6),
                "left_bigram": round(left_bigram, 6),
                "internal_bigram": round(internal, 6),
                "right_bigram": round(right_bigram, 6),
                "trigram": round(trigram, 6),
                "evidence_count": evidence_count,
                "total_before_cap": round(weighted, 6),
                "total_after_cap": round(
                    min(self.max_score_contribution, weighted),
                    6,
                ),
            }
        )
        return result
