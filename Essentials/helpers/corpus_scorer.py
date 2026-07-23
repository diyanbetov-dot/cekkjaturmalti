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
from typing import Dict, List, Optional, Tuple

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
        if not self.is_available():
            return 0.0

        cand_norm = candidate.strip().lower()
        if not cand_norm:
            return 0.0

        bonus = 0.0

        # 1. Unigram log freq bonus (up to 40% of max contribution)
        if self.unigram_enabled and cand_norm in self.unigrams:
            uni_log = self.unigrams[cand_norm]
            # Ratio of unigram log freq vs max observed log freq
            ratio = max(0.0, min(1.0, uni_log / self._max_unigram_log))
            bonus += ratio * (self.max_score_contribution * 0.4)

        # 2. Bigram bonus (prev_word -> candidate) (up to 35% of max contribution)
        if self.bigram_enabled and prev_word:
            p_norm = prev_word.strip().lower()
            if p_norm in self.bigrams and cand_norm in self.bigrams[p_norm]:
                bi_log = self.bigrams[p_norm][cand_norm]
                ratio = max(0.0, min(1.0, bi_log / self._max_unigram_log))
                bonus += ratio * (self.max_score_contribution * 0.35)

        # 3. Bigram bonus (candidate -> next_word) (up to 25% of max contribution)
        if self.bigram_enabled and next_word:
            n_norm = next_word.strip().lower()
            if cand_norm in self.bigrams and n_norm in self.bigrams[cand_norm]:
                bi_log = self.bigrams[cand_norm][n_norm]
                ratio = max(0.0, min(1.0, bi_log / self._max_unigram_log))
                bonus += ratio * (self.max_score_contribution * 0.25)

        # 4. Trigram bonus (prev_prev -> prev -> candidate)
        if self.trigram_enabled and prev_prev_word and prev_word:
            tri_key = f"{prev_prev_word.strip().lower()} {prev_word.strip().lower()}"
            if tri_key in self.trigrams and cand_norm in self.trigrams[tri_key]:
                tri_log = self.trigrams[tri_key][cand_norm]
                ratio = max(0.0, min(1.0, tri_log / self._max_unigram_log))
                bonus += ratio * (self.max_score_contribution * 0.2)

        final_bonus = round(min(self.max_score_contribution, bonus), 4)

        if self.shadow_mode:
            logger.debug(f"[CorpusShadow] candidate={candidate} bonus={final_bonus}")
            return 0.0

        return final_bonus
