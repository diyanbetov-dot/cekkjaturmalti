import time
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ContextAnalysis:
    enabled: bool
    backend: str
    available: bool
    elapsed_ms: float
    token_count: int
    sentence_count: int
    error: str = ""


class OptionalSentenceContextAnalyzer:
    """Optional Stanza/UDPipe wrapper used only for shadow diagnostics.

    The analyzer never rewrites text. It reports whether a backend is available
    and how expensive one pass over the corrected sentence would be.
    """

    def __init__(self, *, backend: str = "stanza", enabled: bool = False) -> None:
        self.backend = str(backend or "stanza").lower()
        self.enabled = bool(enabled)
        self._pipeline: Any | None = None
        self._load_error = ""
        if self.enabled:
            self._try_load()

    def _try_load(self) -> None:
        try:
            if self.backend == "stanza":
                import stanza  # type: ignore

                self._pipeline = stanza.Pipeline(
                    lang="mt",
                    processors="tokenize,pos,lemma",
                    tokenize_no_ssplit=False,
                    verbose=False,
                )
            elif self.backend == "udpipe":
                try:
                    import ufal.udpipe  # type: ignore  # noqa: F401
                except ImportError:
                    import spacy_udpipe  # type: ignore

                    spacy_udpipe.download("mt")
                    self._pipeline = spacy_udpipe.load("mt")
            else:
                self._load_error = f"unsupported backend: {self.backend}"
        except Exception as exc:
            self._pipeline = None
            self._load_error = f"{type(exc).__name__}: {exc}"

    def analyze(self, text: str) -> ContextAnalysis:
        if not self.enabled:
            return ContextAnalysis(False, self.backend, False, 0.0, 0, 0)
        if self._pipeline is None:
            return ContextAnalysis(
                True,
                self.backend,
                False,
                0.0,
                0,
                0,
                self._load_error or "backend unavailable",
            )

        started = time.perf_counter()
        try:
            document = self._pipeline(text)
            elapsed_ms = (time.perf_counter() - started) * 1000
            if self.backend == "stanza":
                sentences = getattr(document, "sentences", []) or []
                token_count = sum(len(getattr(sentence, "words", []) or []) for sentence in sentences)
                return ContextAnalysis(True, self.backend, True, elapsed_ms, token_count, len(sentences))
            sentences = list(getattr(document, "sents", []) or [])
            return ContextAnalysis(
                True,
                self.backend,
                True,
                elapsed_ms,
                len(document),
                len(sentences) or 1,
            )
        except Exception as exc:
            return ContextAnalysis(
                True,
                self.backend,
                False,
                (time.perf_counter() - started) * 1000,
                0,
                0,
                f"{type(exc).__name__}: {exc}",
            )
