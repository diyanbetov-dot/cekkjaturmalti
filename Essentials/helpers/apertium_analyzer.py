import os
import subprocess
import time
from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class ApertiumAnalysis:
    enabled: bool
    available: bool
    elapsed_ms: float
    analyses: tuple[str, ...] = ()
    error: str = ""


class OptionalApertiumAnalyzer:
    """Optional Apertium analyzer used as candidate evidence, not authority."""

    def __init__(
        self,
        *,
        enabled: bool = False,
        command: str | None = None,
        timeout_sec: float = 1.5,
    ) -> None:
        self.enabled = bool(enabled)
        self.command = command or os.environ.get(
            "SPELLCHECK_APERTIUM_COMMAND",
            "apertium -d . mt-morph",
        )
        self.timeout_sec = float(timeout_sec)

    @lru_cache(maxsize=4096)
    def analyze(self, word: str) -> ApertiumAnalysis:
        if not self.enabled:
            return ApertiumAnalysis(False, False, 0.0)
        normalized = str(word or "").strip()
        if not normalized:
            return ApertiumAnalysis(True, False, 0.0)

        started = time.perf_counter()
        try:
            completed = subprocess.run(
                self.command,
                input=normalized + "\n",
                capture_output=True,
                encoding="utf-8",
                errors="replace",
                shell=True,
                timeout=self.timeout_sec,
            )
        except Exception as exc:
            return ApertiumAnalysis(
                True,
                False,
                (time.perf_counter() - started) * 1000,
                error=f"{type(exc).__name__}: {exc}",
            )

        elapsed_ms = (time.perf_counter() - started) * 1000
        if completed.returncode != 0:
            return ApertiumAnalysis(
                True,
                False,
                elapsed_ms,
                error=(completed.stderr or f"exit {completed.returncode}").strip(),
            )
        raw = (completed.stdout or "").strip()
        analyses = tuple(part for part in raw.replace("\n", " ").split("/") if part)
        available = bool(analyses and not raw.startswith("*"))
        return ApertiumAnalysis(True, available, elapsed_ms, analyses=analyses)
