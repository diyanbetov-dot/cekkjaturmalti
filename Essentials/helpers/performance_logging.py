import copy
import logging
import os
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any


LOGGER = logging.getLogger("spellcheck.performance")
if not LOGGER.handlers:
    logging.basicConfig(level=logging.INFO)


def _env_bool(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: str) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return float(default)


@dataclass
class RequestProfiler:
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    profile_enabled: bool = field(default_factory=lambda: _env_bool("SPELLCHECK_PROFILE"))
    slow_stage_ms: float = field(
        default_factory=lambda: _env_float("SPELLCHECK_SLOW_STAGE_MS", "100")
    )
    slow_request_ms: float = field(
        default_factory=lambda: _env_float("SPELLCHECK_SLOW_REQUEST_MS", "1000")
    )
    start_time: float = field(default_factory=time.perf_counter)
    cache: dict[str, dict[Any, Any]] = field(default_factory=dict)
    cache_hits: dict[str, int] = field(default_factory=dict)
    cache_misses: dict[str, int] = field(default_factory=dict)
    counters: dict[str, int] = field(default_factory=dict)

    def cache_get(self, namespace: str, key: Any) -> tuple[bool, Any]:
        bucket = self.cache.setdefault(namespace, {})
        if key in bucket:
            self.cache_hits[namespace] = self.cache_hits.get(namespace, 0) + 1
            value = bucket[key]
            return True, copy.deepcopy(value)
        self.cache_misses[namespace] = self.cache_misses.get(namespace, 0) + 1
        return False, None

    def cache_set(self, namespace: str, key: Any, value: Any) -> Any:
        self.cache.setdefault(namespace, {})[key] = copy.deepcopy(value)
        return copy.deepcopy(value)

    def increment(self, name: str, amount: int = 1) -> None:
        self.counters[name] = self.counters.get(name, 0) + amount

    def log_stage(self, stage: str, elapsed_ms: float, **fields: Any) -> None:
        if not self.profile_enabled and elapsed_ms < self.slow_stage_ms:
            return
        log_spellcheck_event(
            request_id=self.request_id,
            stage=stage,
            elapsed_ms=elapsed_ms,
            **fields,
        )

    @contextmanager
    def span(self, stage: str, **fields: Any):
        started = time.perf_counter()
        try:
            yield
        finally:
            self.log_stage(stage, (time.perf_counter() - started) * 1000, **fields)

    def finish(self, *, token_count: int, unique_tokens: int) -> None:
        elapsed_ms = (time.perf_counter() - self.start_time) * 1000
        if self.profile_enabled or elapsed_ms >= self.slow_request_ms:
            log_spellcheck_event(
                request_id=self.request_id,
                stage="request_complete",
                elapsed_ms=elapsed_ms,
                tokens=token_count,
                unique_tokens=unique_tokens,
                cache_hits=sum(self.cache_hits.values()),
                cache_misses=sum(self.cache_misses.values()),
                distance_calls=self.counters.get("distance_calls", 0),
                candidates_generated=self.counters.get("candidates_generated", 0),
                candidates_scored=self.counters.get("candidates_scored", 0),
            )


_CURRENT_PROFILER: ContextVar[RequestProfiler | None] = ContextVar(
    "spellcheck_request_profiler",
    default=None,
)


def current_profiler() -> RequestProfiler | None:
    return _CURRENT_PROFILER.get()


def set_current_profiler(profiler: RequestProfiler | None):
    return _CURRENT_PROFILER.set(profiler)


def reset_current_profiler(token) -> None:
    _CURRENT_PROFILER.reset(token)


def log_spellcheck_event(**fields: Any) -> None:
    parts = ["SPELLCHECK"]
    for key, value in fields.items():
        if value is None:
            continue
        if isinstance(value, str):
            rendered = repr(value)
        else:
            rendered = str(value)
        parts.append(f"{key}={rendered}")
    LOGGER.info(" ".join(parts))
