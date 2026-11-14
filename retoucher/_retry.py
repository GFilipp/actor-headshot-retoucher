"""Small retry-with-backoff wrapper for transient API failures (HTTP 429 / 5xx).

The Gemini image + vision calls are single network round-trips with no retry; a
transient rate-limit or 5xx would otherwise surface as a hard failure. This retries
only transient errors, with exponential backoff, and re-raises everything else."""
from __future__ import annotations

import time
from typing import Callable, TypeVar

T = TypeVar("T")

_TRANSIENT = {429, 500, 502, 503, 504}


def _is_transient(exc: Exception) -> bool:
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if isinstance(code, int) and code in _TRANSIENT:
        return True
    s = str(exc).lower()
    return any(k in s for k in ("429", "rate limit", "resource_exhausted",
                                "unavailable", "internal error", "503", "502", "504"))


def call_with_backoff(fn: Callable[[], T], retries: int = 3, base: float = 1.0) -> T:
    """Call fn(); on a transient error retry up to ``retries`` times with exponential
    backoff. Non-transient errors and the final attempt propagate unchanged."""
    for attempt in range(retries + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - re-raised unless transient
            if attempt < retries and _is_transient(exc):
                time.sleep(base * (2 ** attempt))
                continue
            raise
