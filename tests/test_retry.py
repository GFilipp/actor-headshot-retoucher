"""_retry.call_with_backoff retries only transient API errors (429 / 5xx). Offline."""
from __future__ import annotations

import pytest

from retoucher import _retry


class _HTTPError(Exception):
    def __init__(self, code: int, msg: str = ""):
        super().__init__(msg or f"HTTP {code}")
        self.code = code


@pytest.fixture(autouse=True)
def sleeps(monkeypatch):
    calls: list[float] = []
    monkeypatch.setattr(_retry.time, "sleep", calls.append)
    return calls


def test_transient_error_is_retried_then_succeeds(sleeps):
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise _HTTPError(429)
        return "ok"

    assert _retry.call_with_backoff(flaky, retries=3, base=1.0) == "ok"
    assert calls["n"] == 3
    assert sleeps == [1.0, 2.0]                      # exponential backoff


def test_non_transient_error_raises_immediately(sleeps):
    calls = {"n": 0}

    def bad_request():
        calls["n"] += 1
        raise _HTTPError(400, "invalid argument")

    with pytest.raises(_HTTPError):
        _retry.call_with_backoff(bad_request, retries=3)
    assert calls["n"] == 1 and sleeps == []


def test_gives_up_after_retries_and_reraises(sleeps):
    calls = {"n": 0}

    def always_unavailable():
        calls["n"] += 1
        raise RuntimeError("503 UNAVAILABLE")

    with pytest.raises(RuntimeError, match="503"):
        _retry.call_with_backoff(always_unavailable, retries=2, base=0.5)
    assert calls["n"] == 3 and sleeps == [0.5, 1.0]
