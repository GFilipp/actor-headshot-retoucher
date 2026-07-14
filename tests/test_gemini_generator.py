"""P2: Gemini backend behind the Generator seam + stochastic sampling + router.
No live API: these cover wiring, key resolution, and sampling (the edit() network
path is exercised manually with a real key)."""
from __future__ import annotations

import numpy as np
import pytest

from retoucher import router
from retoucher.generate import GeminiGenerator, MockGenerator, edit_n, get_generator


def test_factory_and_router_return_gemini():
    assert isinstance(get_generator("gemini"), GeminiGenerator)
    assert isinstance(get_generator("google"), GeminiGenerator)
    assert isinstance(router.pick("gemini"), GeminiGenerator)


def test_key_from_env(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-test-key")
    assert GeminiGenerator()._key() == "AIza-test-key"


def test_key_missing_raises(monkeypatch, tmp_path):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    g = GeminiGenerator(key_path=str(tmp_path / "nope.txt"))
    with pytest.raises(RuntimeError, match="No Gemini key"):
        g._key()


def test_edit_n_draws_n_candidates():
    calls = {"n": 0}

    def t(img):
        calls["n"] += 1
        return img

    out = edit_n(MockGenerator(transform=t), np.zeros((8, 8, 3), np.float32), "prompt", 3)
    assert len(out) == 3 and calls["n"] == 3


def test_edit_n_resilient_to_a_failed_sample():
    # One sample raising (transient API error) must not discard the good donors.
    calls = {"n": 0}

    def flaky(img):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("transient generator failure")
        return img

    out = edit_n(MockGenerator(transform=flaky), np.zeros((8, 8, 3), np.float32), "prompt", 3)
    assert len(out) == 3
    assert out[0] is not None and out[2] is not None and out[1] is None
