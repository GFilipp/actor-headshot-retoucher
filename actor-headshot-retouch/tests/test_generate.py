"""Generator backend tests that need no API key or network."""
from __future__ import annotations

import types

import numpy as np

from retoucher.generate import (
    FALLBACK_OPENAI_IMAGE_MODEL,
    MockGenerator,
    OpenAIGenerator,
    closest_gpt_image_size,
    discover_latest_image_model,
    get_generator,
)


def _fake_client(ids_created):
    data = [types.SimpleNamespace(id=i, created=c) for i, c in ids_created]
    models = types.SimpleNamespace(list=lambda: types.SimpleNamespace(data=data))
    return types.SimpleNamespace(models=models)


def test_closest_size_matches_aspect():
    assert closest_gpt_image_size(1024, 1024) == "1024x1024"
    assert closest_gpt_image_size(2000, 1000) == "1536x1024"   # landscape
    assert closest_gpt_image_size(1000, 2000) == "1024x1536"   # portrait
    assert closest_gpt_image_size(1200, 1500) == "1024x1536"   # 4:5 headshot -> portrait


def test_factory_returns_backends():
    assert isinstance(get_generator("mock"), MockGenerator)
    g = get_generator("openai")
    assert isinstance(g, OpenAIGenerator)
    assert g.pinned_model is None        # unpinned -> auto-discovers latest at call time
    assert g.quality == "high"


def test_pinning_via_arg_or_env(monkeypatch):
    monkeypatch.delenv("OPENAI_IMAGE_MODEL", raising=False)
    assert OpenAIGenerator().pinned_model is None                       # default = auto-discover
    assert OpenAIGenerator(model="gpt-image-9").model == "gpt-image-9"  # arg pins
    monkeypatch.setenv("OPENAI_IMAGE_MODEL", "gpt-image-3")
    assert OpenAIGenerator().model == "gpt-image-3"                     # env pins, no code change


def test_discovery_auto_adopts_newest_non_mini(monkeypatch):
    monkeypatch.delenv("OPENAI_IMAGE_MODEL", raising=False)
    client = _fake_client([
        ("gpt-image-1", 100), ("gpt-image-2", 200), ("gpt-image-2-mini", 250),
        ("dall-e-3", 300), ("gpt-image-3", 400),
    ])
    # Newest gpt-image (a future gpt-image-3) wins; mini and dall-e excluded.
    assert discover_latest_image_model(client) == "gpt-image-3"
    assert OpenAIGenerator()._model_for(client) == "gpt-image-3"


def test_discovery_falls_back_when_api_unreachable(monkeypatch):
    monkeypatch.delenv("OPENAI_IMAGE_MODEL", raising=False)

    def boom():
        raise RuntimeError("network down")

    broken = types.SimpleNamespace(models=types.SimpleNamespace(list=boom))
    assert OpenAIGenerator()._model_for(broken) == FALLBACK_OPENAI_IMAGE_MODEL


def test_mock_generator_runs_offline():
    out = get_generator("mock").edit(np.zeros((32, 32, 3), np.float32), "x")
    assert out.shape == (32, 32, 3) and out.dtype == np.float32
