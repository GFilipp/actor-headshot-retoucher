"""Generator backend tests that need no API key."""
from __future__ import annotations

import numpy as np

from retoucher.generate import MockGenerator, OpenAIGenerator, closest_gpt_image_size, get_generator


def test_closest_size_matches_aspect():
    assert closest_gpt_image_size(1024, 1024) == "1024x1024"
    assert closest_gpt_image_size(2000, 1000) == "1536x1024"   # landscape
    assert closest_gpt_image_size(1000, 2000) == "1024x1536"   # portrait
    assert closest_gpt_image_size(1200, 1500) == "1024x1536"   # 4:5 headshot -> portrait


def test_factory_returns_backends():
    assert isinstance(get_generator("mock"), MockGenerator)
    g = get_generator("openai", model="gpt-image-1")          # constructs without importing openai
    assert isinstance(g, OpenAIGenerator) and g.model == "gpt-image-1" and g.quality == "high"


def test_mock_generator_runs_offline():
    out = get_generator("mock").edit(np.zeros((32, 32, 3), np.float32), "x")
    assert out.shape == (32, 32, 3) and out.dtype == np.float32
