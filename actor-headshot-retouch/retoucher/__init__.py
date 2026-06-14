"""Deterministic, identity-preserving headshot retouch pipeline.

The generative model proposes a retouch target; this package transfers only the
validated, local changes back onto the original high-resolution file, with
automated quality gates. See pipeline.retouch_image for the entry point.
"""
from __future__ import annotations

from .config import PipelineConfig, QAThresholds
from .generate import Generator, MockGenerator, OpenAIGenerator, get_generator
from .pipeline import RetouchResult, retouch_image, retouch_path

__version__ = "2.0.0-dev"

__all__ = [
    "PipelineConfig",
    "QAThresholds",
    "Generator",
    "MockGenerator",
    "OpenAIGenerator",
    "get_generator",
    "RetouchResult",
    "retouch_image",
    "retouch_path",
]
