"""Deterministic, identity-preserving headshot retouch pipeline.

The generative model proposes a retouch target; this package transfers only the
validated, local changes back onto the original high-resolution file, with
automated quality gates. See pipeline.retouch_image for the entry point.
"""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

from .config import PipelineConfig, QAThresholds
from .generate import Generator, MockGenerator, OpenAIGenerator, get_generator
from .pipeline import RetouchResult, retouch_image, retouch_path

# Single source of truth is pyproject.toml; read it from installed metadata.
try:
    __version__ = _pkg_version("actor-headshot-retoucher")
except PackageNotFoundError:  # running from a source tree without install
    __version__ = "0.0.0+source"

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
