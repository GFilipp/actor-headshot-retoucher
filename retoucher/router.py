"""Model router — model-agnostic backend selection (the seam the README promised).

Picks a generator by backend name via the `generate` factory. A per-job policy hook
(fitness scoring per region/defect) can slot in here later without touching callers,
so "best model per job" stays pluggable.
"""
from __future__ import annotations

from .generate import Generator, get_generator

BACKENDS = ("mock", "openai", "gemini")


def pick(backend: str = "gemini", **kwargs) -> Generator:
    """Return a generator for the requested backend (mock | openai | gemini)."""
    return get_generator(backend, **kwargs)
