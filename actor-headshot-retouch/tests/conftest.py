"""Shared fixtures. Also makes the package importable without an editable
install, so `pytest` works straight from a clone."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _synth import make_original, make_target  # noqa: E402


@pytest.fixture
def original():
    return make_original()


@pytest.fixture
def target():
    return make_target()
