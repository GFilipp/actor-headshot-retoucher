"""CLI/preflight behavior that the README promises."""
from __future__ import annotations

from retoucher.cli import _preflight


def test_preflight_accepts_batch_directory(tmp_path):
    shoot = tmp_path / "shoot"
    shoot.mkdir()
    out = tmp_path / "out"

    assert _preflight("hybrid-map", shoot, out, force=False)
