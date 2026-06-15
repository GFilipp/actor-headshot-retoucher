"""CLI/preflight behavior that the README promises."""
from __future__ import annotations

from PIL import Image

from retoucher.cli import _preflight, main
from retoucher.image_io import to_uint8

from _synth import make_original


def test_preflight_accepts_batch_directory(tmp_path):
    shoot = tmp_path / "shoot"
    shoot.mkdir()
    out = tmp_path / "out"

    assert _preflight("hybrid-map", shoot, out, force=False)


def test_v3_engine_dry_run_offline_writes_report(tmp_path):
    # End-to-end CLI through the north-star engine, fully offline (mock generator).
    # The synthetic image has no detectable face -> graceful REFUSED, but the spine
    # runs and a telemetry report is always written (never a crash, never silent).
    src = tmp_path / "head.png"
    Image.fromarray(to_uint8(make_original())).save(src)
    out = tmp_path / "out"
    rc = main([str(src), "--engine", "v3", "--dry-run", "--out-dir", str(out), "--skip-preflight"])
    assert rc == 0
    assert (out / "head.report.json").exists()
