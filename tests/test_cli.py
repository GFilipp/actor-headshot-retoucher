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


def test_v3_max_process_mp_reaches_engine(monkeypatch, tmp_path):
    # The flag was silently ignored on the v3 path (footgun): _run_v3 must pass pipe_cfg.
    src = tmp_path / "head.png"
    Image.fromarray(to_uint8(make_original())).save(src)
    captured = {}

    def fake_retouch(rgb, **kw):
        captured.update(kw)
        raise RuntimeError("captured; stop here")

    monkeypatch.setattr("retoucher.orchestrator.retouch", fake_retouch)
    main([str(src), "--engine", "v3", "--dry-run", "--max-process-mp", "2.5",
          "--out-dir", str(tmp_path / "out"), "--skip-preflight"])
    assert captured["pipe_cfg"].max_process_mp == 2.5


def test_v3_rejects_nonfinite_max_process_mp(tmp_path):
    # NaN slips past a bare `<= 0` guard (NaN comparisons are False) and later crashes resize.
    src = tmp_path / "head.png"
    Image.fromarray(to_uint8(make_original())).save(src)
    rc = main([str(src), "--engine", "v3", "--dry-run", "--max-process-mp", "nan",
               "--out-dir", str(tmp_path / "out"), "--skip-preflight"])
    assert rc == 1                                            # clean error, not a crash


def test_v3_force_writes_image_even_when_not_delivered(tmp_path):
    # --force writes the result for inspection even when the audit doesn't pass (here the
    # synthetic image refuses). Default (no --force) writes only the report, never the image.
    src = tmp_path / "head.png"
    Image.fromarray(to_uint8(make_original())).save(src)
    out = tmp_path / "out"
    rc = main([str(src), "--engine", "v3", "--dry-run", "--force", "--out-dir", str(out), "--skip-preflight"])
    assert rc == 0
    assert list(out.glob("*.jpg")), "expected a forced output image"
