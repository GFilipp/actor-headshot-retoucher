"""Environment / robustness tests — the class of failure that slipped through
before: MediaPipe crashing in a sandbox, bad CLI input, RAW without rawpy."""
from __future__ import annotations

import pytest
from PIL import Image

import retoucher.faceparse as fp
from retoucher import MockGenerator, PipelineConfig
from retoucher.cli import _parse_marks
from retoucher.cli import main as cli_main
from retoucher.image_io import load, to_uint8
from retoucher.pipeline import retouch_image

from _synth import make_original


def _save(img, path):
    Image.fromarray(to_uint8(img)).save(path)


# --- face parser availability / probe ---------------------------------------

def test_env_off_disables_parser_without_subprocess(monkeypatch):
    monkeypatch.setenv("RETOUCH_FACE_PARSER", "off")
    monkeypatch.setattr(fp, "_probe", lambda: pytest.fail("probe must not run when off"))
    assert fp.available() is False
    assert fp.detect(make_original()) is None       # no in-process MediaPipe call


def test_env_on_skips_probe(monkeypatch):
    monkeypatch.setenv("RETOUCH_FACE_PARSER", "on")
    monkeypatch.setattr(fp, "_basic_available", lambda: True)
    monkeypatch.setattr(fp, "_probe", lambda: pytest.fail("probe must be skipped when on"))
    assert fp.available() is True


def test_probe_failure_degrades(monkeypatch):
    # Simulate the sandbox abort: the probe reports failure -> degrade, no crash.
    monkeypatch.delenv("RETOUCH_FACE_PARSER", raising=False)
    monkeypatch.setattr(fp, "_basic_available", lambda: True)
    monkeypatch.setattr(fp, "_probe", lambda: False)
    monkeypatch.setattr(fp, "_PROBE_OK", None)
    assert fp.available() is False


def test_detect_returns_none_when_unavailable(monkeypatch):
    monkeypatch.setattr(fp, "available", lambda: False)
    assert fp.detect(make_original()) is None


# --- degraded (no-geometry) pipeline path, i.e. the Codex path --------------

def test_pipeline_runs_without_face_geometry(tmp_path, monkeypatch, original, target):
    monkeypatch.setattr(fp, "detect", lambda rgb: None)   # force the degraded path
    src = tmp_path / "h.png"
    _save(original, src)
    res = retouch_image(src, tmp_path / "out", MockGenerator(transform=lambda _img: target), PipelineConfig())
    assert res.report["face_geometry"] is False
    assert res.output_path.exists()                       # ran to completion, no crash


# --- CLI input validation ----------------------------------------------------

def test_parse_marks_rejects_bad_input():
    with pytest.raises(ValueError):
        _parse_marks(["100"], [])           # missing Y
    with pytest.raises(ValueError):
        _parse_marks(["a,b"], [])           # non-numeric
    with pytest.raises(ValueError):
        _parse_marks([], ["10,20,30"])      # box needs 4


def test_cli_bad_mark_is_clean_error(tmp_path, capsys, original):
    src = tmp_path / "h.png"
    _save(original, src)
    rc = cli_main([str(src), "--dry-run", "--skip-preflight", "--out-dir", str(tmp_path / "o"), "--mark", "garbage"])
    assert rc == 1
    assert "Invalid mark" in capsys.readouterr().err     # clean message, not a traceback


def test_cli_empty_directory_reports_clearly(tmp_path, capsys):
    empty = tmp_path / "empty"
    empty.mkdir()
    rc = cli_main([str(empty), "--dry-run", "--skip-preflight", "--out-dir", str(tmp_path / "o")])
    assert rc == 1
    assert "No images" in capsys.readouterr().err


# --- RAW without the optional dependency ------------------------------------

def test_raw_without_rawpy_gives_clear_error(tmp_path):
    try:
        import rawpy  # noqa: F401
        pytest.skip("rawpy is installed; the missing-dep path can't be exercised")
    except ImportError:
        pass
    p = tmp_path / "shot.dng"
    p.write_bytes(b"not really raw")
    with pytest.raises(RuntimeError, match="rawpy"):
        load(p)
