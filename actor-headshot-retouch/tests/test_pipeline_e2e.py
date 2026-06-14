"""End-to-end: run the whole pipeline offline with the mock generator."""
from __future__ import annotations

import json

import numpy as np
from PIL import Image

from retoucher import MockGenerator, PipelineConfig
from retoucher.image_io import to_uint8
from retoucher.pipeline import retouch_image, retouch_path

from _synth import CORNER, UNDER_EYE


def _save(img, path):
    Image.fromarray(to_uint8(img)).save(path)


def test_e2e_writes_outputs_and_passes(tmp_path, original, target):
    src = tmp_path / "head.png"
    _save(original, src)
    gen = MockGenerator(transform=lambda _img: target)

    res = retouch_image(src, tmp_path / "out", gen, PipelineConfig(mode="hybrid-map"))

    assert res.output_path.exists()
    assert res.contact_sheet_path.exists()
    assert res.report_path.exists()

    report = json.loads(res.report_path.read_text())
    assert report["qa"]["verdict"] == "pass", report
    assert report["alignment"]["method"]

    # The edit is real and localized.
    assert not np.allclose(res.result_pixels, original, atol=1e-3)
    reg = UNDER_EYE[0]
    assert res.result_pixels[reg[0], reg[1], 0].mean() < original[reg[0], reg[1], 0].mean() - 0.01
    assert float(np.abs(res.result_pixels[CORNER] - original[CORNER]).mean()) < 0.02


def test_e2e_dry_run_no_write(tmp_path, original):
    src = tmp_path / "head.png"
    _save(original, src)
    res = retouch_image(src, tmp_path / "out", MockGenerator(), PipelineConfig(), write=False)
    assert res.output_path is None
    assert res.report["mode"] == "hybrid-map"
    assert res.result_pixels.shape == original.shape


def test_batch_directory(tmp_path, original):
    shoot = tmp_path / "shoot"
    shoot.mkdir()
    for i in range(3):
        _save(original, shoot / f"h{i}.png")
    results = retouch_path(shoot, tmp_path / "out", MockGenerator(), PipelineConfig(), write=True)
    assert len(results) == 3
    assert all(r.output_path.exists() for r in results)
