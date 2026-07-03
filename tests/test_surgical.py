"""The surgical engine — the proven one-region recipe, runnable and tested offline
(mock generator + synthetic geometry)."""
from __future__ import annotations

import numpy as np

from PIL import Image

from retoucher.cli import main
from retoucher.generate import MockGenerator
from retoucher.image_io import to_uint8
from retoucher.regions import build_region
from retoucher.surgical import SurgicalResult, surgical_retouch

from _synth import fake_geometry, make_original


def _bright(img):
    return np.clip(img + 0.12, 0, 1)


def test_surgical_edits_region_and_leaves_background_alone():
    rgb = make_original()
    geom = fake_geometry()
    # feather=30: the 80px default legitimately reaches the corner on this small synthetic
    # frame, which would make the background assertion meaningless.
    res = surgical_retouch(rgb, generator=MockGenerator(transform=_bright), geom=geom,
                           mode="paste", feather=30.0, samples=1)
    assert isinstance(res, SurgicalResult) and res.handleable
    assert res.image.shape == rgb.shape
    region = build_region(geom, "periorbital", grow=0.7) > 0.5
    diff = np.abs(res.image - rgb).mean(axis=2)
    # color_match deliberately cancels most of the donor's uniform lift, so the residual
    # regional edit is modest. The background is untouched up to the polish ops' full-image
    # LAB round-trip epsilon (~6e-4); the region must move far above that.
    assert diff[region].mean() > 0.004
    corner = diff[:24, :24]
    assert corner.mean() < 0.002
    assert diff[region].mean() > 5 * corner.mean()
    assert res.verdict is not None and res.report["engine"] == "surgical"


def test_surgical_picks_cleanest_of_k_donors():
    rgb = make_original()
    geom = fake_geometry()
    calls = {"n": 0}

    def stochastic(img):
        calls["n"] += 1
        if calls["n"] == 1:                                 # bad donor: hard bright square
            bad = img.copy()
            bad[80:180, 80:180] = 1.0
            return bad
        return np.clip(img + 0.05, 0, 1)                    # good donor: smooth lift

    res = surgical_retouch(rgb, generator=MockGenerator(transform=stochastic), geom=geom,
                           mode="paste", samples=2)
    assert calls["n"] == 2
    scores = res.report["candidate_scores"]
    assert res.report["selected"] == int(np.argmax(scores))  # audit picked the cleanest


def test_surgical_refuses_gracefully_without_a_face():
    blank = np.full((300, 300, 3), 0.5, np.float32)
    res = surgical_retouch(blank, generator=MockGenerator(), samples=1)
    assert res.handleable is False and res.verdict is None
    assert np.allclose(res.image, blank)                    # original handed back untouched


def test_cli_surgical_dry_run_writes_report(tmp_path):
    src = tmp_path / "head.png"
    Image.fromarray(to_uint8(make_original())).save(src)
    out = tmp_path / "out"
    rc = main([str(src), "--engine", "surgical", "--dry-run",
               "--out-dir", str(out), "--skip-preflight"])
    assert rc == 0
    assert (out / "head.surgical.report.json").exists()     # refusal still reports
