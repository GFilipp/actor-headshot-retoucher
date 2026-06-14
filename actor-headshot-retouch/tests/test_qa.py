"""Quality-gate tests: a good retouch passes; invisible and plastic results are
rejected; optional gates report 'skipped' rather than silently passing."""
from __future__ import annotations

import cv2

from retoucher.blend import composite
from retoucher.config import PipelineConfig
from retoucher.diff import compute_touch_up_map
from retoucher.mask import build_masks
from retoucher.qa import run_qa


def _retouch(original, target, cfg):
    tmap = compute_touch_up_map(original, target, cfg.freq_sigma)
    masks = build_masks(original, tmap, cfg)
    return composite(original, tmap, masks, cfg), masks


def test_qa_passes_a_good_retouch(original, target):
    cfg = PipelineConfig()
    result, masks = _retouch(original, target, cfg)
    report = run_qa(original, result, masks, cfg)
    statuses = {g.name: g.status for g in report.gates}
    assert statuses["edited_delta_e"] == "pass", report.to_dict()
    assert statuses["texture"] == "pass", report.to_dict()
    assert statuses["untouched_ssim"] == "pass", report.to_dict()
    # No optional backends in the core env -> reported as skipped, not passed.
    assert statuses["identity"] == "skipped"
    assert statuses["untouched_lpips"] == "skipped"
    assert report.verdict == "pass", report.to_dict()


def test_qa_rejects_invisible_edit(original, target):
    cfg = PipelineConfig()
    _, masks = _retouch(original, target, cfg)
    report = run_qa(original, original.copy(), masks, cfg)
    assert any(g.name == "edited_delta_e" and g.status == "fail" for g in report.gates)
    assert report.verdict == "reject"


def test_qa_rejects_plastic_skin(original, target):
    cfg = PipelineConfig()
    _, masks = _retouch(original, target, cfg)
    plastic = cv2.GaussianBlur(original, (0, 0), sigmaX=6)  # wipe texture
    report = run_qa(original, plastic, masks, cfg)
    assert any(g.name == "texture" and g.status == "fail" for g in report.gates)
    assert report.verdict == "reject"
