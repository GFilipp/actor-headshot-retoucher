"""P5: the self-audit. Each artifact this project shipped by mistake is synthesized
and asserted caught — at native resolution. Plus the two NN invariants (runtime
same-shape guard + a source scan banning interpolating resize) and required identity."""
from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np
import pytest

from retoucher import audit
from retoucher.audit import (
    all_clean, audit_map, audit_region, identity_gate, score_verdict,
)
from retoucher.mask import _feather

H = W = 256
_RNG = np.random.RandomState(0)


def _skin():
    base = np.full((H, W, 3), np.array([0.80, 0.66, 0.56], np.float32), np.float32)
    base += _RNG.normal(0, 0.012, (H, W, 3)).astype(np.float32)   # mild, even texture
    return np.clip(base, 0, 1)


def _disc(cy, cx, r):
    yy, xx = np.ogrid[:H, :W]
    return ((yy - cy) ** 2 + (xx - cx) ** 2 <= r * r).astype(np.float32)


BASE = _skin()
REGION = _disc(128, 128, 60)
REF = _disc(40, 40, 22)                 # clean skin reference, away from the region


def _feathered(base, amt):
    m = _feather(REGION, 12.0)[..., None]
    return np.clip(base * (1 - m) + np.clip(base + amt, 0, 1) * m, 0, 1)


def _gate(v, name):
    return next(g for g in v.gates if g["name"] == name)


def test_clean_region_passes_all_gates():
    clean = _feathered(BASE, 0.05)
    v = audit_region(BASE, clean, REGION, skin_ref=REF)
    assert v.clean
    for name in ("seam", "texture", "residual"):
        assert _gate(v, name)["status"] == "pass", name


def test_seam_box_is_caught():
    hard = BASE.copy()
    m = REGION > 0.5
    hard[m] = np.clip(hard[m] + 0.15, 0, 1)            # hard-edged disc, no feather
    v = audit_region(BASE, hard, REGION, skin_ref=REF)
    assert _gate(v, "seam")["status"] == "fail" and not v.clean


def test_blur_smear_is_caught():
    out = BASE.copy()
    blurred = cv2.GaussianBlur(BASE, (0, 0), 4)
    m = REGION > 0.5
    out[m] = blurred[m]
    v = audit_region(BASE, out, REGION, skin_ref=REF)
    g = _gate(v, "texture")
    assert g["status"] == "fail" and "blur" in g["detail"]


def test_stipple_is_caught():
    out = BASE.copy()
    noise = _RNG.normal(0, 0.08, (H, W, 3)).astype(np.float32)
    m = REGION > 0.5
    out[m] = np.clip(out[m] + noise[m], 0, 1)
    v = audit_region(BASE, out, REGION, skin_ref=REF)
    g = _gate(v, "texture")
    assert g["status"] == "fail" and "stipple" in g["detail"]


def test_color_cast_is_caught():
    m = _feather(REGION, 12.0)[..., None]
    shift = np.zeros((H, W, 3), np.float32)
    shift[..., 0] += 0.13; shift[..., 1] -= 0.07; shift[..., 2] -= 0.07   # rouge
    out = np.clip(BASE + shift * m, 0, 1)
    v = audit_region(BASE, out, REGION, skin_ref=REF)
    assert _gate(v, "color")["status"] == "fail"


def test_residual_mark_is_caught():
    out = _feathered(BASE, 0.05)
    blob = _disc(128, 150, 6) > 0.5
    out[blob] = np.array([0.45, 0.16, 0.16], np.float32)   # dark red mark left behind
    v = audit_region(BASE, out, REGION, skin_ref=REF)
    assert _gate(v, "residual")["status"] == "fail"


def test_faded_lashes_is_caught():
    protect = _disc(120, 120, 14)
    out = BASE.copy()
    pm = protect > 0.5
    out[pm] = cv2.GaussianBlur(BASE, (0, 0), 5)[pm]        # feature edge energy lost
    v = audit_region(BASE, out, REGION, skin_ref=REF, protect=protect)
    assert _gate(v, "lashes")["status"] == "fail"
    clean = audit_region(BASE, BASE.copy(), REGION, skin_ref=REF, protect=protect)
    assert _gate(clean, "lashes")["status"] == "pass"


def test_empty_mask_is_reported_not_silently_passed():
    v = audit_region(BASE, BASE.copy(), np.zeros((H, W), np.float32))
    assert not v.clean and _gate(v, "coverage")["status"] == "skipped"


def test_coverage_equals_map():
    regions = [{"op_id": "a", "mask": REGION}, {"op_id": "b", "mask": _disc(200, 200, 30)}]
    verdicts = audit_map(BASE, _feathered(BASE, 0.04), regions, skin=None)
    assert [v.op_id for v in verdicts] == ["a", "b"]      # one verdict per mapped op


def test_nn_native_runtime_guard_rejects_resize():
    smaller = cv2.resize(BASE, (128, 128), interpolation=cv2.INTER_NEAREST)
    with pytest.raises(ValueError, match="native"):
        audit_region(BASE, smaller, REGION)


def test_no_interpolating_resize_in_audit_source():
    src = Path(audit.__file__).read_text()
    banned = ["INTER_LINEAR", "INTER_CUBIC", "INTER_AREA", "INTER_LANCZOS",
              "transform.resize", "rescale("]
    for token in banned:
        assert token not in src, f"audit.py must not interpolate: found {token}"
    # cv2.resize is allowed ONLY with nearest-neighbor; none should appear otherwise.
    for call in re.findall(r"cv2\.resize\([^)]*\)", src):
        assert "INTER_NEAREST" in call, call


def test_identity_is_required_never_skipped():
    # InsightFace is absent in CI -> the defined fallback must still produce a verdict.
    same = identity_gate(BASE, BASE.copy())
    assert same["status"] == "pass" and same["status"] != "skipped"
    diff = identity_gate(BASE, np.clip(BASE + _RNG.normal(0, 0.4, BASE.shape), 0, 1).astype(np.float32))
    assert diff["status"] in ("pass", "fail") and diff["status"] != "skipped"


def test_audit_map_precompute_equivalence_and_conversion_count(monkeypatch):
    # The precompute is PURE speedup: audit_map verdicts must be identical to standalone
    # per-region calls, and rgb2lab must run once per image, not once per region.
    hard = BASE.copy()
    m = REGION > 0.5
    hard[m] = np.clip(hard[m] + 0.15, 0, 1)
    regions = [{"op_id": "a", "mask": REGION, "skin_ref": REF},
               {"op_id": "b", "mask": _disc(200, 200, 30), "skin_ref": REF}]
    vm = audit_map(BASE, hard, regions)
    vs = [audit_region(BASE, hard, r["mask"], op_id=r["op_id"], skin_ref=r["skin_ref"])
          for r in regions]
    assert [v.to_dict() for v in vm] == [v.to_dict() for v in vs]

    calls = {"n": 0}
    real = audit.rgb2lab

    def counting(x):
        calls["n"] += 1
        return real(x)

    monkeypatch.setattr(audit, "rgb2lab", counting)
    audit.audit_map(BASE, hard, regions)
    assert calls["n"] == 2                     # once per image, regardless of region count


def test_score_and_all_clean_gate_delivery():
    clean = audit_region(BASE, _feathered(BASE, 0.05), REGION, skin_ref=REF)
    hard = BASE.copy(); hard[REGION > 0.5] = np.clip(hard[REGION > 0.5] + 0.15, 0, 1)
    dirty = audit_region(BASE, hard, REGION, skin_ref=REF)
    assert score_verdict(clean) > score_verdict(dirty)
    assert all_clean([clean], identity={"status": "pass"})
    assert not all_clean([dirty], identity={"status": "pass"})
    assert not all_clean([clean], identity={"status": "fail"})   # identity gates delivery
