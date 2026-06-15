"""Audit — the FOURTH contract. A native-resolution, per-region self-audit whose
coverage EQUALS the map: every op is checked, and a region we cannot check is
reported `skipped`-and-not-clean, never a silent pass. Only clean regions ship.

This module exists because verifying on interpolated zoom repeatedly hid artifacts
and made a human catch every box, blur, missed mark, and faded lash. The hard rule:

    THE AUDIT RUNS AT NEAREST-NEIGHBOR NATIVE RESOLUTION.

Detectors compare original and retouched at identical native shape and never call an
interpolating resize. `_assert_native` enforces equal shape at runtime; a source-scan
test (test_audit.py) enforces that no interpolating resize appears in this module.

Detectors (each a gate dict {name,status,value,threshold,detail,required}):
  seam      box/seam at the mask boundary (gradient of the edit *delta* in the edge band)
  texture   blur/plastic (HF below local skin baseline) or stipple (HF above it)
  color     rouge / cast: region-mean vs clean-skin-ref-mean LAB drift
  residual  missed mark: a pigment/dark candidate still present after the edit
  lashes    protected-feature edge energy retained (feather didn't bleed onto lashes)
Plus a map-level identity gate that is REQUIRED — ArcFace when available, else a
defined SSIM fallback; never skipped-as-clean.
"""
from __future__ import annotations

import cv2
import numpy as np
from skimage.color import deltaE_ciede2000, rgb2lab
from skimage.metrics import structural_similarity

from .config import AuditThresholds
from .detect import detect_blemishes
from .image_io import to_uint8
from .mask import _dilate, _erode
from .schema import RegionVerdict

# NN-native invariant: anything in this module that must change pixel grid uses
# nearest-neighbor ONLY. We never up/down-sample for a comparison; we compare native.
_NN = cv2.INTER_NEAREST


def _assert_native(a: np.ndarray, b: np.ndarray) -> None:
    if a.shape[:2] != b.shape[:2]:
        raise ValueError(
            f"audit requires native, same-shape arrays (no interpolated resize); "
            f"got {a.shape[:2]} vs {b.shape[:2]}"
        )


def _gray(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(to_uint8(rgb), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0


def _hard(mask: np.ndarray) -> np.ndarray:
    return mask > 0.5


def _hf_energy(gray: np.ndarray, mask: np.ndarray, min_px: int) -> float | None:
    m = _hard(mask)
    if int(m.sum()) < min_px:
        return None
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    return float(np.mean(lap[m] ** 2))


def _mean_lab(rgb: np.ndarray, mask: np.ndarray, min_px: int) -> np.ndarray | None:
    m = _hard(mask)
    if int(m.sum()) < min_px:
        return None
    lab = rgb2lab(np.clip(to_uint8(rgb).astype(np.float32) / 255.0, 0, 1))
    return lab[m].mean(axis=0)


def _gate(name, status, value, threshold, detail, required=True) -> dict:
    return {"name": name, "status": status,
            "value": (round(float(value), 4) if value is not None else None),
            "threshold": threshold, "detail": detail, "required": required}


def _baseline_ring(mask: np.ndarray, *, skin: np.ndarray | None, band_px: float) -> np.ndarray:
    """A ring of adjacent (unedited) skin around the region — the local texture/tone the
    region should match. Per-region, so a hand's baseline differs from the face's."""
    ring = _dilate(mask, 3 * band_px) * (1.0 - _dilate(mask, band_px))
    if skin is not None:
        ring = ring * skin
    return ring


# ---- detectors (native resolution) ------------------------------------------------

def _seam_gate(original, retouched, mask, band_px, cfg) -> dict:
    delta = np.abs(_gray(retouched) - _gray(original))     # isolates the edit footprint
    band = _hard(_dilate(mask, band_px) * (1.0 - _erode(mask, band_px)))
    if int(band.sum()) < cfg.min_region_px:
        return _gate("seam", "skipped", None, cfg.seam_max, "edge band too small")
    gx = cv2.Sobel(delta, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(delta, cv2.CV_32F, 0, 1, ksize=3)
    gmag = np.sqrt(gx * gx + gy * gy)
    val = float(np.percentile(gmag[band], 95))
    ok = val <= cfg.seam_max
    return _gate("seam", "pass" if ok else "fail", val, cfg.seam_max,
                 "organic blend" if ok else "hard edge / box at the mask boundary")


def _texture_gate(retouched, mask, *, skin, band_px, cfg) -> dict:
    g = _gray(retouched)
    core = _erode(mask, band_px)
    region_e = _hf_energy(g, core, cfg.min_region_px)
    base_e = _hf_energy(g, _baseline_ring(mask, skin=skin, band_px=band_px), cfg.min_region_px)
    if region_e is None or base_e is None or base_e <= 1e-9:
        return _gate("texture", "skipped", None, cfg.texture_lo, "too little skin to baseline")
    ratio = region_e / base_e
    if ratio < cfg.texture_lo:
        return _gate("texture", "fail", ratio, cfg.texture_lo, "blur / plastic: texture below local skin")
    if ratio > cfg.texture_hi:
        return _gate("texture", "fail", ratio, cfg.texture_hi, "stipple: texture above local skin")
    return _gate("texture", "pass", ratio, cfg.texture_lo, "texture matches local skin")


def _color_gate(retouched, mask, skin_ref, cfg) -> dict:
    if skin_ref is None:
        return _gate("color", "skipped", None, cfg.color_max_delta_e,
                     "no clean-skin reference", required=False)
    reg = _mean_lab(retouched, _erode(mask, 2.0), cfg.min_region_px)
    ref = _mean_lab(retouched, skin_ref, cfg.min_region_px)
    if reg is None or ref is None:
        return _gate("color", "skipped", None, cfg.color_max_delta_e, "too few pixels", required=False)
    de = float(deltaE_ciede2000(reg.reshape(1, 1, 3), ref.reshape(1, 1, 3))[0, 0])
    ok = de <= cfg.color_max_delta_e
    return _gate("color", "pass" if ok else "fail", de, cfg.color_max_delta_e,
                 "tone matches clean skin" if ok else "color cast / rouge vs clean skin")


def _residual_gate(retouched, mask, geom, cfg) -> dict:
    # Form tight candidate blobs at a low score floor, then judge by peak severity:
    # normal skin texture peaks well below residual_severity; a real missed mark spikes.
    cands = detect_blemishes(retouched, geom, region=mask, top=8, score_thresh=cfg.residual_form_thresh)
    worst = max((c.severity for c in cands), default=0.0)
    ok = worst < cfg.residual_severity
    return _gate("residual", "pass" if ok else "fail", worst, cfg.residual_severity,
                 "no residual mark" if ok else "pigment/dark mark still present after edit")


def _lash_gate(original, retouched, protect, cfg) -> dict:
    if protect is None or int(_hard(protect).sum()) < cfg.min_region_px:
        return _gate("lashes", "skipped", None, cfg.lash_min_retention,
                     "no protected features here", required=False)
    m = _hard(protect)
    e0 = _hf_energy(_gray(original), protect, cfg.min_region_px)
    e1 = _hf_energy(_gray(retouched), protect, cfg.min_region_px)
    if not e0 or e0 <= 1e-9:
        return _gate("lashes", "skipped", None, cfg.lash_min_retention, "no edge energy to compare",
                     required=False)
    retention = float(e1 / e0)
    ok = retention >= cfg.lash_min_retention
    return _gate("lashes", "pass" if ok else "fail", retention, cfg.lash_min_retention,
                 "lashes/brows preserved" if ok else "feather bled onto lashes/brows (faded)")


# ---- region + map audit ------------------------------------------------------------

def audit_region(
    original: np.ndarray, retouched: np.ndarray, mask: np.ndarray, *,
    op_id: str = "", skin_ref=None, protect=None, skin=None, geom=None,
    band_px: float = 6.0, kind: str = "skin", cfg: AuditThresholds | None = None,
) -> RegionVerdict:
    """Run every applicable detector at native resolution. `clean` requires that at
    least one detector ran, none failed, and no REQUIRED detector was skipped.

    `kind` makes the audit region-aware: the skin gates (texture/residual/color) assume a
    skin region and are NOT applied to an eyeball edit (kind="eyes"), where the meaningful
    checks are seam, lashes, and the map-level identity gate. Reported skipped, not passed."""
    cfg = cfg or AuditThresholds()
    _assert_native(original, retouched)
    if int(_hard(mask).sum()) < cfg.min_region_px:
        g = _gate("coverage", "skipped", None, None, "region mask empty/too small")
        return RegionVerdict(op_id=op_id, clean=False, gates=[g])
    if kind == "eyes":
        na = lambda n: _gate(n, "skipped", None, None, "not applicable to an eye region", required=False)
        gates = [
            _seam_gate(original, retouched, mask, band_px, cfg),
            na("texture"), na("color"), na("residual"),
            _lash_gate(original, retouched, protect, cfg),
        ]
    else:
        gates = [
            _seam_gate(original, retouched, mask, band_px, cfg),
            _texture_gate(retouched, mask, skin=skin, band_px=band_px, cfg=cfg),
            _color_gate(retouched, mask, skin_ref, cfg),
            _residual_gate(retouched, mask, geom, cfg),
            _lash_gate(original, retouched, protect, cfg),
        ]
    ran = [g for g in gates if g["status"] != "skipped"]
    failed = [g for g in gates if g["status"] == "fail"]
    req_skipped = [g for g in gates if g["status"] == "skipped" and g["required"]]
    clean = bool(ran) and not failed and not req_skipped
    return RegionVerdict(op_id=op_id, clean=clean, gates=gates)


def identity_gate(
    original: np.ndarray, retouched: np.ndarray, *, protect=None, cfg: AuditThresholds | None = None
) -> dict:
    """REQUIRED for delivery. ArcFace cosine when InsightFace is available; otherwise a
    DEFINED fallback (protected-feature SSIM, else whole-frame SSIM). Always yields a
    pass/fail verdict — identity is NEVER reported skipped-and-assumed-clean."""
    cfg = cfg or AuditThresholds()
    _assert_native(original, retouched)
    try:  # pragma: no cover - optional dep
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(allowed_modules=["detection", "recognition"])
        app.prepare(ctx_id=-1, det_size=(640, 640))
        fo = app.get(cv2.cvtColor(to_uint8(original), cv2.COLOR_RGB2BGR))
        fr = app.get(cv2.cvtColor(to_uint8(retouched), cv2.COLOR_RGB2BGR))
        if fo and fr:
            cos = float(np.dot(fo[0].normed_embedding, fr[0].normed_embedding))
            ok = cos >= cfg.identity_min_cosine
            return _gate("identity", "pass" if ok else "fail", cos, cfg.identity_min_cosine,
                         "ArcFace cosine")
    except Exception:
        pass
    # Defined fallback — never a silent skip.
    go, gr = _gray(original), _gray(retouched)
    if protect is not None and int(_hard(protect).sum()) >= cfg.min_region_px:
        _, smap = structural_similarity(go, gr, data_range=1.0, full=True)
        score = float(smap[_hard(protect)].mean())
        detail = "fallback: protected-feature SSIM (InsightFace absent)"
    else:
        score = float(structural_similarity(go, gr, data_range=1.0))
        detail = "fallback: whole-frame SSIM (InsightFace absent, no geometry)"
    ok = score >= cfg.identity_fallback_min_ssim
    return _gate("identity", "pass" if ok else "fail", score, cfg.identity_fallback_min_ssim, detail)


def audit_map(
    original: np.ndarray, retouched: np.ndarray, regions: list[dict], *,
    geom=None, skin=None, cfg: AuditThresholds | None = None,
) -> list[RegionVerdict]:
    """Coverage == map: one verdict per region entry (each {op_id, mask, skin_ref?,
    protect?, band_px?}). A region we can't check is reported, never silently passed."""
    cfg = cfg or AuditThresholds()
    out: list[RegionVerdict] = []
    for r in regions:
        out.append(audit_region(
            original, retouched, r["mask"], op_id=r.get("op_id", ""),
            skin_ref=r.get("skin_ref"), protect=r.get("protect"), skin=skin, geom=geom,
            band_px=r.get("band_px", 6.0), kind=r.get("kind", "skin"), cfg=cfg,
        ))
    return out


def score_verdict(verdict: RegionVerdict) -> float:
    """Higher = cleaner. For audit-driven sampling: pick the candidate with the best
    total across regions; ties broken toward fewer fails. Never used to ship a fail."""
    s = 0.0
    for g in verdict.gates:
        s += {"pass": 1.0, "skipped": 0.0, "fail": -2.0}.get(g["status"], 0.0)
    return s


def all_clean(verdicts: list[RegionVerdict], identity: dict | None = None) -> bool:
    """Audit-gated delivery: ship only if every region is clean AND identity passes."""
    if identity is not None and identity["status"] != "pass":
        return False
    return all(v.clean for v in verdicts)
