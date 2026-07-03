"""Orchestrator — wires the four contracts into one spine:

    ingest -> analyze -> map -> calibrate -> execute -> audit -> deliver

Execute is per region: propose (generative donor) -> register -> composite (calibrated
mode) -> deterministic cleanup. Audit-driven sampling picks the cleanest of K candidates;
a failing region is escalated (bounded) and re-executed. Delivery is audit-gated: ship
only when every region is clean AND identity passes. Every decision lands in `report`
(the JSON telemetry), so the run is auditable and replayable after the fact.

Fully offline with MockAssessor + MockGenerator (the --dry-run path); no network, no GPU.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import cv2
import numpy as np

from . import faceparse
from .analyze import analyze
from .audit import all_clean, audit_map, identity_gate, score_verdict
from .blend import (
    even_skin_tone, heal_marks, reduce_discoloration, smooth_under_eye_texture,
    whiten_eye_whites,
)
from .calibrate import calibrate, escalate
from .config import AuditThresholds, CalibrationConfig, PipelineConfig
from .detect import detect_blemishes
from .generate import edit_n
from .image_io import resize_to_megapixels
from .mask import _dilate, _erode
from .prompts import build_edit_prompt
from .regions import build_region, composite_region, register_donor
from .schema import CalibrationRecord, PhotoAssessment, RegionVerdict, RetouchMap


@dataclass
class RetouchOutcome:
    """Result of a v3 run. Named distinctly from the legacy ``pipeline.RetouchResult``."""
    image: np.ndarray
    assessment: PhotoAssessment
    retouch_map: RetouchMap
    calibrations: list[CalibrationRecord]
    verdicts: list[RegionVerdict]
    identity: dict
    delivered: bool
    report: dict = field(default_factory=dict)


# ---- mask + reference helpers ------------------------------------------------------

def _disc_from_bbox(bbox, shape_hw, grow=1.0) -> np.ndarray:
    h, w = shape_hw
    x0, y0, x1, y1 = bbox
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    rx = max(4.0, (x1 - x0) / 2.0 * grow)
    ry = max(4.0, (y1 - y0) / 2.0 * grow)
    yy, xx = np.ogrid[:h, :w]
    return (((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0).astype(np.float32)


def _region_mask(record, op, geom, shape_hw) -> np.ndarray:
    named = {"periorbital": "periorbital", "under_eye": "under_eye", "face": "face"}
    if geom is not None and record.mask_kind in named:
        return np.clip(build_region(geom, named[record.mask_kind], grow=record.grow), 0, 1)
    if geom is not None and record.mask_kind == "eyes":
        return np.clip(geom.eyes, 0, 1)
    return _disc_from_bbox(op.bbox, shape_hw, grow=record.grow)   # disc / ecc_patch


def _clean_skin_ref(geom, region, shape_hw, *, is_face: bool = True) -> np.ndarray:
    """Per-region clean-skin reference for de-discoloration. A FACE region pulls toward clean
    face skin; a hand/neck/chest region pulls toward its OWN adjacent skin (the annulus), not
    the cheek (a hand de-discolored toward cheek color would shift wrong)."""
    if is_face and geom is not None:
        ref = geom.face_oval * (1.0 - np.clip(geom.protect + geom.under_eye + geom.eyes, 0, 1))
        ref = _erode(ref, 6.0)
        if float(ref.sum()) > 200:
            return ref
    return _dilate(region, 18.0) * (1.0 - _dilate(region, 6.0))


def _lowres_safe(rec):
    """When the donor is much lower-res than the working image, a paste/luma carries the
    donor's upscaling stipple into the region. Transfer (low-frequency tone only) keeps the
    original texture, so downgrade to it. Texture-fixing power is traded for no stipple."""
    if rec.composite_mode in ("paste", "luma"):
        return replace(rec, composite_mode="transfer",
                       rationale=rec.rationale + " | low-res donor: paste/luma -> transfer (keep texture)")
    return rec


# ---- per-region execution ----------------------------------------------------------

def _apply_det(rgb, name, region, strength, *, geom, skin_ref, protect, pipe_cfg):
    if strength <= 0:
        return rgb
    if name == "reduce_discoloration":
        ref = skin_ref if skin_ref is not None and float(skin_ref.sum()) > 50 else region
        return reduce_discoloration(rgb, region, strength, ref)
    if name == "smooth_under_eye_texture":
        return smooth_under_eye_texture(rgb, region, strength, pipe_cfg.freq_sigma, protect=protect)
    if name == "even_skin_tone":
        return even_skin_tone(rgb, region, strength, pipe_cfg.freq_sigma)
    if name == "whiten_eye_whites":
        eyes = geom.eyes if geom is not None else region
        return whiten_eye_whites(rgb, eyes, strength)
    if name == "heal_marks":
        cands = detect_blemishes(rgb, geom, region=region, top=8, score_thresh=0.1)
        mh = np.zeros(rgb.shape[:2], np.float32)
        for c in cands:
            cv2.circle(mh, (c.cx, c.cy), int(c.radius), 1.0, -1)
        mh *= np.clip(region, 0, 1)
        return heal_marks(rgb, mh) if float(mh.sum()) > 0 else rgb
    return rgb


def _execute_region(result, original, donor, record, op, *, geom, region, skin_ref, protect, pipe_cfg):
    out = result
    if record.gen_weight > 0 and donor is not None and record.composite_mode != "none":
        out = composite_region(out, donor, region, mode=record.composite_mode,
                               strength=record.gen_weight, feather=record.feather_px, protect=protect)
    for name in record.det_ops:
        out = _apply_det(out, name, region, record.strength.get(name, 0.4),
                         geom=geom, skin_ref=skin_ref, protect=protect, pipe_cfg=pipe_cfg)
    return out


def _build_result(original, donor, specs, geom, pipe_cfg):
    result = original.copy()
    for op, rec, region, skin_ref, protect in specs:
        result = _execute_region(result, original, donor, rec, op, geom=geom, region=region,
                                 skin_ref=skin_ref, protect=protect, pipe_cfg=pipe_cfg)
    return result


# ---- the spine ---------------------------------------------------------------------

def retouch(
    rgb: np.ndarray, *, generator=None, assessor=None, geom=None,
    calib_cfg: CalibrationConfig | None = None, audit_cfg: AuditThresholds | None = None,
    pipe_cfg: PipelineConfig | None = None, samples: int = 1, max_escalate: int = 1,
) -> RetouchOutcome:
    pipe_cfg = pipe_cfg or PipelineConfig()
    rgb = rgb.astype(np.float32)
    if geom is None:
        # Cap the working/audit resolution once, BEFORE any edit (mirrors the v2 pipeline).
        # The native-res audit invariant is about not verifying the RESULT on an interpolated
        # zoom; this single pre-edit downscale sets the working "native" — the audit still
        # compares result vs original at identical working shape. 8 MP is casting-grade.
        if pipe_cfg.max_process_mp:
            rgb, _ = resize_to_megapixels(rgb, pipe_cfg.max_process_mp)
        geom = faceparse.detect(rgb)

    assessment, rmap, geom = analyze(rgb, assessor, geom=geom)
    report: dict = {"assessment": assessment.to_dict(), "map": rmap.to_dict()}

    if not assessment.handleable:
        # Refuse/flag, never crash and never silently ship an unhandled photo.
        report["delivered"] = False
        identity = {"name": "identity", "status": "skipped", "value": None,
                    "threshold": None, "detail": "not handleable", "required": True}
        report["identity"] = identity
        return RetouchOutcome(rgb, assessment, rmap, [], [], identity, False, report)

    calibs = calibrate(assessment, rmap, calib_cfg)
    report["calibrations"] = [c.to_dict() for c in calibs]
    protect = geom.protect if geom is not None else None

    # Generative donors first: regenerate the whole photo once, K samples (audit-driven sampling).
    need_gen = any(r.gen_weight > 0 for r in calibs)
    raw_donors = [None]
    if need_gen and generator is not None:
        prompt = build_edit_prompt(rmap)
        report["prompt"] = prompt
        raw_donors = edit_n(generator, rgb, prompt, n=max(1, samples))
    # A donor much lower-res than the working image injects upscaling stipple wherever it is
    # pasted/luma'd. Keep the original texture instead: downgrade those modes to transfer.
    work_long = max(rgb.shape[:2])
    lowres_donor = any(d is not None and max(d.shape[:2]) < 0.6 * work_long for d in raw_donors)
    report["donor_lowres"] = lowres_donor
    if lowres_donor:
        calibs = [_lowres_safe(c) for c in calibs]
    donors = [register_donor(rgb, d)[0] if d is not None else None for d in raw_donors]
    report["samples"] = len(donors)

    # Build region masks / references once (they don't depend on the donor).
    specs = []
    region_audit = []
    for op, rec in zip(rmap.ops, calibs):
        region = _region_mask(rec, op, geom, rgb.shape[:2])
        skin_ref = _clean_skin_ref(geom, region, rgb.shape[:2], is_face=op.region in ("face", "eye_area"))
        kind = "eyes" if rec.mask_kind == "eyes" else "skin"
        specs.append((op, rec, region, skin_ref, protect))
        region_audit.append({"op_id": op.op_id, "mask": region, "skin_ref": skin_ref,
                             "protect": protect, "kind": kind,
                             "band_px": max(4.0, rec.feather_px * 0.5)})

    # Pick the cleanest candidate at native resolution.
    scored = []
    for di, donor in enumerate(donors):
        result = _build_result(rgb, donor, specs, geom, pipe_cfg)
        verdicts = audit_map(rgb, result, region_audit, geom=geom, cfg=audit_cfg)
        scored.append((sum(score_verdict(v) for v in verdicts), di, result, donor, verdicts))
    scored.sort(key=lambda t: -t[0])
    _, sel_di, result, donor, verdicts = scored[0]
    report["selected_sample"] = sel_di
    report["sample_scores"] = [round(s[0], 3) for s in scored]

    # Bounded escalation of failing regions, then re-execute + re-audit the whole result.
    cur = list(calibs)
    escalations: list[dict] = []
    for _ in range(max_escalate):
        failing = [i for i, v in enumerate(verdicts) if not v.clean]
        if not failing:
            break
        for i in failing:
            op, rec, _, skin_ref, _ = specs[i]
            new_rec = escalate(rec, verdicts[i], calib_cfg)
            cur[i] = new_rec
            region = _region_mask(new_rec, op, geom, rgb.shape[:2])
            specs[i] = (op, new_rec, region, skin_ref, protect)
            region_audit[i] = {**region_audit[i], "mask": region,
                               "band_px": max(4.0, new_rec.feather_px * 0.5)}
            escalations.append({"op_id": op.op_id, "rationale": new_rec.rationale})
        result = _build_result(rgb, donor, specs, geom, pipe_cfg)
        verdicts = audit_map(rgb, result, region_audit, geom=geom, cfg=audit_cfg)

    identity = identity_gate(rgb, result, protect=protect, cfg=audit_cfg)
    delivered = all_clean(verdicts, identity)
    report.update({
        "calibrations_final": [c.to_dict() for c in cur],
        "escalations": escalations,
        "verdicts": [v.to_dict() for v in verdicts],
        "identity": identity,
        "delivered": delivered,
    })
    return RetouchOutcome(result, assessment, rmap, cur, verdicts, identity, delivered, report)
