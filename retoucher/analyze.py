"""Analyze — the FIRST contract. Look at the whole photo and decide how to treat it
*before* any pixel moves: shot type, face size, resolution, what's wrong in every
region, and whether we can handle it at all (else refuse/flag, never crash).

Hybrid: CV inventory (faceparse geometry + face size + blemish candidates) +
a VisionAssessor's whole-photo defect list (default `MockAssessor`, offline). The
VLM proposes; CV corroborates and supplies geometry/face-count guards.
"""
from __future__ import annotations

import numpy as np

from . import faceparse
from .faceparse import FaceGeometry
from .schema import IN_SCOPE_REGIONS, PhotoAssessment, RetouchMap, RetouchOp, Subject
from .vision import MockAssessor, VisionAssessor

HEADSHOT_WIDTH_FRAC = 0.30      # face bbox width / frame width
THREE_QUARTER_WIDTH_FRAC = 0.15
HIGH_RES_LONG_EDGE = 2000


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0.5)
    if xs.size == 0:
        return (0, 0, 0, 0)
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


def _sanitize_bbox(raw, w: int, h: int) -> tuple[int, int, int, int] | None:
    """Make a VLM-reported bbox safe to build a mask from. VLMs variously return pixel
    coords (sometimes for a different resolution than ours) or normalized 0-1 fractions;
    unvalidated coords put masks in the wrong place (the ghost-blob bug). Normalized
    values are scaled to the working frame, everything is clamped to the frame, and a
    box that is degenerate after clamping returns None (the caller drops + flags it)."""
    try:
        vals = [float(v) for v in raw]
    except (TypeError, ValueError):
        return None
    if len(vals) != 4:
        return None
    if all(0.0 <= v <= 1.5 for v in vals):        # normalized 0-1 (tolerate slight overshoot)
        vals = [vals[0] * w, vals[1] * h, vals[2] * w, vals[3] * h]
    x0, y0, x1, y1 = vals
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    x0 = int(np.clip(x0, 0, w - 1)); x1 = int(np.clip(x1, 0, w - 1))
    y0 = int(np.clip(y0, 0, h - 1)); y1 = int(np.clip(y1, 0, h - 1))
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def analyze(
    rgb: np.ndarray,
    assessor: VisionAssessor | None = None,
    *,
    geom: FaceGeometry | None = None,
) -> tuple[PhotoAssessment, RetouchMap, FaceGeometry | None]:
    """Return (assessment, retouch_map, face_geometry). `assessor` defaults to the
    offline MockAssessor so this runs without a vision API. Pass `geom` to reuse an
    already-detected geometry (orchestrator detects once; tests inject a synthetic one)."""
    rgb = rgb.astype(np.float32)
    h, w = rgb.shape[:2]
    if geom is None:
        geom = faceparse.detect(rgb)
    # Default mock shares the detected geometry so its inventory matches (and we don't
    # pay for a second detect). A real VLM assessor ignores geom and inspects pixels.
    assessor = assessor or MockAssessor(geom=geom)

    face_px_frac = float((geom.face_oval > 0.5).mean()) if geom is not None else 0.0
    fb = _bbox(geom.face_oval) if geom is not None else (0, 0, 0, 0)
    face_w_frac = (fb[2] - fb[0]) / float(w) if geom is not None else 0.0
    res_class = "native_high" if max(h, w) >= HIGH_RES_LONG_EDGE else "native_low"

    a = assessor.assess(rgb)
    vlm_faces = int(a.get("face_count", 1 if geom is not None else 0))
    lighting = str(a.get("lighting", "unknown"))
    # CV geometry is what the masks are built from, so it wins a face-count disagreement:
    # record the conflict instead of silently proceeding with face_count=0.
    cv_vlm_disagree = geom is not None and vlm_faces == 0
    if cv_vlm_disagree:
        vlm_faces = 1

    if face_w_frac >= HEADSHOT_WIDTH_FRAC:
        shot = "headshot"
    elif face_w_frac >= THREE_QUARTER_WIDTH_FRAC:
        shot = "three_quarter"
    elif geom is not None:
        shot = "bodyshot"
    else:
        shot = "unknown"

    # Handleable = exactly one registerable frontal face. Else refuse/flag (no crash).
    out_of_scope: list[str] = []
    if geom is None:
        handleable, reason = False, "no clear frontal face detected"
        out_of_scope.append("no-face/occluded/profile")
    elif vlm_faces > 1:
        handleable, reason = False, f"{vlm_faces} faces — multi-person not in scope"
        out_of_scope.append("multi-person")
    else:
        handleable = True
        reason = ("VLM reported 0 faces; CV geometry found one (CV wins)"
                  if cv_vlm_disagree else "")

    subjects: list[Subject] = []
    skin_refs: dict[str, tuple[int, int, int, int]] = {}
    if geom is not None:
        subjects.append(Subject("face", fb, face_px_frac, landmarks_available=True))
        skin_refs["face"] = fb

    ops: list[RetouchOp] = []
    for i, d in enumerate(a.get("defects", [])):
        region = str(d.get("region", "face"))
        defect = str(d.get("defect", "blemish"))
        sev = float(d.get("severity", 0.5))
        if region not in IN_SCOPE_REGIONS:
            if region not in out_of_scope:
                out_of_scope.append(region)
            continue
        bbox = _sanitize_bbox(d.get("bbox", (0, 0, 0, 0)), w, h)
        if bbox is None:
            # Never build a mask from a bogus box (the ghost-blob bug) — drop + flag.
            flag = f"{region}: invalid bbox"
            if flag not in out_of_scope:
                out_of_scope.append(flag)
            continue
        ops.append(RetouchOp(op_id=f"op{i}", region=region, defect=defect,
                             severity=sev, bbox=bbox, source="both"))
    ops.sort(key=lambda o: -o.severity)

    assessment = PhotoAssessment(
        shot_type=shot, face_px_frac=face_px_frac, resolution_class=res_class,
        face_count=vlm_faces, handleable=handleable, reason=reason, lighting=lighting,
        subjects=subjects, skin_refs=skin_refs, out_of_scope=out_of_scope,
    )
    return assessment, RetouchMap(ops=ops), geom
