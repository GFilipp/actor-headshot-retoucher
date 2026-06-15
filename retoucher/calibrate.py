"""Calibrate — the THIRD contract. A pure policy function that decides, per op, the
generative-vs-deterministic split: composite mode, mask shape, feather, deterministic
follow-ups, and strengths — each with a recorded `rationale`. NOT a fixed
"paste + polish" recipe; the decision is a function of (defect, severity, face size,
resolution, identity sensitivity).

This is the rulebook this project learned the hard way (see references/retouch_learnings):
  - small / low-res face  -> NO raw paste (texture distorts at pixel zoom) -> lighter luma
  - pigmentation/discolor  -> chromatic, deterministic 'barely dents' it    -> generative-led
  - crepe / under-eye bags -> textural, the regenerated smooth region carries the fix
  - mild unevenness        -> deterministic-only (even-skin a*/b*, keep form)
  - stray / flyaway hair   -> generative-only (deterministic cannot remove it)
  - eye-white cast         -> deterministic sclera de-cast only (never generative near the eye)
  - isolated blemish       -> targeted deterministic heal (no field regenerate)
  - identity-sensitive     -> cap generative share, downgrade paste -> luma
"""
from __future__ import annotations

import math

from .config import CalibrationConfig
from .schema import CalibrationRecord, PhotoAssessment, RetouchMap, RetouchOp

# Deterministic follow-up op names — MUST match callables in blend.py.
DET_DISCOLOR = "reduce_discoloration"
DET_SMOOTH = "smooth_under_eye_texture"
DET_WHITEN = "whiten_eye_whites"
DET_EVEN = "even_skin_tone"
DET_HEAL = "heal_marks"


def _feather_px(bbox: tuple[int, int, int, int], cfg: CalibrationConfig) -> float:
    x0, y0, x1, y1 = bbox
    diag = math.hypot(max(1, x1 - x0), max(1, y1 - y0))
    return round(diag * cfg.feather_frac, 1)


def _mask_kind(region: str, defect: str) -> str:
    if region == "eye_area":
        return "eyes" if defect == "eye_white_cast" else "periorbital"
    if region in ("hands", "neck", "chest"):
        return "ecc_patch"          # no face landmarks -> ECC/ORB-registered crop
    return "disc"                   # face / hair: organic rounded disc


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _calibrate_op(
    op: RetouchOp, a: PhotoAssessment, big_face: bool, cfg: CalibrationConfig
) -> CalibrationRecord:
    defect, sev = op.defect, op.severity
    feather = _feather_px(op.bbox, cfg)
    mask_kind = _mask_kind(op.region, defect)
    det_strength = round(_clamp(
        cfg.det_strength_floor + sev * (cfg.det_strength_ceiling - cfg.det_strength_floor),
        cfg.det_strength_floor, cfg.det_strength_ceiling), 2)
    gen_weight, mode = 0.0, "none"
    det_ops: list[str] = []
    strength: dict[str, float] = {}
    why: list[str] = []
    small_note = "paste" if big_face else "lighter luma pass (small/low-res face)"

    if defect == "eye_white_cast":
        det_ops, strength = [DET_WHITEN], {DET_WHITEN: det_strength}
        why.append("eye-white cast: deterministic sclera de-cast only, nothing generative near the eyeball")

    elif defect == "flyaway":
        gen_weight, mode = 1.0, "paste"
        why.append("stray hair: deterministic cannot remove it; regenerate and paste the clean patch")

    elif defect in ("pigmentation", "discoloration"):
        gen_weight = cfg.gen_weight_pigment if big_face else cfg.gen_weight_small_face
        mode = "paste" if big_face else "luma"
        det_ops, strength = [DET_DISCOLOR], {DET_DISCOLOR: det_strength}
        why.append(f"pigmentation is chromatic: generative-led {small_note} + deterministic pull "
                   "toward the clean skin reference")

    elif defect in ("under_eye", "crepe"):
        gen_weight = cfg.gen_weight_strong if big_face else cfg.gen_weight_small_face
        mode = "paste" if big_face else "luma"
        det_ops = [DET_SMOOTH, DET_DISCOLOR]
        strength = {DET_SMOOTH: det_strength, DET_DISCOLOR: round(det_strength * 0.8, 2)}
        why.append(f"crepe/bags are textural: generative carries the fix ({small_note}); "
                   "deterministic smooth + de-discolor follow-up")

    elif defect == "skin_unevenness":
        if sev < cfg.mild_unevenness_sev:
            det_ops, strength = [DET_EVEN], {DET_EVEN: det_strength}
            why.append("mild unevenness: deterministic even-skin only (a*/b*); keep form and texture")
        else:
            gen_weight = cfg.gen_weight_pigment if big_face else cfg.gen_weight_small_face
            mode = "paste" if big_face else "luma"
            det_ops, strength = [DET_EVEN], {DET_EVEN: round(det_strength * 0.7, 2)}
            why.append(f"strong unevenness: generative-led {small_note} + light deterministic even-skin")

    elif defect == "blemish":
        det_ops, strength = [DET_HEAL], {DET_HEAL: det_strength}
        why.append("isolated blemish: targeted deterministic heal (small radius), no field regenerate")

    else:
        det_ops, strength = [DET_EVEN], {DET_EVEN: cfg.det_strength_floor}
        why.append(f"unrecognized defect '{defect}': conservative deterministic even-skin only")

    # Identity gate: cap the generative share and never raw-paste over a feature.
    if op.identity_sensitive and gen_weight > cfg.identity_gen_cap:
        gen_weight = cfg.identity_gen_cap
        if mode == "paste":
            mode = "luma"
        why.append(f"identity-sensitive: generative share capped at {cfg.identity_gen_cap}, paste -> luma")

    if gen_weight > 0:
        strength["composite"] = round(gen_weight, 2)

    return CalibrationRecord(
        op_id=op.op_id, gen_weight=round(gen_weight, 2), composite_mode=mode,
        det_ops=det_ops, mask_kind=mask_kind, grow=cfg.mask_grow, feather_px=feather,
        strength=strength, rationale="; ".join(why),
    )


def calibrate(
    assessment: PhotoAssessment, retouch_map: RetouchMap, cfg: CalibrationConfig | None = None
) -> list[CalibrationRecord]:
    """Pure: (assessment, map, cfg) -> one CalibrationRecord per op. No side effects,
    no mutation of inputs; same inputs always yield the same records."""
    cfg = cfg or CalibrationConfig()
    big_face = (assessment.face_px_frac >= cfg.large_face_frac
                and assessment.resolution_class in cfg.paste_resolution_classes)
    return [_calibrate_op(op, assessment, big_face, cfg) for op in retouch_map.ops]
