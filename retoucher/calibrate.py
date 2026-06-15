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
from dataclasses import replace

from .config import CalibrationConfig
from .schema import CalibrationRecord, PhotoAssessment, RegionVerdict, RetouchMap, RetouchOp

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


def escalate(
    record: CalibrationRecord, verdict: RegionVerdict, cfg: CalibrationConfig | None = None
) -> CalibrationRecord:
    """Pure re-calibration after a failed audit: map each failing gate to a bounded,
    targeted change (the orchestrator re-executes and re-audits, capped). Returns a NEW
    record; the input is untouched. The fixes are the ones this project learned:
      seam     -> wider feather + more organic spread
      blur     -> stop pasting low-res donor; keep original texture (transfer), less gen
      stipple  -> lighter luma + even-skin to calm injected noise
      color    -> stronger de-discolor toward the clean reference, cap generative share
      residual -> max de-discolor + targeted heal; the mark must come out
      lashes   -> pull the mask off the features (less grow, tighter feather)
    """
    cfg = cfg or CalibrationConfig()
    fails = {g["name"]: g for g in verdict.gates if g["status"] == "fail"}
    if not fails:
        return record
    mode, gen = record.composite_mode, record.gen_weight
    det = list(record.det_ops)
    strength = dict(record.strength)
    grow, feather = record.grow, record.feather_px
    notes = []

    if "seam" in fails:
        feather = round(feather * 1.6 + 2.0, 1)
        grow = round(grow + 0.15, 2)
        notes.append("seam: wider feather + organic spread")
    if "texture" in fails:
        if "blur" in fails["texture"]["detail"]:
            if gen > 0:                             # there's a paste smearing texture
                mode = "transfer"                   # keep the original skin texture
                gen = round(gen * 0.6, 2)
            if DET_SMOOTH in strength:              # over-smoothing is itself a blur cause
                strength[DET_SMOOTH] = round(strength[DET_SMOOTH] * 0.5, 2)
            notes.append("blur: keep original texture (transfer if pasting), back off smoothing")
        else:                                       # stipple
            if gen > 0:
                mode = "luma" if mode == "paste" else mode
                gen = round(gen * 0.7, 2)
            if DET_EVEN not in det:
                det.append(DET_EVEN)
            strength[DET_EVEN] = cfg.det_strength_ceiling
            notes.append("stipple: luma + even-skin to calm injected texture")
    if "color" in fails:
        if DET_DISCOLOR not in det:
            det.append(DET_DISCOLOR)
        strength[DET_DISCOLOR] = cfg.det_strength_ceiling
        gen = min(gen, 0.6)
        notes.append("color: stronger de-discolor toward clean reference, cap generative")
    if "residual" in fails:
        if DET_DISCOLOR not in det:
            det.append(DET_DISCOLOR)
        strength[DET_DISCOLOR] = cfg.det_strength_ceiling
        if DET_HEAL not in det:
            det.append(DET_HEAL)
            strength[DET_HEAL] = cfg.det_strength_ceiling
        notes.append("residual: max de-discolor + targeted heal until the mark clears")
    if "lashes" in fails:
        grow = round(max(0.7, grow - 0.2), 2)
        feather = round(max(1.0, feather * 0.8), 1)
        notes.append("lashes: pull mask off features, tighter feather")

    if gen > 0:
        strength["composite"] = round(gen, 2)
    elif "composite" in strength:
        del strength["composite"]
    rationale = record.rationale + " | escalated[" + "; ".join(notes) + "]"
    return replace(record, composite_mode=mode, gen_weight=round(gen, 2), det_ops=det,
                   strength=strength, grow=grow, feather_px=feather, rationale=rationale)
