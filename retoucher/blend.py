"""Apply the touch-up map onto the original, preserving form and texture.

Tone fixes (chroma-only): the masked low-frequency delta is applied to the
original's a*/b* (chroma) in LAB while the original L (luminance) is kept. Form
and contour live in luminance, so the nose/cheek shading survives instead of
being flattened. A guided filter blends the chroma correction edge-aware, so it
"smooths into" the skin with no visible patch.

Mark fixes: flagged spots are healed with ``cv2.inpaint`` from surrounding
original texture (never the generator).

Under-eye: a deterministic corrector lifts tear-trough shadow toward the
surrounding cheek lightness, independent of what the model proposed.
"""
from __future__ import annotations

import cv2
import numpy as np
from skimage.color import lab2rgb, rgb2lab

from .config import PipelineConfig
from .diff import TouchUpMap, frequency_separate
from .image_io import clip01, to_float, to_uint8
from .mask import RegionMasks


def _guided(src: np.ndarray, guide01: np.ndarray, radius: int, eps: float) -> np.ndarray:
    """Edge-aware smoothing of a correction field; falls back to a feather if
    opencv-contrib (ximgproc) is unavailable."""
    try:
        return cv2.ximgproc.guidedFilter(
            guide=guide01.astype(np.float32), src=src.astype(np.float32),
            radius=int(radius), eps=float(eps),
        )
    except Exception:  # pragma: no cover - depends on opencv build
        return cv2.GaussianBlur(src.astype(np.float32), (0, 0), sigmaX=max(1.0, radius / 2.0))


def apply_tone(
    rgb: np.ndarray, tmap: TouchUpMap, mask_tone: np.ndarray, strength: float, cfg: PipelineConfig
) -> np.ndarray:
    """Shift only chroma (a*/b*) by the masked, clamped low-freq delta; keep L."""
    tgt_low = clip01(tmap.orig_low + tmap.low_delta)
    lab = rgb2lab(clip01(rgb)).astype(np.float32)            # original LAB; L preserved
    dab = (rgb2lab(tgt_low) - rgb2lab(clip01(tmap.orig_low)))[..., 1:].astype(np.float32)
    dab = np.clip(dab, -cfg.max_chroma_delta, cfg.max_chroma_delta)

    applied = strength * mask_tone[..., None] * dab
    guide = lab[..., 0] / 100.0                              # luminance guide for edge-aware blend
    for c in (0, 1):
        applied[..., c] = _guided(applied[..., c], guide, cfg.guided_radius, cfg.guided_eps)
    lab[..., 1] += applied[..., 0]
    lab[..., 2] += applied[..., 1]
    return clip01(lab2rgb(lab)).astype(np.float32)


def heal_marks(image_rgb: np.ndarray, mask_heal: np.ndarray, radius: int = 3) -> np.ndarray:
    """Inpaint flagged marks from surrounding original texture."""
    hard = (mask_heal > 0.5).astype(np.uint8)
    if hard.sum() == 0:
        return image_rgb
    bgr = cv2.cvtColor(to_uint8(image_rgb), cv2.COLOR_RGB2BGR)
    healed = to_float(cv2.cvtColor(cv2.inpaint(bgr, hard, radius, cv2.INPAINT_TELEA), cv2.COLOR_BGR2RGB))
    m = mask_heal[..., None]
    return clip01(image_rgb * (1.0 - m) + healed * m).astype(np.float32)


def correct_under_eye(rgb: np.ndarray, region: np.ndarray, strength: float, sigma: float) -> np.ndarray:
    """Lift tear-trough shadow toward surrounding cheek lightness; texture kept.

    Deterministic: runs regardless of what the model proposed, so the inner
    corner near the nose is actually addressed.
    """
    if float(region.max()) <= 0:
        return rgb
    lab = rgb2lab(clip01(rgb)).astype(np.float32)
    L = lab[..., 0]
    surround = cv2.GaussianBlur(L, (0, 0), sigmaX=max(2.0, sigma * 2.0))
    lift = np.clip(surround - L, 0.0, 12.0)                 # darker-than-surroundings, capped (L units)
    lab[..., 0] = L + strength * region * lift              # smooth lift -> texture detail survives
    lab[..., 2] += strength * region * np.clip(lift, 0.0, 6.0) * 0.3  # ease a blue cast (b* up = warmer)
    return clip01(lab2rgb(lab)).astype(np.float32)


def smooth_under_eye_texture(
    rgb: np.ndarray, region: np.ndarray, strength: float, sigma: float,
    protect: np.ndarray | None = None,
) -> np.ndarray:
    """Soften crepey / scaly tear-trough texture by attenuating the high-frequency
    band inside the region, while keeping low-frequency tone untouched.

    The lift in ``correct_under_eye`` only fixes the *shadow*; the scaly *texture*
    survives because it lives in the high-frequency band. Here we keep
    ``(1 - strength)`` of that band's amplitude in the region, so the crepe softens
    but pores remain (it reads as smoother skin, not airbrushed plastic). The
    region weight is feathered so the change ramps in with no visible seam, and
    outside the region the image is unchanged (atten == 1).

    ``protect`` (brows/eyes/lips) is subtracted from the feathered weight so the
    feather can't bleed a texture change onto the lower lid / lash line; without
    this the feather pushes past the eye boundary and fails the protected-features
    QA gate.
    """
    if strength <= 0 or float(region.max()) <= 0:
        return rgb
    rgb = rgb.astype(np.float32)
    low, high = frequency_separate(rgb, sigma)
    weight = cv2.GaussianBlur(
        np.clip(region, 0.0, 1.0).astype(np.float32), (0, 0), sigmaX=max(2.0, sigma * 0.5)
    )
    if protect is not None:
        keep = 1.0 - cv2.GaussianBlur(
            np.clip(protect, 0.0, 1.0).astype(np.float32), (0, 0), sigmaX=max(1.0, sigma * 0.25)
        )
        weight = weight * np.clip(keep, 0.0, 1.0)
    atten = 1.0 - float(strength) * weight[..., None]
    return clip01(low + high * atten).astype(np.float32)


def even_skin_tone(
    rgb: np.ndarray, region: np.ndarray, strength: float, sigma: float
) -> np.ndarray:
    """Even skin-colour blotchiness by easing a*/b* toward their local average,
    leaving L untouched.

    Because luminance is preserved, facial form (shading) and pore texture are
    unchanged; this only calms uneven colour / redness, so it cannot go plastic
    or flatten the face. Confined to ``region`` (facial skin minus features).
    """
    if strength <= 0 or float(region.max()) <= 0:
        return rgb
    lab = rgb2lab(clip01(rgb)).astype(np.float32)
    w = float(strength) * np.clip(region, 0.0, 1.0)
    for c in (1, 2):  # a*, b* only; L (index 0) preserved -> form + texture kept
        smooth = cv2.GaussianBlur(lab[..., c], (0, 0), sigmaX=sigma)
        lab[..., c] = lab[..., c] * (1.0 - w) + smooth * w
    return clip01(lab2rgb(lab)).astype(np.float32)


def whiten_eye_whites(rgb: np.ndarray, eyes: np.ndarray, strength: float) -> np.ndarray:
    """De-red / de-yellow and gently brighten the sclera (eye whites).

    Sclera = the brighter pixels inside the eye mask (the iris, pupil, and lashes
    are darker, so a median-L threshold isolates the whites). Reduces a*/b* toward
    neutral and lifts L a touch. Conservative by design — fully white sclera looks
    dead/fake.
    """
    if strength <= 0 or float(eyes.max()) <= 0:
        return rgb
    lab = rgb2lab(clip01(rgb)).astype(np.float32)
    e = eyes > 0.5
    if int(e.sum()) < 20:
        return rgb
    L = lab[..., 0]
    sclera = (e & (L > float(np.median(L[e])))).astype(np.float32)
    w = cv2.GaussianBlur(sclera, (0, 0), sigmaX=2.0) * float(strength)
    lab[..., 1] *= (1.0 - 0.85 * w)                 # cut redness (a*)
    lab[..., 2] *= (1.0 - 0.7 * w)                  # cut yellowness (b*)
    lab[..., 0] = np.clip(L + 7.0 * w, 0.0, 100.0)  # brighten the white
    return clip01(lab2rgb(lab)).astype(np.float32)


def reduce_discoloration(
    rgb: np.ndarray, region: np.ndarray, strength: float, ref_mask: np.ndarray,
    max_l_lift: float = 9.0,
) -> np.ndarray:
    """Pull red/brown skin around the eye toward CLEAN reference skin (cheek/forehead).

    Only reduces EXCESS vs the reference — de-reddens (a*), de-yellows/browns (b*), and
    lifts dark/brown areas (L) — so clean skin is untouched and it can't invert. The L
    lift skips very dark pixels so eyelashes don't fade. ``region`` is the feathered
    area to treat; ``ref_mask`` marks clean skin to sample the target tone from.
    """
    if strength <= 0 or float(region.max()) <= 0:
        return rgb
    lab = rgb2lab(clip01(rgb)).astype(np.float32)
    ref = ref_mask > 0.5
    if int(ref.sum()) < 200:
        return rgb
    aT = float(np.median(lab[..., 1][ref]))
    bT = float(np.median(lab[..., 2][ref]))
    LT = float(np.percentile(lab[..., 0][ref], 55))
    w = float(strength) * np.clip(region, 0.0, 1.0)
    lab[..., 1] -= w * np.maximum(lab[..., 1] - aT, 0.0)              # de-redden
    lab[..., 2] -= w * np.maximum(lab[..., 2] - bT, 0.0)              # de-brown / de-yellow
    lift = np.minimum(w * np.maximum(LT - lab[..., 0], 0.0), max_l_lift)
    lab[..., 0] += lift * (lab[..., 0] > 25.0)                       # de-darken brown; spare lashes
    return clip01(lab2rgb(lab)).astype(np.float32)


def composite(
    original_rgb: np.ndarray, tmap: TouchUpMap, masks: RegionMasks, cfg: PipelineConfig
) -> np.ndarray:
    result = original_rgb.astype(np.float32)
    if "tone" in masks.by_kind:
        result = apply_tone(result, tmap, masks.by_kind["tone"], cfg.tone_strength, cfg)
    if float(masks.face.max()) > 0 and cfg.skin_even_strength > 0:
        region = cv2.GaussianBlur(
            (masks.face * (1.0 - masks.protect)).astype(np.float32),
            (0, 0), sigmaX=max(2.0, cfg.feather_px),
        )
        result = even_skin_tone(result, region, cfg.skin_even_strength, cfg.freq_sigma * 2.0)
    if "heal" in masks.by_kind:
        result = heal_marks(result, masks.by_kind["heal"])
    if float(masks.under_eye.max()) > 0:
        result = correct_under_eye(result, masks.under_eye, cfg.under_eye_strength, cfg.freq_sigma)
        result = smooth_under_eye_texture(
            result, masks.under_eye, cfg.under_eye_texture_strength, cfg.freq_sigma,
            protect=masks.protect,
        )
    return clip01(result).astype(np.float32)
