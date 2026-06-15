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
from .diff import TouchUpMap
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


def composite(
    original_rgb: np.ndarray, tmap: TouchUpMap, masks: RegionMasks, cfg: PipelineConfig
) -> np.ndarray:
    result = original_rgb.astype(np.float32)
    if "tone" in masks.by_kind:
        result = apply_tone(result, tmap, masks.by_kind["tone"], cfg.tone_strength, cfg)
    if "heal" in masks.by_kind:
        result = heal_marks(result, masks.by_kind["heal"])
    if float(masks.under_eye.max()) > 0:
        result = correct_under_eye(result, masks.under_eye, cfg.under_eye_strength, cfg.freq_sigma)
    return clip01(result).astype(np.float32)
