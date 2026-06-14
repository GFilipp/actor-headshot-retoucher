"""Apply the touch-up map onto the original, preserving its texture.

Tone fixes: add the masked low-frequency delta to the original's low band, then
re-add the original's OWN high-frequency band. Texture is therefore identical to
the source by construction; only colour/tone moves.

Mark fixes: heal flagged spots with ``cv2.inpaint`` so the fill comes from
surrounding original texture, never the generator.

Global cast: the median of the delta (a uniform recolor the model often adds) is
removed before transfer, so broad complexion never shifts.
"""
from __future__ import annotations

import cv2
import numpy as np

from .config import PipelineConfig
from .diff import TouchUpMap
from .image_io import clip01, to_uint8, to_float
from .mask import RegionMasks


def apply_tone(tmap: TouchUpMap, mask_tone: np.ndarray, strength: float) -> np.ndarray:
    """Add the masked low-frequency delta, then re-add the ORIGINAL texture.

    The delta is already global-cast-neutralized in diff.compute_touch_up_map,
    so only local tone/colour moves; high-frequency texture is the original's.
    """
    toned_low = tmap.orig_low + strength * mask_tone[..., None] * tmap.low_delta
    return clip01(toned_low + tmap.orig_high).astype(np.float32)


def heal_marks(image_rgb: np.ndarray, mask_heal: np.ndarray, radius: int = 3) -> np.ndarray:
    """Inpaint flagged marks from surrounding original texture."""
    hard = (mask_heal > 0.5).astype(np.uint8)
    if hard.sum() == 0:
        return image_rgb
    bgr = cv2.cvtColor(to_uint8(image_rgb), cv2.COLOR_RGB2BGR)
    healed = cv2.inpaint(bgr, hard, radius, cv2.INPAINT_TELEA)
    healed_rgb = to_float(cv2.cvtColor(healed, cv2.COLOR_BGR2RGB))
    # Soft-composite so the heal edge is feathered (mask is already feathered).
    m = mask_heal[..., None]
    return clip01(image_rgb * (1.0 - m) + healed_rgb * m).astype(np.float32)


def composite(
    original_rgb: np.ndarray, tmap: TouchUpMap, masks: RegionMasks, cfg: PipelineConfig
) -> np.ndarray:
    result = original_rgb.astype(np.float32)
    if "tone" in masks.by_kind:
        result = apply_tone(tmap, masks.by_kind["tone"], cfg.strength)
    if "heal" in masks.by_kind:
        result = heal_marks(result, masks.by_kind["heal"])
    return clip01(result).astype(np.float32)
