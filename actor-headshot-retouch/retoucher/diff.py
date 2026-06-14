"""Frequency-separated touch-up map.

The map captures the generator's *intent* as a low-frequency (tone/colour)
delta. High-frequency texture is intentionally NOT transferred from the
generated target: the target is lower resolution and its texture is fabricated,
so importing it is what creates plastic skin. Texture always stays the
original's own; see ``blend.py`` for how marks are healed from surrounding
original pixels.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


def frequency_separate(rgb: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """Split into (low, high) so that ``low + high == rgb``.

    ``low`` holds colour/tone; ``high`` holds texture/detail as a signed
    residual centred on zero.
    """
    low = cv2.GaussianBlur(rgb, (0, 0), sigmaX=sigma, sigmaY=sigma)
    high = rgb.astype(np.float32) - low
    return low.astype(np.float32), high.astype(np.float32)


def _luma(rgb: np.ndarray) -> np.ndarray:
    # Rec. 709 luma.
    return rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


@dataclass
class TouchUpMap:
    low_delta: np.ndarray   # tgt_low - orig_low, float RGB (the tone/colour fix)
    orig_low: np.ndarray    # original low-frequency band
    orig_high: np.ndarray   # original high-frequency band (texture, preserved)
    luma_delta: np.ndarray  # luma of low_delta (HxW), where the target lightened tone
    mark_score: np.ndarray  # HxW: original has a dark mark the target removed


def compute_touch_up_map(
    original_rgb: np.ndarray,
    target_aligned_rgb: np.ndarray,
    sigma: float,
    neutralize_cast: bool = True,
) -> TouchUpMap:
    orig_low, orig_high = frequency_separate(original_rgb, sigma)
    tgt_low, _tgt_high = frequency_separate(target_aligned_rgb, sigma)

    low_delta = (tgt_low - orig_low).astype(np.float32)
    if neutralize_cast:
        # Remove a uniform per-channel shift (the model's global recolor) so the
        # mask and transfer respond only to LOCAL retouch intent.
        cast = np.median(low_delta.reshape(-1, 3), axis=0).reshape(1, 1, 3)
        low_delta = low_delta - cast
    luma_delta = _luma(low_delta)

    # Marks live in the high-frequency band, so a low-freq delta misses them.
    # Detect a local-dark response in BOTH images and flag where the original
    # has a dark spot the target removed. This is robust to the blur radius.
    orig_luma = _luma(original_rgb)
    tgt_luma = _luma(target_aligned_rgb)
    orig_dark = np.clip(cv2.GaussianBlur(orig_luma, (0, 0), sigmaX=sigma) - orig_luma, 0.0, 1.0)
    tgt_dark = np.clip(cv2.GaussianBlur(tgt_luma, (0, 0), sigmaX=sigma) - tgt_luma, 0.0, 1.0)
    mark_score = np.clip(orig_dark - tgt_dark, 0.0, 1.0)
    # Blur border handling fabricates a dark response along the frame edge.
    # Real blemishes never sit at the extreme edge, so suppress a thin band.
    b = max(2, int(round(sigma)))
    mark_score[:b, :] = mark_score[-b:, :] = mark_score[:, :b] = mark_score[:, -b:] = 0.0

    return TouchUpMap(
        low_delta=low_delta.astype(np.float32),
        orig_low=orig_low.astype(np.float32),
        orig_high=orig_high.astype(np.float32),
        luma_delta=luma_delta.astype(np.float32),
        mark_score=mark_score.astype(np.float32),
    )
