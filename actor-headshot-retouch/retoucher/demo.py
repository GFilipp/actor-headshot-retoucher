"""Synthetic headshot data for tests, examples, and a no-image trial run.

Real faces aren't needed to exercise the pipeline. `make_original` builds a
skin-toned image with pore texture, eyes/brows (alignment features), broad
under-eye discoloration (a tone defect), and small dark blemishes (mark
defects). `make_target` is the smooth "what good looks like" proposal with the
defects removed plus a uniform warm cast (so cast-neutralization is exercised).
"""
from __future__ import annotations

import cv2
import numpy as np

H = W = 256
SKIN = np.array([0.80, 0.63, 0.55], dtype=np.float32)

# Defect locations reused by tests and examples.
UNDER_EYE = ((slice(120, 140), slice(72, 112)), (slice(120, 140), slice(144, 184)))
BLEMISHES = ((150, 130), (175, 95), (165, 175))
CORNER = (slice(0, 20), slice(0, 20))

__all__ = [
    "H", "W", "SKIN", "UNDER_EYE", "BLEMISHES", "CORNER",
    "luma", "disk", "make_original", "make_target", "translate",
]


def luma(rgb: np.ndarray) -> np.ndarray:
    return rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def disk(cy: int, cx: int, r: int) -> np.ndarray:
    yy, xx = np.ogrid[:H, :W]
    return ((yy - cy) ** 2 + (xx - cx) ** 2) <= r * r


def _features(img: np.ndarray) -> None:
    for cx in (100, 156):                       # eyes (large -> not marks)
        img[disk(96, cx, 17)] = (0.12, 0.10, 0.10)
    img[70:78, 84:116] = (0.30, 0.22, 0.18)     # brows
    img[70:78, 140:172] = (0.30, 0.22, 0.18)


def make_original(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.ones((H, W, 3), np.float32) * SKIN
    img += rng.normal(0, 0.02, (H, W, 1)).astype(np.float32)  # pore texture
    _features(img)
    for reg in UNDER_EYE:                        # brown/purple discoloration
        img[reg[0], reg[1], 0] += 0.06
        img[reg[0], reg[1], 2] -= 0.04
    for cy, cx in BLEMISHES:                      # small dark marks
        img[disk(cy, cx, 4)] = SKIN * 0.55
    return np.clip(img, 0, 1).astype(np.float32)


def make_target() -> np.ndarray:
    tgt = np.ones((H, W, 3), np.float32) * SKIN   # smooth, defects gone
    _features(tgt)
    tgt[..., 0] = np.clip(tgt[..., 0] + 0.03, 0, 1)  # uniform warm cast
    return tgt.astype(np.float32)


def translate(img: np.ndarray, dx: float, dy: float) -> np.ndarray:
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
