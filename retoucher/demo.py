"""Synthetic headshot data for tests, examples, and a no-image trial run.

Real faces aren't needed to exercise most of the pipeline. `make_original`
builds a skin-toned image with pore texture, eyes/brows (alignment features),
broad under-eye discoloration (a tone defect), small dark blemishes, and a
REDDISH (not dark) blemish on the lower "chest" skin (the failure mode a
luma-only detector missed). `make_target` is the smooth "what good looks like"
proposal with defects removed plus a uniform warm cast. `fake_geometry` returns
a hand-built FaceGeometry over this layout so feature-protection / under-eye
logic can be tested without running MediaPipe.
"""
from __future__ import annotations

import cv2
import numpy as np

from .faceparse import FaceGeometry

H = W = 256
SKIN = np.array([0.80, 0.63, 0.55], dtype=np.float32)

# Defect locations reused by tests and examples.
UNDER_EYE = ((slice(120, 140), slice(72, 112)), (slice(120, 140), slice(144, 184)))
BLEMISHES = ((150, 130), (175, 95), (165, 175))
RED_BLEMISH = (230, 128)                 # reddish, not dark; on lower "chest" skin
CORNER = (slice(0, 20), slice(0, 20))

__all__ = [
    "H", "W", "SKIN", "UNDER_EYE", "BLEMISHES", "RED_BLEMISH", "CORNER",
    "luma", "disk", "make_original", "make_target", "translate", "fake_geometry",
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
    cy, cx = RED_BLEMISH                           # reddish blemish (luma ~ unchanged)
    img[disk(cy, cx, 5)] = (0.93, 0.55, 0.50)
    return np.clip(img, 0, 1).astype(np.float32)


def make_target() -> np.ndarray:
    tgt = np.ones((H, W, 3), np.float32) * SKIN   # smooth, defects gone
    _features(tgt)
    tgt[..., 0] = np.clip(tgt[..., 0] + 0.03, 0, 1)  # uniform warm cast
    return tgt.astype(np.float32)


def translate(img: np.ndarray, dx: float, dy: float) -> np.ndarray:
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _rect(r0: int, r1: int, c0: int, c1: int) -> np.ndarray:
    m = np.zeros((H, W), np.float32)
    m[r0:r1, c0:c1] = 1.0
    return m


def _ellipse(cy: int, cx: int, ay: int, ax: int) -> np.ndarray:
    m = np.zeros((H, W), np.uint8)
    cv2.ellipse(m, (cx, cy), (ax, ay), 0, 0, 360, 1, -1)
    return m.astype(np.float32)


def fake_geometry() -> FaceGeometry:
    """Hand-built geometry matching make_original's layout (no MediaPipe needed).

    The face oval deliberately excludes the lower 'chest' RED_BLEMISH, so tests
    can confirm heals still reach skin outside the face while tone edits do not.
    """
    eyes = np.clip(disk(96, 100, 19).astype(np.float32) + disk(96, 156, 19).astype(np.float32), 0, 1)
    brows = _rect(64, 84, 80, 176)
    lips = _rect(182, 198, 104, 152)
    protect = np.clip(eyes + brows + lips, 0, 1)
    face_oval = _ellipse(cy=104, cx=128, ay=108, ax=80)   # excludes corners + chest
    under_eye = np.clip(_rect(116, 150, 70, 186) - eyes, 0, 1)
    return FaceGeometry(face_oval=face_oval, protect=protect, under_eye=under_eye,
                        brows=brows, eyes=eyes, lips=lips)
