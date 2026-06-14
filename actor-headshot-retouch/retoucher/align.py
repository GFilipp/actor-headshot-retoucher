"""Register the generated target onto the original's exact pixel grid.

This is the keystone. A generated image is never pixel-aligned to the source;
without alignment, every downstream difference is dominated by sub-pixel shift
rather than real retouch intent, which is what produces "streaky" transfers.

Strategy, most precise first:
1. Optional face-landmark affine pre-align (MediaPipe, if installed).
2. ECC intensity alignment (``cv2.findTransformECC``), affine model.
3. ORB feature matching + RANSAC homography (``cv2.findHomography``) fallback.
4. Identity (resized only) with ``success=False`` if all else fails.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .image_io import resize_to, to_uint8


@dataclass
class AlignResult:
    warped: np.ndarray  # float RGB aligned to the reference grid
    method: str
    score: float
    success: bool


def _gray_f32(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(to_uint8(rgb), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0


def _try_ecc(ref_rgb: np.ndarray, mov_rgb: np.ndarray, iterations: int, epsilon: float):
    ref_g = _gray_f32(ref_rgb)
    mov_g = _gray_f32(mov_rgb)
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iterations, epsilon)
    try:
        cc, warp = cv2.findTransformECC(ref_g, mov_g, warp, cv2.MOTION_AFFINE, criteria)
    except cv2.error:
        return None, 0.0
    h, w = ref_rgb.shape[:2]
    warped = cv2.warpAffine(
        mov_rgb, warp, (w, h),
        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return warped.astype(np.float32), float(cc)


def _try_orb(ref_rgb: np.ndarray, mov_rgb: np.ndarray):
    ref_u8 = cv2.cvtColor(to_uint8(ref_rgb), cv2.COLOR_RGB2GRAY)
    mov_u8 = cv2.cvtColor(to_uint8(mov_rgb), cv2.COLOR_RGB2GRAY)
    orb = cv2.ORB_create(nfeatures=2000)
    k_ref, d_ref = orb.detectAndCompute(ref_u8, None)
    k_mov, d_mov = orb.detectAndCompute(mov_u8, None)
    if d_ref is None or d_mov is None or len(k_ref) < 4 or len(k_mov) < 4:
        return None, 0.0

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw = matcher.knnMatch(d_mov, d_ref, k=2)
    good = [m for pair in raw if len(pair) == 2 for m, n in [pair] if m.distance < 0.75 * n.distance]
    if len(good) < 8:
        return None, 0.0

    src = np.float32([k_mov[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([k_ref[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if H is None:
        return None, 0.0

    h, w = ref_rgb.shape[:2]
    warped = cv2.warpPerspective(
        mov_rgb, H, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )
    inlier_ratio = float(mask.sum()) / max(1, len(good))
    return warped.astype(np.float32), inlier_ratio


def align_to_reference(
    reference_rgb: np.ndarray,
    moving_rgb: np.ndarray,
    *,
    iterations: int = 200,
    epsilon: float = 1e-5,
    ecc_min_cc: float = 0.80,
) -> AlignResult:
    """Warp ``moving_rgb`` (the target) onto ``reference_rgb`` (the original)."""
    h, w = reference_rgb.shape[:2]
    if moving_rgb.shape[:2] != (h, w):
        moving_rgb = resize_to(moving_rgb, (w, h))

    warped, cc = _try_ecc(reference_rgb, moving_rgb, iterations, epsilon)
    if warped is not None and np.isfinite(cc) and cc >= ecc_min_cc:
        return AlignResult(warped, "ecc-affine", cc, True)

    orb_warped, score = _try_orb(reference_rgb, moving_rgb)
    if orb_warped is not None and score >= 0.30:
        return AlignResult(orb_warped, "orb-homography", score, True)

    # Best effort: keep the better ECC result if it ran at all, else identity.
    if warped is not None:
        return AlignResult(warped, "ecc-affine-weak", cc if np.isfinite(cc) else 0.0, False)
    return AlignResult(moving_rgb.astype(np.float32), "identity", 0.0, False)
