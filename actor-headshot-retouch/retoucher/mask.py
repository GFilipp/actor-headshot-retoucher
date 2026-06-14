"""Region masks that confine edits to where they belong.

Two edit kinds drive the blend:
- ``tone``: low-frequency colour/tone correction (under-eye, lids, eye whites,
  neck, hand discoloration).
- ``heal``: local marks removed using surrounding original texture.

Default path needs no face model: it edits only where the generator actually
proposed a change, refined by a skin estimate, and locates marks from the
original's dark-detail response. When MediaPipe is installed, per-region face
masks refine the result. Everything outside a mask is untouched by
construction, which kills cheek-bleaching and eye-white spill.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .config import PipelineConfig
from .diff import TouchUpMap


@dataclass
class RegionMasks:
    by_kind: dict[str, np.ndarray]
    skin: np.ndarray
    by_region: dict[str, np.ndarray] = field(default_factory=dict)

    def edited(self) -> np.ndarray:
        """Union of everything the pipeline will touch (feathered, 0..1)."""
        if not self.by_kind:
            h, w = self.skin.shape[:2]
            return np.zeros((h, w), np.float32)
        return np.clip(np.maximum.reduce(list(self.by_kind.values())), 0.0, 1.0)

    def untouched(self, erode_px: int = 6) -> np.ndarray:
        """Hard mask of pixels that must NOT change (for QA), eroded for safety."""
        edited = (self.edited() > 0.05).astype(np.uint8)
        if erode_px > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_px * 2 + 1,) * 2)
            edited = cv2.dilate(edited, k)
        return (1 - edited).astype(np.float32)


def _feather(mask: np.ndarray, feather_px: float) -> np.ndarray:
    if feather_px <= 0:
        return mask.astype(np.float32)
    blurred = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigmaX=feather_px)
    return np.clip(blurred, 0.0, 1.0)


def skin_mask(rgb: np.ndarray) -> np.ndarray:
    """Classic YCrCb skin heuristic. Returns float 0/1 (unfeathered)."""
    ycrcb = cv2.cvtColor((np.clip(rgb, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2YCrCb)
    cr, cb = ycrcb[..., 1], ycrcb[..., 2]
    m = (cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)
    return m.astype(np.float32)


def _change_mask(tmap: TouchUpMap, thresh: float) -> np.ndarray:
    """Where the target proposed a tone change of at least ``thresh``."""
    chroma = np.max(np.abs(tmap.low_delta), axis=2)
    luma = np.abs(tmap.luma_delta)
    score = np.maximum(chroma, luma)
    return (score > thresh).astype(np.float32)


def _marks_mask(tmap: TouchUpMap, thresh: float, max_blob_frac: float) -> np.ndarray:
    """Small dark spots in the original that the target removed."""
    dark = tmap.mark_score > thresh
    dark_u8 = dark.astype(np.uint8)
    if dark_u8.sum() == 0:
        return np.zeros_like(tmap.mark_score, np.float32)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(dark_u8, connectivity=8)
    total = dark_u8.size
    keep = np.zeros_like(dark_u8)
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if area <= max_blob_frac * total:  # marks are small; skip big shadows
            keep[labels == i] = 1
    # Grow slightly so healing covers the mark's soft edge.
    keep = cv2.dilate(keep, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    return keep.astype(np.float32)


def build_masks(
    original_rgb: np.ndarray,
    tmap: TouchUpMap,
    cfg: PipelineConfig,
    *,
    change_thresh: float = 0.02,
    mark_thresh: float = 0.06,
    mark_max_blob_frac: float = 0.01,
) -> RegionMasks:
    skin = skin_mask(original_rgb)
    skin_frac = float(skin.mean())

    tone = _change_mask(tmap, change_thresh)
    heal = _marks_mask(tmap, mark_thresh, mark_max_blob_frac)

    # Refine by skin only when skin detection is confident enough to be useful;
    # otherwise the change-mask alone is the confinement.
    if skin_frac > 0.05:
        skin_soft = _feather(skin, cfg.feather_px * 0.5)
        tone = tone * np.maximum(skin_soft, 0.15)  # allow a little off-skin tone work
        heal = heal * skin

    by_region: dict[str, np.ndarray] = {}
    landmarks = _try_mediapipe_landmarks(original_rgb)
    if landmarks is not None:
        by_region = _face_region_masks(original_rgb.shape[:2], landmarks)
        # Bias tone work toward face regions when we have them.
        face_union = np.clip(np.maximum.reduce(list(by_region.values())), 0, 1) if by_region else None
        if face_union is not None:
            tone = np.maximum(tone * 0.6, tone * face_union)

    by_kind = {
        "tone": _feather(tone, cfg.feather_px),
        # Marks are small; a tight feather (derived from feather_px so it scales
        # with resolution) keeps the core at full strength so the heal lands.
        "heal": _feather(heal, max(1.0, cfg.feather_px / 6.0)),
    }
    return RegionMasks(by_kind=by_kind, skin=skin, by_region=by_region)


# --- Optional MediaPipe face-region refinement (used only if installed) -------

# Landmark index groups for the FaceMesh 468-point topology.
_LEFT_UNDER_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133]
_RIGHT_UNDER_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263]


def _try_mediapipe_landmarks(rgb: np.ndarray):  # pragma: no cover - optional dep
    try:
        import mediapipe as mp
    except Exception:
        return None
    try:
        mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)
        res = mesh.process((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
        mesh.close()
        if not res.multi_face_landmarks:
            return None
        h, w = rgb.shape[:2]
        lms = res.multi_face_landmarks[0].landmark
        return np.array([[p.x * w, p.y * h] for p in lms], dtype=np.float32)
    except Exception:
        return None


def _poly_mask(shape_hw, points) -> np.ndarray:  # pragma: no cover - optional dep
    h, w = shape_hw
    m = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(m, np.int32(points), 1)
    return m.astype(np.float32)


def _face_region_masks(shape_hw, landmarks) -> dict:  # pragma: no cover - optional dep
    out = {}
    for name, idx in (("left_under_eye", _LEFT_UNDER_EYE), ("right_under_eye", _RIGHT_UNDER_EYE)):
        out[name] = _poly_mask(shape_hw, landmarks[idx])
    return out
