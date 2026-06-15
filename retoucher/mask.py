"""Region masks that confine edits to where they belong.

Edit kinds:
- ``tone``: colour/tone correction (under-eye, lids, neck). Confined to the
  eroded face-oval (so ears/hair/neck are excluded) minus protected features.
- ``heal``: local marks removed via inpainting. Confined to skin (which DOES
  include neck/chest, so a chest blemish is reachable) minus protected features.

When face geometry (MediaPipe) is available it drives confinement and feature
protection. Without it the pipeline degrades to a YCrCb-skin + edge-gated
fallback and flags ``geom_used=False`` so callers know quality is reduced.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import PipelineConfig
from .diff import TouchUpMap
from .faceparse import FaceGeometry
from .image_io import to_uint8


@dataclass
class RegionMasks:
    by_kind: dict[str, np.ndarray]
    skin: np.ndarray
    under_eye: np.ndarray   # lower lid / tear trough for the corrector (0 if no geom)
    protect: np.ndarray     # features that must not change (0 if no geom); for QA
    geom_used: bool

    def edited(self) -> np.ndarray:
        if not self.by_kind:
            return np.zeros(self.skin.shape[:2], np.float32)
        union = np.clip(np.maximum.reduce(list(self.by_kind.values())), 0.0, 1.0)
        return np.clip(np.maximum(union, self.under_eye), 0.0, 1.0)

    def untouched(self, erode_px: int = 6) -> np.ndarray:
        """Hard mask of pixels that must NOT change (for QA), dilated for safety."""
        edited = (self.edited() > 0.05).astype(np.uint8)
        if erode_px > 0:
            edited = cv2.dilate(edited, _kernel(erode_px))
        return (1 - edited).astype(np.float32)


def _kernel(px: float) -> np.ndarray:
    r = max(1, int(round(px)))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r * 2 + 1, r * 2 + 1))


def _feather(mask: np.ndarray, feather_px: float) -> np.ndarray:
    if feather_px <= 0:
        return mask.astype(np.float32)
    return np.clip(cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigmaX=feather_px), 0.0, 1.0)


def _dilate(mask: np.ndarray, px: float) -> np.ndarray:
    return cv2.dilate(mask.astype(np.float32), _kernel(px))


def _erode(mask: np.ndarray, px: float) -> np.ndarray:
    return cv2.erode(mask.astype(np.float32), _kernel(px))


def skin_mask(rgb: np.ndarray) -> np.ndarray:
    """Classic YCrCb skin heuristic (includes neck/chest). Float 0/1."""
    ycrcb = cv2.cvtColor(to_uint8(rgb), cv2.COLOR_RGB2YCrCb)
    cr, cb = ycrcb[..., 1], ycrcb[..., 2]
    m = (cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)
    return m.astype(np.float32)


def _edge_gate(rgb: np.ndarray, band_px: float) -> np.ndarray:
    """Weight ~1 on smooth skin, →0 within ``band_px`` of hard edges (hair/ear/jaw)."""
    if band_px <= 0:
        return np.ones(rgb.shape[:2], np.float32)
    edges = cv2.Canny(cv2.cvtColor(to_uint8(rgb), cv2.COLOR_RGB2GRAY), 50, 150)
    edges = cv2.dilate(edges, _kernel(band_px))
    return 1.0 - _feather((edges > 0).astype(np.float32), band_px)


def _near_face_zone(face_oval: np.ndarray, erode_px: float) -> np.ndarray:
    """Face plus a bounded neck / open-collar-chest band below it.

    Heals are confined here so a blemish near the collar is reachable while
    skin-coloured CLOTHING further down the torso is never edited.
    """
    rows = np.where(face_oval.max(axis=1) > 0.5)[0]
    face_h = float(rows.max() - rows.min()) if rows.size else float(face_oval.shape[0])
    return _dilate(face_oval, max(erode_px, 0.45 * face_h))


def _change_mask(tmap: TouchUpMap, thresh: float) -> np.ndarray:
    """Where the target proposed a tone change of at least ``thresh``."""
    score = np.maximum(np.max(np.abs(tmap.low_delta), axis=2), np.abs(tmap.luma_delta))
    return (score > thresh).astype(np.float32)


def _marks_mask(tmap: TouchUpMap, luma_thresh: float, red_thresh: float, max_blob_frac: float) -> np.ndarray:
    """Small blemishes: dark spots the target removed OR model-independent red spots."""
    cand = ((tmap.mark_score > luma_thresh) | (tmap.red_score > red_thresh)).astype(np.uint8)
    if cand.sum() == 0:
        return np.zeros_like(tmap.mark_score, np.float32)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
    total = cand.size
    # Vectorized: keep small components (skip big shadows/moles + background 0).
    # A loop here is a real hang on a large noisy image (tens of thousands of blobs).
    areas = stats[:, cv2.CC_STAT_AREA]
    keep_labels = np.where((np.arange(num) != 0) & (areas <= max_blob_frac * total))[0]
    keep = np.isin(labels, keep_labels).astype(np.uint8)
    return cv2.dilate(keep, _kernel(2)).astype(np.float32)


def build_masks(
    original_rgb: np.ndarray,
    tmap: TouchUpMap,
    cfg: PipelineConfig,
    *,
    geom: FaceGeometry | None = None,
    forced: np.ndarray | None = None,
    change_thresh: float = 0.02,
) -> RegionMasks:
    h, w = original_rgb.shape[:2]
    skin = skin_mask(original_rgb)
    tone = _change_mask(tmap, change_thresh)
    heal = _marks_mask(tmap, cfg.mark_luma_thresh, cfg.mark_red_thresh, cfg.mark_max_blob_frac)
    under_eye = np.zeros((h, w), np.float32)
    protect = np.zeros((h, w), np.float32)

    # Grow skin slightly so a small blemish (whose own pixels may fail the skin
    # test) is still inside the healable region when surrounded by skin.
    skin_grow = _dilate(skin, max(6.0, cfg.feather_px))

    if geom is not None:
        protect = _dilate(geom.protect, cfg.protect_dilate_px)
        face_in = _erode(geom.face_oval, cfg.skin_erode_px)
        tone = tone * face_in                    # inside the face (excludes ears/hair/neck)
        # Heal on skin near the face only (face/neck/upper-collar chest), so
        # skin-coloured clothing down the torso is never edited.
        heal = heal * skin_grow * _near_face_zone(geom.face_oval, cfg.skin_erode_px)
        under_eye = geom.under_eye * (1.0 - protect)
    else:
        # Degraded fallback: skin-confine + edge-gate to keep edits off ear/hair.
        if float(skin.mean()) > 0.05:
            keep = _edge_gate(original_rgb, max(1.0, cfg.feather_px * 0.5))
            tone = tone * np.maximum(_feather(skin, cfg.feather_px * 0.5), 0.15) * keep
            heal = heal * skin_grow

    if forced is not None:
        heal = np.clip(heal + (forced > 0).astype(np.float32) * skin_grow, 0.0, 1.0)

    tone_f = _feather(tone, cfg.feather_px)
    heal_f = _feather(heal, max(1.0, cfg.feather_px / 6.0))
    if geom is not None:
        # Re-apply protection AFTER feathering so a wide feather can't bleed an
        # edit back onto brows/eyes/lips/nostrils.
        tone_f = tone_f * (1.0 - _feather(protect, max(1.0, cfg.feather_px * 0.25)))
        heal_f = heal_f * (1.0 - geom.protect)

    by_kind = {"tone": tone_f, "heal": heal_f}
    return RegionMasks(by_kind=by_kind, skin=skin, under_eye=under_eye,
                       protect=protect, geom_used=geom is not None)
