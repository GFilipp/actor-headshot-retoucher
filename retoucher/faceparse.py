"""Face geometry from MediaPipe Face Mesh, turned into masks.

Provides the geometry the v2.1 fixes depend on:
- ``face_oval``  : filled face-skin region. Ears, hair, and neck fall OUTSIDE it,
                   so confining tone edits here fixes the ear-discoloration bug.
- ``protect``    : brows, eyes/lashes, lips, nostrils. Subtracted from every edit
                   mask so retouching never damages these features.
- ``under_eye``  : lower lid + inner-corner / tear-trough, for the dedicated
                   under-eye corrector.

Uses the MediaPipe Tasks FaceLandmarker (the legacy ``mp.solutions`` API was
removed in 0.10.x) with a model bundled at ``assets/face_landmarker.task`` so it
runs offline. ``detect`` returns None only when the parser/asset is unavailable
or no face is found, so callers degrade explicitly rather than silently.
"""
from __future__ import annotations

import atexit
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .image_io import to_uint8

_ASSET = Path(__file__).resolve().parent / "assets" / "face_landmarker.task"

# Standard MediaPipe Face Mesh (478-pt) index groups. Convex hulls of these fill
# each region; hulls slightly over-cover, which is the safe direction for a
# protection mask.
_FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365,
              379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93,
              234, 127, 162, 21, 54, 103, 67, 109]
_LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
_RIGHT_EYE = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]
_LEFT_BROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
_RIGHT_BROW = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
_LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0,
         37, 39, 40, 185]
_NOSTRILS = [98, 97, 2, 326, 327, 1, 94]
# Lower-lid rings; inner corners (133 / 362) sit next to the nose (tear trough).
_LEFT_UNDER = [33, 7, 163, 144, 145, 153, 154, 155, 133]
_RIGHT_UNDER = [362, 382, 381, 380, 374, 373, 390, 249, 263]

_LANDMARKER = None
_PROBE_OK = None  # cached result of the sandbox-safe MediaPipe subprocess probe


@dataclass
class FaceGeometry:
    face_oval: np.ndarray   # float 0/1, face skin (excludes ears/hair/neck)
    protect: np.ndarray     # float 0/1, features that must never be edited
    under_eye: np.ndarray   # float 0/1, lower lid + tear trough
    brows: np.ndarray       # float 0/1 (for QA feature-integrity checks)
    eyes: np.ndarray
    lips: np.ndarray


def _hull_mask(shape_hw: tuple[int, int], pts: np.ndarray) -> np.ndarray:
    h, w = shape_hw
    m = np.zeros((h, w), np.uint8)
    if len(pts) >= 3:
        cv2.fillConvexPoly(m, cv2.convexHull(np.int32(pts)), 1)
    return m.astype(np.float32)


def _get_landmarker():
    global _LANDMARKER
    if _LANDMARKER is None:
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        opts = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(_ASSET)), num_faces=1
        )
        _LANDMARKER = vision.FaceLandmarker.create_from_options(opts)
    return _LANDMARKER


@atexit.register
def _close_landmarker() -> None:  # avoid MediaPipe's noisy __del__ at shutdown
    global _LANDMARKER
    try:
        if _LANDMARKER is not None:
            _LANDMARKER.close()
    except Exception:
        pass
    _LANDMARKER = None


def _basic_available() -> bool:
    try:
        import mediapipe  # noqa: F401
    except Exception:
        return False
    return _ASSET.exists()


def _probe() -> bool:
    """Run FaceLandmarker once in an ISOLATED subprocess.

    MediaPipe can abort *natively* (not a catchable Python exception) on
    headless / sandboxed macOS — e.g. inside Codex — during its graphics setup.
    A native abort in a subprocess is contained and just yields a non-zero exit,
    so this lets us detect "MediaPipe will crash here" and fall back to the
    no-geometry path WITHOUT taking down the main process.
    """
    code = (
        "import numpy as np, mediapipe as mp\n"
        "from mediapipe.tasks import python\n"
        "from mediapipe.tasks.python import vision\n"
        "o = vision.FaceLandmarkerOptions("
        "base_options=python.BaseOptions(model_asset_path=r%r), num_faces=1)\n"
        "lm = vision.FaceLandmarker.create_from_options(o)\n"
        "lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.zeros((64,64,3), np.uint8)))\n"
        "print('OK')\n"
    ) % str(_ASSET)
    try:
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, timeout=60)
        return r.returncode == 0 and b"OK" in r.stdout
    except Exception:
        return False


def available() -> bool:
    """True only if MediaPipe can actually run here: asset present AND it survives
    a sandbox-safe subprocess probe. Override with ``RETOUCH_FACE_PARSER=off|on``
    (off = never use it; on = trust it and skip the probe)."""
    global _PROBE_OK
    env = os.environ.get("RETOUCH_FACE_PARSER", "").strip().lower()
    if env in ("off", "0", "no", "false", "disable"):
        return False
    if not _basic_available():
        return False
    if env in ("on", "1", "yes", "true", "force"):
        return True
    if _PROBE_OK is None:
        _PROBE_OK = _probe()
    return _PROBE_OK


def detect(rgb: np.ndarray) -> FaceGeometry | None:
    if not available():            # never touch MediaPipe in-process if it would crash
        return None
    try:
        import mediapipe as mp
        landmarker = _get_landmarker()
        u8 = np.ascontiguousarray(to_uint8(rgb))
        res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=u8))
    except Exception:
        return None
    if not res.face_landmarks:
        return None

    h, w = rgb.shape[:2]
    pts = np.array([[p.x * w, p.y * h] for p in res.face_landmarks[0]], dtype=np.float32)

    def region(idx: list[int]) -> np.ndarray:
        return _hull_mask((h, w), pts[idx])

    face_oval = region(_FACE_OVAL)
    brows = np.clip(region(_LEFT_BROW) + region(_RIGHT_BROW), 0, 1)
    eyes = np.clip(region(_LEFT_EYE) + region(_RIGHT_EYE), 0, 1)
    lips = region(_LIPS)
    nostrils = region(_NOSTRILS)
    protect = np.clip(brows + eyes + lips + nostrils, 0, 1)

    def under(idx: list[int]) -> np.ndarray:
        lid = pts[idx]
        drop = lid.copy()
        drop[:, 1] += 0.045 * h  # extend toward the cheek to cover the tear trough
        return _hull_mask((h, w), np.vstack([lid, drop]))

    under_eye = np.clip(under(_LEFT_UNDER) + under(_RIGHT_UNDER) - eyes, 0, 1)

    return FaceGeometry(face_oval, protect, under_eye, brows, eyes, lips)
