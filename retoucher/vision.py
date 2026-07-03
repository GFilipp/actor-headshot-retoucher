"""Vision assessor seam — the "look at the whole picture" step.

A real assessor is a VLM that returns a structured defect inventory for the WHOLE
photo (face + hands/neck/chest/hair). `MockAssessor` derives an equivalent structure
from local CV (faceparse + detect) so the pipeline and tests run fully offline. Same
Protocol pattern as `generate.Generator`, so it's swappable and mockable.

An assessment dict is: {shot_type, lighting, face_count, defects:[{region, defect,
severity, bbox}]}.
"""
from __future__ import annotations

import json
from typing import Protocol

import numpy as np

from . import faceparse
from .detect import detect_blemishes
from .image_io import to_uint8


class VisionAssessor(Protocol):
    def assess(self, image_rgb: np.ndarray) -> dict:
        """Return a structured whole-photo defect inventory (see module docstring)."""
        ...


def _bbox_of(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0.5)
    if xs.size == 0:
        return (0, 0, 0, 0)
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


class MockAssessor:
    """Deterministic, offline assessor. Synthesizes a plausible defect inventory from
    CV so the whole system runs without a vision API (used by --dry-run and tests).

    Pass `geom` to reuse an already-detected (or synthetic) geometry instead of
    running MediaPipe — keeps the mock's inventory consistent with the caller's."""

    def __init__(self, geom: "faceparse.FaceGeometry | None" = None):
        self._geom = geom

    def assess(self, image_rgb: np.ndarray) -> dict:
        rgb = image_rgb.astype(np.float32)
        geom = self._geom if self._geom is not None else faceparse.detect(rgb)
        defects: list[dict] = []
        if geom is not None:
            defects.append({"region": "eye_area", "defect": "under_eye",
                            "severity": 0.6, "bbox": _bbox_of(geom.under_eye)})
            defects.append({"region": "eye_area", "defect": "discoloration",
                            "severity": 0.5, "bbox": _bbox_of(geom.under_eye)})
            defects.append({"region": "eye_area", "defect": "eye_white_cast",
                            "severity": 0.4, "bbox": _bbox_of(geom.eyes)})
            for c in detect_blemishes(rgb, geom, top=6):
                r = c.radius
                defects.append({"region": "face", "defect": "blemish", "severity": c.severity,
                                "bbox": (c.cx - r, c.cy - r, c.cx + r, c.cy + r)})
        return {"shot_type": "headshot" if geom is not None else "unknown",
                "lighting": "unknown", "face_count": 1 if geom is not None else 0,
                "defects": defects}


_VLM_PROMPT = (
    "You are a best-in-class film-industry retoucher inspecting a photo of an actor. "
    "Return STRICT JSON only, no prose: "
    '{"shot_type":"headshot|three_quarter|bodyshot","lighting":"soft|hard|mixed",'
    '"face_count":N,"defects":[{"region":"face|eye_area|neck|chest|hands|hair",'
    '"defect":"under_eye|crepe|pigmentation|discoloration|blemish|skin_unevenness|'
    'eye_white_cast|flyaway","severity":0.0-1.0,"bbox":[x0,y0,x1,y1]}]}. '
    "bbox values MUST be fractions of image width/height in 0.0-1.0 (x0,y0 = top-left, "
    "x1,y1 = bottom-right), NOT pixels — pixel coords at a guessed resolution put masks "
    "in the wrong place. "
    "Inventory EVERY region incl. hands/neck/chest/stray hairs. Identity must be "
    "preserved; flag only temporary/cosmetic defects, never bone structure or features."
)


class GeminiVisionAssessor:
    """Real VLM assessor — one Gemini vision call returning the structured inventory.
    Best-effort JSON parse; callers corroborate with CV (see analyze)."""

    def __init__(self, model: str = "gemini-2.5-flash", key_path: str = "~/Desktop/gemini.txt"):
        self.model = model
        self.key_path = key_path

    def assess(self, image_rgb: np.ndarray) -> dict:
        import os

        from google import genai
        from PIL import Image
        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            key = open(os.path.expanduser(self.key_path)).read().strip()
        client = genai.Client(api_key=key)
        resp = client.models.generate_content(
            model=self.model, contents=[_VLM_PROMPT, Image.fromarray(to_uint8(image_rgb))]
        )
        text = (getattr(resp, "text", "") or "").strip().strip("`")
        if text.startswith("json"):
            text = text[4:]
        try:
            return json.loads(text)
        except Exception:
            return {"shot_type": "unknown", "lighting": "unknown", "face_count": -1, "defects": []}
