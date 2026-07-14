"""The surgical engine — the recipe that actually delivered the 20260509 shoot,
promoted from chat-history ad-hoc python into a runnable, tested path.

One targeted pass, human in the loop:

    donor (Gemini regenerates the photo)  ->  register to the original
    ->  color-match to clean face skin (kills donor rouge)
    ->  composite ONE organic region (paste / luma / transfer, protect after feather)
    ->  light deterministic polish (whites / de-discolor / lines)
    ->  audit the region at native resolution (a CHECK, not a gate)

Contrast with the v3 orchestrator: no whole-photo defect inventory, no per-op
auto-composites (that painted blobs on a real photo) — the operator picks the one
region that matters and judges the result. Defaults are the winning photo-3 recipe.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import faceparse
from .audit import audit_region, score_verdict
from .cleanup import polish_eyes
from .config import PipelineConfig
from .generate import edit_n
from .image_io import resize_to_megapixels
from .orchestrator import _clean_skin_ref
from .prompts import SURGICAL_UNDER_EYE
from .regions import MODES, build_region, color_match, composite_region, register_donor
from .schema import RegionVerdict

REGIONS = ("periorbital", "under_eye", "face")


@dataclass
class SurgicalResult:
    image: np.ndarray
    handleable: bool
    verdict: RegionVerdict | None = None
    report: dict = field(default_factory=dict)


def surgical_retouch(
    rgb: np.ndarray, *, generator, geom=None, region: str = "periorbital",
    mode: str = "paste", grow: float = 0.7, feather: float = 80.0, samples: int = 1,
    whites: float = 0.55, discolor: float = 0.6, lines: float = 0.2,
    prompt: str | None = None, pipe_cfg: PipelineConfig | None = None,
) -> SurgicalResult:
    """Run the surgical recipe on one photo. Returns the retouched image, the region's
    native-resolution audit verdict (informational — the operator decides), and a
    telemetry report. Refuses gracefully (returns the working image) when no frontal face."""
    if region not in REGIONS:
        raise ValueError(f"region must be one of {REGIONS}, got {region!r}")
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    pipe_cfg = pipe_cfg or PipelineConfig()
    rgb = rgb.astype(np.float32)
    if pipe_cfg.max_process_mp:
        rgb, _ = resize_to_megapixels(rgb, pipe_cfg.max_process_mp)
    # A caller-supplied geom computed on the pre-cap image would mismatch the resized frame
    # (masks at the wrong shape -> crash). Trust it only if it matches; else (re)detect.
    if geom is not None and geom.face_oval.shape[:2] != rgb.shape[:2]:
        geom = None
    if geom is None:
        geom = faceparse.detect(rgb)

    report: dict = {"engine": "surgical", "region": region, "mode": mode, "grow": grow,
                    "feather": feather, "samples": max(1, samples),
                    "polish": {"whites": whites, "discolor": discolor, "lines": lines}}
    if geom is None:
        report.update({"handleable": False, "reason": "no clear frontal face detected"})
        return SurgicalResult(rgb, handleable=False, report=report)

    region_mask = np.clip(build_region(geom, region, grow=grow), 0, 1)
    skin_ref = _clean_skin_ref(geom, region_mask, rgb.shape[:2], is_face=True)
    donors = edit_n(generator, rgb, prompt or SURGICAL_UNDER_EYE, n=max(1, samples))
    work_long = max(rgb.shape[:2])

    candidates: list[tuple[float, int, np.ndarray, RegionVerdict]] = []
    registrations: list[dict] = []
    for i, donor in enumerate(donors):
        if donor is None:
            registrations.append({"index": i, "method": "none", "score": 0.0})
            continue
        # A donor much smaller than the working frame (Gemini caps at ~1536px) would upscale
        # into a blurry/stippled paste — keep the original texture via transfer instead.
        eff_mode = "transfer" if (mode == "paste" and max(donor.shape[:2]) < 0.6 * work_long) else mode
        aligned, method, reg_score = register_donor(rgb, donor)
        registrations.append({"index": i, "method": method, "score": round(float(reg_score), 3),
                             "mode": eff_mode})
        aligned = color_match(aligned, rgb, skin_ref)         # match CLEAN skin, not the defect area
        out = composite_region(rgb, aligned, region_mask, mode=eff_mode, strength=1.0,
                               feather=feather, protect=geom.protect)
        out = polish_eyes(out, geom, whites=whites, discolor=discolor, lines=lines)
        verdict = audit_region(rgb, out, region_mask, op_id=f"surgical:{region}",
                               skin_ref=skin_ref, protect=geom.protect, geom=geom)
        candidates.append((score_verdict(verdict), i, out, verdict))
    report["registrations"] = registrations

    if not candidates:
        report.update({"handleable": False, "reason": "generator returned no usable donor"})
        return SurgicalResult(rgb, handleable=False, report=report)

    # Best score wins; ties break toward a clean verdict, then fewer failed gates.
    def _fails(v):
        return sum(1 for g in v.gates if g["status"] == "fail")
    candidates.sort(key=lambda t: (-t[0], not t[3].clean, _fails(t[3])))
    best_score, sel, image, verdict = candidates[0]
    report.update({
        "handleable": True, "selected": sel,
        "candidate_scores": [round(c[0], 3) for c in sorted(candidates, key=lambda t: t[1])],
        "verdict": verdict.to_dict(),
    })
    return SurgicalResult(image, handleable=True, verdict=verdict, report=report)
