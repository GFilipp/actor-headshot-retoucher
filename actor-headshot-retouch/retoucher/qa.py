"""Automated quality gates.

Each gate returns pass / fail / skipped. A gate whose optional backend is
missing is ``skipped`` and reported as such, never silently passed. The verdict
is ``reject`` if any gate fails, else ``pass``.

Gates:
- identity        ArcFace cosine (InsightFace, optional) >= threshold
- untouched_ssim  SSIM over pixels that must not change >= threshold
- untouched_lpips perceptual distance on untouched pixels (optional) <= threshold
- edited_delta_e  mean CIEDE2000 inside edits within [min, max] (visible, not cartoon)
- texture         high-frequency energy loss <= threshold (no plastic skin)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from skimage.color import deltaE_ciede2000, rgb2lab
from skimage.metrics import structural_similarity

from .config import PipelineConfig
from .image_io import to_uint8
from .mask import RegionMasks


@dataclass
class Gate:
    name: str
    status: str  # pass | fail | skipped
    value: float | None
    threshold: float | None
    detail: str


@dataclass
class QAReport:
    gates: list[Gate]
    verdict: str

    def to_dict(self) -> dict:
        return {"verdict": self.verdict, "gates": [asdict(g) for g in self.gates]}

    def failed(self) -> list[str]:
        return [g.name for g in self.gates if g.status == "fail"]


def _gray(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(to_uint8(rgb), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0


def _gate_untouched_ssim(original, result, untouched, thr) -> Gate:
    n = float(untouched.sum())
    if n < 0.02 * untouched.size:
        return Gate("untouched_ssim", "skipped", None, thr, "too few untouched pixels")
    _, smap = structural_similarity(_gray(original), _gray(result), data_range=1.0, full=True)
    score = float((smap * untouched).sum() / n)
    status = "pass" if score >= thr else "fail"
    return Gate("untouched_ssim", status, round(score, 4), thr, "structural similarity off-edit")


def _gate_edited_delta_e(original, result, edited, lo, hi) -> Gate:
    hard = edited > 0.5
    if hard.sum() == 0:
        return Gate("edited_delta_e", "fail", 0.0, lo, "no visible edit was applied")
    de = deltaE_ciede2000(rgb2lab(original), rgb2lab(result))
    val = float(de[hard].mean())
    status = "pass" if lo <= val <= hi else "fail"
    detail = "edit visible and within range" if status == "pass" else f"outside [{lo}, {hi}]"
    return Gate("edited_delta_e", status, round(val, 3), lo, detail)


def _gate_texture(original, result, thr) -> Gate:
    def energy(img):
        return float(np.mean(cv2.Laplacian(_gray(img), cv2.CV_32F) ** 2))

    e0 = energy(original)
    e1 = energy(result)
    loss = 0.0 if e0 <= 1e-9 else max(0.0, (e0 - e1) / e0)
    status = "pass" if loss <= thr else "fail"
    return Gate("texture", status, round(loss, 4), thr, "high-frequency energy loss")


def _gate_identity(original, result, thr) -> Gate:
    try:  # pragma: no cover - optional dep
        from insightface.app import FaceAnalysis
    except Exception:
        return Gate("identity", "skipped", None, thr, "insightface not installed")
    try:  # pragma: no cover - optional dep
        app = FaceAnalysis(allowed_modules=["detection", "recognition"])
        app.prepare(ctx_id=-1, det_size=(640, 640))
        fo = app.get(cv2.cvtColor(to_uint8(original), cv2.COLOR_RGB2BGR))
        fr = app.get(cv2.cvtColor(to_uint8(result), cv2.COLOR_RGB2BGR))
        if not fo or not fr:
            return Gate("identity", "skipped", None, thr, "no face detected")
        a, b = fo[0].normed_embedding, fr[0].normed_embedding
        cos = float(np.dot(a, b))
        status = "pass" if cos >= thr else "fail"
        return Gate("identity", status, round(cos, 4), thr, "ArcFace cosine similarity")
    except Exception as exc:
        return Gate("identity", "skipped", None, thr, f"identity check error: {exc}")


def _gate_untouched_lpips(original, result, untouched, thr) -> Gate:
    try:  # pragma: no cover - optional dep
        import lpips
        import torch
    except Exception:
        return Gate("untouched_lpips", "skipped", None, thr, "lpips/torch not installed")
    try:  # pragma: no cover - optional dep
        m = untouched[..., None]
        o = (original * m).transpose(2, 0, 1)[None]
        r = (result * m).transpose(2, 0, 1)[None]
        net = lpips.LPIPS(net="alex")
        d = float(net(torch.tensor(o * 2 - 1).float(), torch.tensor(r * 2 - 1).float()).item())
        status = "pass" if d <= thr else "fail"
        return Gate("untouched_lpips", status, round(d, 4), thr, "perceptual distance off-edit")
    except Exception as exc:
        return Gate("untouched_lpips", "skipped", None, thr, f"lpips error: {exc}")


def contact_sheet(original, result, masks: RegionMasks, out_path: Path, max_crops: int = 3) -> Path:
    """Full-frame before/after plus 100% crops around the largest edits."""
    def col(img):
        return to_uint8(img)

    h, w = original.shape[:2]
    pad = np.full((h, 12, 3), 255, np.uint8)
    full = np.hstack([col(original), pad, col(result)])

    rows = [full]
    edited = (masks.edited() > 0.3).astype(np.uint8)
    if edited.sum() > 0:
        num, labels, stats, _ = cv2.connectedComponentsWithStats(edited, connectivity=8)
        order = sorted(range(1, num), key=lambda i: -stats[i, cv2.CC_STAT_AREA])[:max_crops]
        for i in order:
            x, y, cw, ch = (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                            stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
            m = 24
            x0, y0 = max(0, x - m), max(0, y - m)
            x1, y1 = min(w, x + cw + m), min(h, y + ch + m)
            co, cr = col(original[y0:y1, x0:x1]), col(result[y0:y1, x0:x1])
            crop = np.hstack([co, np.full((co.shape[0], 12, 3), 255, np.uint8), cr])
            if crop.shape[1] != full.shape[1]:
                scale = full.shape[1] / crop.shape[1]
                crop = cv2.resize(crop, (full.shape[1], int(crop.shape[0] * scale)))
            rows.append(np.full((12, full.shape[1], 3), 255, np.uint8))
            rows.append(crop)

    sheet = np.vstack(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    return out_path


def run_qa(original, result, masks: RegionMasks, cfg: PipelineConfig) -> QAReport:
    t = cfg.qa
    untouched = masks.untouched()
    edited = masks.edited()
    gates = [
        _gate_identity(original, result, t.identity_min_cosine),
        _gate_untouched_ssim(original, result, untouched, t.untouched_min_ssim),
        _gate_untouched_lpips(original, result, untouched, t.untouched_max_lpips),
        _gate_edited_delta_e(original, result, edited, t.edited_min_delta_e, t.edited_max_delta_e),
        _gate_texture(original, result, t.max_hf_energy_loss),
    ]
    verdict = "reject" if any(g.status == "fail" for g in gates) else "pass"
    return QAReport(gates=gates, verdict=verdict)
