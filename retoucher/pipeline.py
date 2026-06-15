"""End-to-end orchestration: original -> target -> align -> diff -> mask ->
blend -> QA -> versioned output + report.

The generated target only ever proposes direction. Every pixel that ships comes
from the original, moved by a masked low-frequency delta or healed from the
original's own texture.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np

from . import faceparse, image_io, prompts
from .align import AlignResult, align_to_reference
from .blend import composite
from .config import PipelineConfig
from .diff import compute_touch_up_map
from .generate import Generator
from .mask import build_masks
from .qa import QAReport, contact_sheet, run_qa


def _build_forced_mask(shape_hw: tuple[int, int], marks, default_r: int) -> np.ndarray | None:
    """Rasterize user marks into a forced-fix mask. ``marks`` is a list of
    ('point', x, y, r|None) or ('box', x1, y1, x2, y2)."""
    if not marks:
        return None
    h, w = shape_hw
    m = np.zeros((h, w), np.float32)
    for spec in marks:
        if spec[0] == "point":
            _, x, y, r = spec
            cv2.circle(m, (int(x), int(y)), int(r or default_r), 1, -1)
        elif spec[0] == "box":
            _, x1, y1, x2, y2 = spec
            cv2.rectangle(m, (int(x1), int(y1)), (int(x2), int(y2)), 1, -1)
    return m if m.max() > 0 else None


@dataclass
class RetouchResult:
    source_path: Path
    output_path: Path | None
    contact_sheet_path: Path | None
    report_path: Path | None
    report: dict
    qa: QAReport
    align: AlignResult
    result_pixels: np.ndarray


def retouch_image(
    source_path: str | Path,
    out_dir: str | Path,
    generator: Generator,
    cfg: PipelineConfig | None = None,
    *,
    marks: list | None = None,
    write: bool = True,
) -> RetouchResult:
    cfg = cfg or PipelineConfig()
    source_path = Path(source_path)
    loaded = image_io.load(source_path)
    original = loaded.pixels

    # Scale pixel-based params to the image's actual size so behaviour is
    # resolution-independent (an 8px blur on a 4096px file would be useless).
    spatial = max(original.shape[:2]) / float(cfg.reference_dim)
    cfg = replace(
        cfg,
        freq_sigma=cfg.freq_sigma * spatial,
        feather_px=cfg.feather_px * spatial,
        protect_dilate_px=cfg.protect_dilate_px * spatial,
        skin_erode_px=cfg.skin_erode_px * spatial,
    )

    # Face geometry drives feature protection + confinement (quality path).
    geom = faceparse.detect(original)
    forced = _build_forced_mask(
        original.shape[:2], marks, default_r=int(0.012 * max(original.shape[:2]))
    )

    # 1) Propose a retouch target (downscaled to keep generation affordable).
    gen_input, scale = image_io.resize_to_megapixels(original, cfg.generator_max_mp)
    target = generator.edit(gen_input, prompts.prompt_for(cfg.mode))

    # 2) Register target onto the original's exact pixel grid.
    align = align_to_reference(
        original, target, iterations=cfg.ecc_iterations, epsilon=cfg.ecc_epsilon
    )

    # 3) Build the frequency-separated touch-up map.
    tmap = compute_touch_up_map(
        original, align.warped, cfg.freq_sigma, neutralize_cast=cfg.neutralize_global_cast
    )

    # 4) Mask to intended regions (feature-protected), then 5) transfer.
    masks = build_masks(original, tmap, cfg, geom=geom, forced=forced)
    result = composite(original, tmap, masks, cfg)

    # 6) Quality gates.
    qa = run_qa(original, result, masks, cfg)

    report = {
        "source": str(source_path),
        "mode": cfg.mode,
        "spatial_scale": round(spatial, 4),
        "generator_scale": round(scale, 4),
        "alignment": {"method": align.method, "score": round(align.score, 4), "success": align.success},
        "face_geometry": masks.geom_used,
        "forced_marks": 0 if not marks else len(marks),
        "edited_fraction": round(float((masks.edited() > 0.05).mean()), 4),
        "qa": qa.to_dict(),
        "config": cfg.to_dict(),
    }

    output_path = sheet_path = report_path = None
    if write:
        out_dir = Path(out_dir)
        output_path = image_io.save_versioned(
            result, out_dir, source_path.stem,
            icc=loaded.icc, exif=loaded.exif, quality=cfg.jpeg_quality,
        )
        sheet_path = contact_sheet(original, result, masks, output_path.with_name(output_path.stem + "_qa.png"))
        report_path = output_path.with_name(output_path.stem + "_report.json")
        report["output"] = str(output_path)
        report["contact_sheet"] = str(sheet_path)
        report_path.write_text(json.dumps(report, indent=2))

    return RetouchResult(
        source_path=source_path,
        output_path=output_path,
        contact_sheet_path=sheet_path,
        report_path=report_path,
        report=report,
        qa=qa,
        align=align,
        result_pixels=result,
    )


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"} | image_io.RAW_EXTENSIONS


def retouch_path(
    source: str | Path,
    out_dir: str | Path,
    generator: Generator,
    cfg: PipelineConfig | None = None,
    *,
    marks: list | None = None,
    write: bool = True,
) -> list[RetouchResult]:
    """Process a single image or every image in a directory (batch).

    ``marks`` (forced fixes) apply only to single-image runs, not batches.
    """
    source = Path(source)
    if source.is_dir():
        files = sorted(p for p in source.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        return [retouch_image(f, out_dir, generator, cfg, write=write) for f in files]
    return [retouch_image(source, out_dir, generator, cfg, marks=marks, write=write)]
