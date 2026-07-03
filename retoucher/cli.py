"""Command-line entry point.

Examples:
    # Offline smoke (no API key, no cost):
    python -m retoucher headshot.jpg --dry-run --out-dir out

    # Real run against OpenAI (reads OPENAI_API_KEY):
    python -m retoucher headshot.jpg --mode hybrid-map --out-dir out

    # Batch a folder:
    python -m retoucher ./shoot --out-dir ./shoot-retouched
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import traceback
from pathlib import Path

from .config import PipelineConfig
from .generate import get_generator
from .pipeline import retouch_path


def _parse_marks(mark_pts: list[str], mark_boxes: list[str]) -> list:
    marks: list = []
    for s in mark_pts:
        v = [float(x) for x in s.split(",")]   # raises ValueError on non-numeric
        if len(v) < 2:
            raise ValueError(f"--mark needs X,Y[,R]; got {s!r}")
        marks.append(("point", v[0], v[1], v[2] if len(v) > 2 else None))
    for s in mark_boxes:
        v = [float(x) for x in s.split(",")]
        if len(v) != 4:
            raise ValueError(f"--mark-box needs X1,Y1,X2,Y2; got {s!r}")
        marks.append(("box", v[0], v[1], v[2], v[3]))
    return marks


def _openai_ready() -> str | None:
    """Return an error string if the OpenAI backend can't run, else None."""
    try:
        import openai  # noqa: F401
    except ImportError:
        return ("OpenAI backend needs the 'openai' package: "
                "pip install 'actor-headshot-retoucher[openai]'  (or use --dry-run).")
    if not os.environ.get("OPENAI_API_KEY"):
        return "OPENAI_API_KEY is not set. Run: export OPENAI_API_KEY=sk-...  (or use --dry-run)."
    return None


def _load_readiness():
    path = Path(__file__).resolve().parents[1] / "scripts" / "check_readiness.py"
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location("check_readiness", path)
    mod = importlib.util.module_from_spec(spec)
    # Register before exec so dataclasses can resolve the module (Python 3.12+).
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _preflight(mode: str, source: Path, out_dir: Path, *, force: bool) -> bool:
    mod = _load_readiness()
    if mod is None:
        print("preflight: check_readiness.py not found; skipping", file=sys.stderr)
        return True
    ns = argparse.Namespace(
        mode=mode, source=str(source), output_dir=str(out_dir), image_gen_available="yes"
    )
    checks = mod.build_checks(ns)
    failures = mod.blocking_failures(checks)
    for c in checks:
        print(mod.format_check(c))
    if failures and not force:
        print("\nPreflight blocked. Fix the items above or pass --force.", file=sys.stderr)
        return False
    return True


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="retoucher", description="Deterministic headshot retouch.")
    p.add_argument("source", help="Image file or a directory of images.")
    p.add_argument("--out-dir", default="retouch-out", help="Output directory.")
    p.add_argument(
        "--mode", default="hybrid-map",
        choices=["hybrid-map", "light-retouch", "light-regen"], help="Retouch mode.",
    )
    p.add_argument("--backend", default="openai", choices=["openai", "mock"], help="Generator backend.")
    p.add_argument("--model", default=None,
                   help="OpenAI model id. Default: $OPENAI_IMAGE_MODEL or gpt-image-2 (current latest).")
    p.add_argument("--dry-run", action="store_true", help="Use the mock generator (no API, no cost).")
    p.add_argument("--strength", type=float, default=None, help="Tone transfer strength 0..1.")
    p.add_argument("--under-eye-texture", type=float, default=None,
                   help="Tear-trough texture smoothing 0..1 (softens crepey/scaly under-eye skin).")
    p.add_argument("--skin-even", type=float, default=None,
                   help="Skin colour evening 0..1 (calms blotchiness/redness; keeps form + texture).")
    p.add_argument("--mark", action="append", default=[], metavar="X,Y[,R]",
                   help="Force a fix at a point (pixels; repeatable).")
    p.add_argument("--mark-box", action="append", default=[], metavar="X1,Y1,X2,Y2",
                   help="Force a fix inside a box (pixels; repeatable).")
    p.add_argument("--max-mp", type=float, default=None, help="Max megapixels sent to the generator.")
    p.add_argument("--max-process-mp", type=float, default=None,
                   help="Cap working/output megapixels (default 8). Lower it if a big image is slow.")
    p.add_argument("--engine", default="v2", choices=["v2", "v3"],
                   help="v2=legacy deterministic pipeline; v3=north-star dynamic hybrid system.")
    p.add_argument("--samples", type=int, default=2,
                   help="v3: generative candidates to draw and audit at native res (ship the cleanest).")
    p.add_argument("--max-escalate", type=int, default=1,
                   help="v3: bounded audit-driven re-calibration rounds on failing regions.")
    p.add_argument("--no-write", action="store_true", help="Run without writing outputs.")
    p.add_argument("--skip-preflight", action="store_true", help="Skip the readiness check.")
    p.add_argument("--force", action="store_true",
                   help="v2: run even if preflight fails. v3: also write the result image even "
                        "when the audit flags it (for inspection), labelled as flagged.")
    p.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    return p


_IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


def _run_v3(source: Path, out_dir: Path, args) -> int:
    """North-star engine: analyze -> map -> calibrate -> execute -> audit -> deliver.
    Audit-gated: an image is written ONLY when every region is clean and identity passes;
    otherwise a telemetry report is written for inspection (never ship least-bad)."""
    from .image_io import load, save_versioned
    from .orchestrator import retouch
    from .vision import GeminiVisionAssessor

    paths = ([source] if source.is_file()
             else sorted(p for p in source.iterdir() if p.suffix.lower() in _IMG_EXTS))
    if not paths:
        print(f"No images found in {source}", file=sys.stderr)
        return 1

    backend = "mock" if args.dry_run else "gemini"
    generator = get_generator(backend)
    # A real run uses the VLM assessor so the WHOLE photo is inventoried (hands/neck/chest/
    # hair), not just the face-derived CV inventory. --dry-run stays fully offline (Mock).
    assessor = None if args.dry_run else GeminiVisionAssessor()
    cfg = PipelineConfig()
    if args.max_process_mp is not None:
        if args.max_process_mp <= 0:
            print("--max-process-mp must be greater than 0", file=sys.stderr)
            return 1
        cfg.max_process_mp = args.max_process_mp
    reports, rc = [], 0
    for p in paths:
        try:
            img = load(p)
            res = retouch(img.pixels, generator=generator, assessor=assessor, pipe_cfg=cfg,
                          samples=max(1, args.samples), max_escalate=max(0, args.max_escalate))
        except Exception as exc:
            print(f"{p.name}: ERROR {type(exc).__name__}: {exc}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            rc = 1
            continue
        reports.append(res.report)
        status = ("REFUSED" if not res.assessment.handleable
                  else "DELIVERED" if res.delivered else "FLAGGED")
        flagged = [v.op_id for v in res.verdicts if not v.clean]
        if not args.json:
            print(f"{p.name}: {status}  [{res.assessment.shot_type}, "
                  f"{len(res.retouch_map.ops)} ops, identity={res.identity['status']}]")
            if not res.assessment.handleable:
                print(f"    reason: {res.assessment.reason}")
            if flagged:
                print(f"    flagged regions: {', '.join(flagged)}")
        if not args.no_write:
            out_dir.mkdir(parents=True, exist_ok=True)
            rep = out_dir / f"{p.stem}.report.json"
            rep.write_text(json.dumps(res.report, indent=2))
            # Write the image when delivered, or when --force (labelled flagged) so a human
            # can judge it directly. Audit-gated by default: a flagged image is NOT shipped.
            if res.delivered or args.force:
                suffix = "v3" if res.delivered else "v3-flagged"
                outp = save_versioned(res.image, out_dir, p.stem, suffix=suffix,
                                      icc=img.icc, exif=img.exif, quality=cfg.jpeg_quality)
                if not args.json:
                    print(f"    output: {outp}")
            if not args.json:
                print(f"    report: {rep}")
    if args.json:
        print(json.dumps(reports, indent=2))
    return rc


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source = Path(args.source).expanduser()
    out_dir = Path(args.out_dir).expanduser()

    if not source.exists():
        print(f"Source not found: {source}", file=sys.stderr)
        return 1

    if args.engine == "v3":
        return _run_v3(source, out_dir, args)

    if not args.skip_preflight:
        if not _preflight(args.mode, source, out_dir, force=args.force):
            return 1
        print()

    cfg = PipelineConfig(mode=args.mode)
    if args.strength is not None:
        cfg.tone_strength = max(0.0, min(1.0, args.strength))
    if args.under_eye_texture is not None:
        cfg.under_eye_texture_strength = max(0.0, min(1.0, args.under_eye_texture))
    if args.skin_even is not None:
        cfg.skin_even_strength = max(0.0, min(1.0, args.skin_even))
    if args.max_mp is not None:
        if args.max_mp <= 0:
            print("--max-mp must be greater than 0", file=sys.stderr)
            return 1
        cfg.generator_max_mp = args.max_mp
    if args.max_process_mp is not None:
        if args.max_process_mp <= 0:
            print("--max-process-mp must be greater than 0", file=sys.stderr)
            return 1
        cfg.max_process_mp = args.max_process_mp

    try:
        marks = _parse_marks(args.mark, args.mark_box)
    except ValueError as exc:
        print(f"Invalid mark: {exc}", file=sys.stderr)
        return 1

    backend = "mock" if args.dry_run else args.backend
    if backend == "openai":
        err = _openai_ready()
        if err:
            print(err, file=sys.stderr)
            return 1
    generator = get_generator(backend, **({"model": args.model} if backend == "openai" else {}))

    try:
        results = retouch_path(source, out_dir, generator, cfg, marks=marks, write=not args.no_write)
    except Exception as exc:
        print(f"Retouch failed: {exc}", file=sys.stderr)
        return 1

    if not results:
        print(f"No images found in {source}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps([r.report for r in results], indent=2))
        return 0

    rejects = 0
    for r in results:
        verdict = r.qa.verdict.upper()
        if r.qa.verdict == "reject":
            rejects += 1
        print(f"{r.source_path.name}: {verdict}  "
              f"[align={r.align.method} {r.align.score:.2f}, edited={r.report['edited_fraction']:.1%}]")
        failed = r.qa.failed()
        if failed:
            print(f"    failed gates: {', '.join(failed)}")
        if r.output_path:
            print(f"    output: {r.output_path}")
            print(f"    review: {r.contact_sheet_path}")
    if rejects:
        print(f"\n{rejects}/{len(results)} flagged for review (artifacts written for inspection).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
