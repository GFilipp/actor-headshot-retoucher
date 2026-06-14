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
import sys
from pathlib import Path

from .config import PipelineConfig
from .generate import get_generator
from .pipeline import retouch_path


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
    p.add_argument("--strength", type=float, default=None, help="Transfer strength 0..1.")
    p.add_argument("--max-mp", type=float, default=None, help="Max megapixels sent to the generator.")
    p.add_argument("--no-write", action="store_true", help="Run without writing outputs.")
    p.add_argument("--skip-preflight", action="store_true", help="Skip the readiness check.")
    p.add_argument("--force", action="store_true", help="Run even if preflight fails.")
    p.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source = Path(args.source).expanduser()
    out_dir = Path(args.out_dir).expanduser()

    if not source.exists():
        print(f"Source not found: {source}", file=sys.stderr)
        return 1

    if not args.skip_preflight:
        if not _preflight(args.mode, source, out_dir, force=args.force):
            return 1
        print()

    cfg = PipelineConfig(mode=args.mode)
    if args.strength is not None:
        cfg.strength = args.strength
    if args.max_mp is not None:
        cfg.generator_max_mp = args.max_mp

    backend = "mock" if args.dry_run else args.backend
    generator = get_generator(backend, **({"model": args.model} if backend == "openai" else {}))

    try:
        results = retouch_path(source, out_dir, generator, cfg, write=not args.no_write)
    except Exception as exc:
        print(f"Retouch failed: {exc}", file=sys.stderr)
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
