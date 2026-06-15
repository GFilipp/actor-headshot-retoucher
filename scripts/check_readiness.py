#!/usr/bin/env python3
"""Readiness gate for actor-headshot-retouch.

Checks capabilities rather than package managers so it works across macOS,
Windows, and Linux.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class Check:
    name: str
    status: str
    required: bool
    detail: str

    @property
    def ok(self) -> bool:
        return self.status == "pass"


# The deterministic pipeline runs on these four. They are the only required
# Python dependencies.
CORE_IMPORTS = [
    ("numpy", "numpy"),
    ("Pillow", "PIL"),
    ("OpenCV", "cv2"),
    ("scikit-image", "skimage"),
]

# Optional: stronger masks / quality gates / inputs. Missing ones are reported,
# never blocking; the pipeline degrades gracefully.
OPTIONAL_IMPORTS = [
    ("MediaPipe", "mediapipe"),    # quality path: feature protection (Python 3.12)
    ("OpenAI SDK", "openai"),      # required only for --backend openai
    ("InsightFace", "insightface"),
    ("LPIPS", "lpips"),
    ("rawpy", "rawpy"),
    ("pyvips", "pyvips"),
]

# Optional CLI tools (nice for 16-bit masters / metadata). Never blocking.
CLI_TOOLS = [
    ("ImageMagick", "magick"),
    ("libvips", "vips"),
    ("ExifTool", "exiftool"),
]


def nearest_existing_parent(path: Path) -> Path | None:
    current = path
    while not current.exists() and current != current.parent:
        current = current.parent
    return current if current.exists() else None


def is_writable_dir(path: Path) -> tuple[bool, str]:
    if path.exists():
        if not path.is_dir():
            return False, f"exists but is not a directory: {path}"
        return os.access(path, os.W_OK), str(path)

    parent = nearest_existing_parent(path.parent)
    if parent is None:
        return False, f"no existing parent for: {path}"
    return os.access(parent, os.W_OK), f"directory does not exist; parent checked: {parent}"


def check_python_stack(required: bool) -> Check:
    missing: list[str] = []
    versions: list[str] = []
    optional_notes: list[str] = []

    version = sys.version_info
    if version < (3, 10):
        missing.append(f"Python 3.10+ (found {platform.python_version()})")
    else:
        versions.append(f"Python {platform.python_version()}")

    for label, module in CORE_IMPORTS:
        try:
            imported = importlib.import_module(module)
        except Exception as exc:  # pragma: no cover - exact import failures vary
            missing.append(f"{label} ({module}): {exc.__class__.__name__}")
            continue
        module_version = getattr(imported, "__version__", None)
        versions.append(f"{label}{' ' + module_version if module_version else ''}")

    for label, module in OPTIONAL_IMPORTS:
        try:
            importlib.import_module(module)
            optional_notes.append(f"{label}: present")
        except Exception:
            optional_notes.append(f"{label}: absent (optional)")

    detail = "; ".join(versions + optional_notes)
    if missing:
        return Check("Python imaging stack", "fail", required, "missing core: " + "; ".join(missing))
    return Check("Python imaging stack", "pass", required, detail)


def check_cli_tool(label: str, command: str, required: bool) -> Check:
    path = shutil.which(command)
    if path:
        return Check(label, "pass", required, path)
    return Check(label, "fail", required, f"`{command}` not found on PATH")


def check_image_gen(value: str, required: bool) -> Check:
    if value == "yes":
        return Check("image_gen available for retouch map / light regen", "pass", required, "confirmed by Codex tool context")
    if value == "no":
        return Check("image_gen available for retouch map / light regen", "fail", required, "not available in current tool context")
    return Check(
        "image_gen available for retouch map / light regen",
        "unknown",
        required,
        "must be confirmed by Codex; this is not a local CLI capability",
    )


def check_source(path_text: str | None, required: bool) -> Check:
    if not path_text:
        return Check("Source image readable", "unknown", required, "no --source provided")
    path = Path(path_text).expanduser()
    if path.is_file() and os.access(path, os.R_OK):
        return Check("Source image readable", "pass", required, str(path))
    return Check("Source image readable", "fail", required, f"not readable: {path}")


def check_output_dir(path_text: str | None, required: bool) -> Check:
    path = Path(path_text or ".").expanduser()
    ok, detail = is_writable_dir(path)
    return Check("Output folder writable", "pass" if ok else "fail", required, detail)


def required_names_for_mode(mode: str) -> set[str]:
    base = {"Source image readable", "Output folder writable"}
    stack = {"Python imaging stack"}
    regen = {"image_gen available for retouch map / light regen"}

    # CLI tools (ImageMagick/libvips/ExifTool) are optional everywhere: the
    # pipeline uses OpenCV/Pillow, not those binaries.
    if mode in {"local", "light-retouch"}:
        return base | stack
    if mode == "light-regen":
        return base | regen
    if mode == "hybrid-map":
        return base | stack | regen
    return base | stack | regen


def build_checks(args: argparse.Namespace) -> list[Check]:
    required = required_names_for_mode(args.mode)
    checks: list[Check] = []

    checks.append(check_python_stack("Python imaging stack" in required))
    for label, command in CLI_TOOLS:
        checks.append(check_cli_tool(label, command, label in required))
    checks.append(check_image_gen(args.image_gen_available, "image_gen available for retouch map / light regen" in required))
    checks.append(check_source(args.source, "Source image readable" in required))
    checks.append(check_output_dir(args.output_dir, "Output folder writable" in required))
    return checks


def blocking_failures(checks: Iterable[Check]) -> list[Check]:
    return [check for check in checks if check.required and check.status != "pass"]


def format_check(check: Check) -> str:
    marker = "[x]" if check.status == "pass" else "[ ]" if check.status == "fail" else "[?]"
    required = "required" if check.required else "optional"
    return f"{marker} {check.name} ({required}) - {check.detail}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check readiness for actor headshot retouching.")
    parser.add_argument(
        "--mode",
        choices=["all", "local", "light-retouch", "hybrid-map", "light-regen"],
        default="all",
        help="Capability set to require.",
    )
    parser.add_argument("--source", help="Source image path to verify.")
    parser.add_argument("--output-dir", help="Output directory to verify. Defaults to current directory.")
    parser.add_argument(
        "--image-gen-available",
        choices=["yes", "no", "unknown"],
        default="unknown",
        help="Whether Codex's image_gen tool is available for retouch maps or light regen.",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    args = parser.parse_args(argv)

    checks = build_checks(args)
    failures = blocking_failures(checks)

    if args.json:
        payload = {
            "mode": args.mode,
            "ready": not failures,
            "platform": platform.platform(),
            "python": platform.python_version(),
            "checks": [asdict(check) for check in checks],
            "blocking_failures": [asdict(check) for check in failures],
        }
        print(json.dumps(payload, indent=2))
    else:
        print(f"Actor headshot retouch readiness ({args.mode})")
        print(f"Platform: {platform.platform()}")
        print()
        print("Readiness:")
        for check in checks:
            print(format_check(check))
        print()
        if failures:
            print("Status: NOT READY")
            print("Blocking items:")
            for check in failures:
                print(f"- {check.name}: {check.detail}")
        else:
            print("Status: READY")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
