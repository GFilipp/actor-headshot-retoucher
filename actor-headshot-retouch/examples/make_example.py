"""Generate the demonstration artifacts in this folder.

No API key and no real photo required: it runs the full pipeline on synthetic
data (retoucher.demo) using a mock generator that returns the "good retouch"
target, so the result is a clean, passing before/after that shows alignment,
masked tone transfer, mark healing, and the QA gates working.

    python examples/make_example.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Run straight from a clone, no install required: put the package dir on the path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image  # noqa: E402

from retoucher import MockGenerator, PipelineConfig  # noqa: E402
from retoucher.demo import make_original, make_target  # noqa: E402
from retoucher.image_io import to_uint8  # noqa: E402
from retoucher.pipeline import retouch_image  # noqa: E402

HERE = Path(__file__).resolve().parent


def main() -> None:
    # Clean prior generated artifacts so names stay stable (v1).
    for p in HERE.glob("synthetic_*"):
        p.unlink()

    before = HERE / "synthetic.png"
    Image.fromarray(to_uint8(make_original())).save(before)

    generator = MockGenerator(transform=lambda _img: make_target())
    res = retouch_image(before, HERE, generator, PipelineConfig(mode="hybrid-map"))

    print(f"verdict: {res.qa.verdict}")
    print(f"alignment: {res.align.method} ({res.align.score:.2f})")
    print(f"edited fraction: {res.report['edited_fraction']:.1%}")
    for g in res.qa.gates:
        v = "" if g.value is None else f" = {g.value}"
        print(f"  {g.name}: {g.status}{v}")
    print(f"output:        {res.output_path}")
    print(f"contact sheet: {res.contact_sheet_path}")
    print(f"report:        {res.report_path}")


if __name__ == "__main__":
    main()
