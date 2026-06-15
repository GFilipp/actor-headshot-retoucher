"""Test synthetic data. Single source of truth lives in the package
(`retoucher.demo`) so tests and the shipped examples can't drift apart."""
from __future__ import annotations

from retoucher.demo import (  # noqa: F401
    BLEMISHES,
    CORNER,
    RED_BLEMISH,
    SKIN,
    UNDER_EYE,
    disk,
    fake_geometry,
    luma,
    make_original,
    make_target,
    translate,
)
