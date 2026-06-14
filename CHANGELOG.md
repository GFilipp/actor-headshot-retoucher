# Changelog

Versions follow the GitHub releases. The package version (`pyproject.toml`) is
the single source of truth.

## v2.0.0

Major rebuild from an instruction-only skill into a deterministic retouch pipeline.

- The image model now only proposes a retouch *target*; deterministic code transfers validated, local fixes back onto the original: align (ECC / ORB), frequency-separated touch-up map, region masks, blend with the original's own texture, heal marks from surrounding pixels, neutralize any global colour cast.
- Automated QA gates: identity (optional), untouched-region SSIM / LPIPS, edit visibility (CIEDE2000), and texture-loss.
- The OpenAI model is auto-discovered at runtime; nothing is hard-pinned. Pin with `$OPENAI_IMAGE_MODEL` or `--model`.
- Spatial parameters scale with image resolution, so behaviour is consistent from 512px to 4096px.
- Adds a CLI, batch processing, packaging, an offline test suite, CI, and a reproducible example.

Breaking: behaviour and outputs differ from the v1.x prose workflow. The `retoucher` package and CLI are new.

## v1.2.0

Added hybrid-map as the explicit default workflow and readiness support for it (prose skill).

## v1.1.0

Added the minimum-viable-edit threshold and before/after QA expectations (prose skill).

## v1.0.0

Initial actor headshot retouch skill: prose workflow plus the readiness checker.
