# Changelog

Versions follow the GitHub releases. The package version (`pyproject.toml`) is
the single source of truth.

## v2.1.0

Fixes from real-headshot A/B testing: feature protection, form-preserving colour, and corrections that no longer depend on the model.

- **Feature protection (new `faceparse.py`, bundled MediaPipe model).** Brows, eyes/lashes, lips, and nostrils are never edited; tone edits are confined to the eroded face oval (ears/hair/neck excluded) and heals to skin near the face (skin-coloured clothing is left alone). New QA gate fails if protected features change.
- **Form-preserving tone (chroma-only).** Colour correction moves only a*/b* in LAB and keeps the original luminance, so facial contour (e.g., the nose) is no longer flattened; an edge-aware guided filter removes visible patch edges.
- **Model-independent corrections.** Reddish/pigmented blemishes are detected by a LAB a* anomaly (not just darkness) and healed, and a dedicated under-eye / tear-trough corrector lifts shadow regardless of what the model proposed.
- **Forced fixes.** `--mark x,y[,r]` and `--mark-box` force a fix at a location the user points to.
- **Robustness.** The OpenAI backend fails fast with a clear message when the SDK or `OPENAI_API_KEY` is missing. Pinned to Python 3.12 (MediaPipe wheels); `opencv-contrib-python-headless` for the guided filter.

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
