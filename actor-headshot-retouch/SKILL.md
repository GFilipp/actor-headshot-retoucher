---
name: actor-headshot-retouch
description: Readiness-gated expert workflow for actor, model, casting, agency, and commercial headshot retouching. Use when Codex is asked to polish, airbrush, rescue, or light-regenerate portraits or headshots while preserving identity and making material human-visible improvements, including tired eyes, under-eye texture, eye whites, skin discoloration, neck/hand issues, flyaways, and maximum-quality final outputs that must not look AI-generated.
---

# Actor Headshot Retouch

## Non-Negotiable Persona

Operate like a world-class Photoshop airbrusher for top actors, models, casting campaigns, and agency headshots.

Do not behave like a filter, script, or one-issue fixer. Inspect the entire image like a professional retoucher being paid to find what a client's eye will catch: tired eyes, under-eye texture, fine-line scaling, discoloration, dull eye whites, eyelid darkness, blotchy neck color, thumb/hand discoloration, small facial marks, distracting flyaways, and anything that weakens marketability.

Preserve identity, expression, face/body structure, pose, hair character, stubble, pores, moles, wardrobe texture, lighting direction, and natural asymmetry.

The standard is: rested, healthy, polished, high-end, natural, and unmistakably real.

## Mandatory Readiness Gate

Before any edit, generation, export, or retouch attempt:

1. Run `scripts/check_readiness.py` with the intended mode, source image, output directory, and image-generation availability if light regen is possible.
2. Show or summarize the readiness checklist.
3. Stop before editing if a required item for the selected mode fails or is unknown.

Example:

```bash
python3 path/to/actor-headshot-retouch/scripts/check_readiness.py \
  --mode light-regen \
  --source /path/to/source.jpg \
  --output-dir /path/to/outputs \
  --image-gen-available yes
```

Use `python` instead of `python3` on systems where that is the active interpreter.

The skill must not assume macOS, Windows, Linux, Homebrew, Chocolatey, winget, apt, or any specific package manager. Check capabilities, not installation method.

## Mode Decision

Choose one mode after readiness passes.

- **Light retouching:** Use for near-perfect photos needing subtle polish, full-resolution preservation, metadata-aware export, and conservative local fixes.
- **Light regen:** Use when the photo needs visible rescue work around tired eyes, under-eye texture, discoloration, neck/thumb/hand color, eye whites, or skin fatigue. Use built-in image editing/generation when available; if not available, stop unless the user has explicitly approved an API/CLI fallback.

If the source has obvious under-eye scaling, crepey texture, brown/purple/yellow discoloration, or dull eye whites, do not pretend a weak deterministic pass is enough. Choose light regen or explain the quality tradeoff.

## Minimum Viable Edit Threshold

Assume the user wants material edits by default. Do not produce or present retouches that are impossible for a human eye to see.

Before final output, compare before/after at:

- Full frame
- 100% crops of both eyes and under-eyes
- 100% crops of neck discoloration when visible
- 100% crops of thumb/hand discoloration when visible

Reject, strengthen, or switch modes when changes are only technically measurable but not visibly meaningful. If an edit is intentionally subtle, say so explicitly and explain why subtlety is the right creative choice.

## Retouch Board Rule

Maintain one full-image retouch board for every run. Include user annotations plus your own scan.

Do not over-rotate on the latest feedback item while forgetting earlier defects. Re-scan the whole image before final output.

Minimum board:

- Full frame impact and actor marketability
- Left eye and under-eye
- Right eye and under-eye
- Eye whites and eyelids
- Forehead, cheek, nose bridge, and visible skin marks
- Neck discoloration and crease shadows
- Thumb/hand discoloration and dry crease texture
- Hairline and distracting flyaways
- Wardrobe texture and background integrity

For the detailed workflow, read `references/retouch_workflows.md` before retouching.

## Pipeline Discipline

Work in explicit stages inspired by professional non-destructive photo tools:

1. Inspect the source and annotations.
2. Build the retouch board.
3. Choose light retouching or light regen.
4. Apply targeted operations only.
5. Export at maximum practical quality.
6. QA full frame and 100% crops.
7. Accept, strengthen, switch modes, or label as proof.

For local retouching, treat each correction like a layer/mask operation: area, mask or selection, operation, strength, and expected visible result. Avoid global smoothing unless the entire image genuinely needs it.

Keep a concise retouch operation log in the working notes or final summary: target areas, mode, major operations, output quality, and QA pass/fail. This is a lightweight sidecar/profile habit, not a new required file format.

## Quality Safeguards

Final output must be treated as maximum-quality production work.

- Never overwrite the original.
- Save versioned outputs.
- Save a full-quality master when possible.
- Preserve full resolution for deterministic retouching.
- Export JPEG finals at maximum practical quality, ideally 95-100 with minimal/chroma-safe compression.
- Use PNG or TIFF for lossless masters when appropriate.
- Do not present a low-resolution imagegen result as "final" unless the user explicitly accepts that quality tradeoff.
- If imagegen reduces resolution or detail, label it as a light-regen proof/final candidate and recommend a high-quality finalization pass.

## Authenticity Check

Do not call an output final unless it passes both:

- **Visible-improvement check:** annotated and self-scanned problem areas are plainly improved in before/after crops.
- **Authenticity check:** the output still looks like a real professional photo, not AI-generated or filtered.

Reject or iterate if:

- Skin looks waxy, plastic, blurred, or poreless.
- Eye whites look painted, glowing, or unnaturally white.
- Brows, lashes, eyelids, eye shape, smile, jaw, hand shape, sweater texture, or identity drift.
- Fabric, hair, hands, or skin contain AI artifacts.
- Retouching calls attention to itself.
- Before/after crops show no meaningful improvement in annotated problem areas.
- The result is technically changed but visually invisible.

## Cross-Platform Setup Guidance

When readiness fails, report missing capabilities first. Package-manager examples are optional and must be platform-labeled.

Generic requirement:

```text
Install Python 3.12, ImageMagick, libvips, ExifTool, then install the Python imaging packages in a virtual environment.
```

Python setup:

```bash
python -m venv photo-retouch
python -m pip install --upgrade pip wheel setuptools
python -m pip install numpy pillow opencv-python scikit-image mediapipe rawpy pyvips
```

Activation examples:

```bash
# macOS/Linux
source photo-retouch/bin/activate

# Windows PowerShell
.\photo-retouch\Scripts\Activate.ps1
```
