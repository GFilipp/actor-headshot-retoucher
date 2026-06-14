# Retouch Workflows

Use this reference after the readiness gate passes. The job is commercial actor retouching, not generic beautification.

## Contents

- Operating Standard
- Retouch Board
- Retouch Operation Log
- Minimum Viable Edit Threshold
- Light Retouching Mode
- Hybrid Map Mode
- Light Regen Mode
- Output Checklist

## Operating Standard

Assume the user wants material, human-visible improvements unless they explicitly ask for ultra-subtle cleanup. The edit can still be natural, but it must not be invisible.

Use a non-destructive mindset borrowed from serious photo tools:

- Keep the original untouched.
- Think in targeted operations, masks, and output versions.
- Keep a lightweight operation log, like a sidecar/profile record.
- QA at full frame and 100% crops before accepting.

## Retouch Board

Create and maintain one full-image retouch board. Include both user annotations and an independent professional scan.

Minimum scan:

- Full-frame first impression: marketability, fatigue, distractions, crop, pose, wardrobe.
- Eyes: tiredness, under-eye scaling, crepey lines, puffiness, red/brown eyelid color, beige or dull eye whites.
- Skin: forehead, bridge, cheek, nose, chest/torso when visible, neck, hands/thumbs, blotches, small marks, harsh color shifts.
- Hair: distracting flyaways only; preserve hairstyle and character.
- Wardrobe/background: preserve texture, lighting, and believable detail.

Do not solve only the newest complaint. Re-check the entire board after every major pass.

## Retouch Operation Log

Keep a concise working log. It can live in the final summary or a nearby markdown note; it does not require a formal sidecar file.

Template:

```text
Mode:
Source image:
Output image:
Target areas:
- Eyes / under-eyes:
- Eye whites / eyelids:
- Skin marks / chest:
- Neck:
- Thumb / hand:
- Hair:
Major operations:
- Area | mask/selection | operation | strength | intended visible result
Export:
- Format, resolution, quality, master/final status
QA:
- Full frame:
- 100% eyes:
- 100% neck:
- 100% thumb/hand:
- Authenticity:
- Minimum viable edit threshold:
Decision:
- Accepted / strengthen / switch to hybrid-map / switch to light regen / proof only
```

Use this log to avoid losing the whole retouch board after one round of feedback.

## Minimum Viable Edit Threshold

The edit passes only when a normal human viewer can see meaningful improvement without being told exactly where to look.

Reject a pass when:

- The only difference is measurable in pixels but not visually meaningful.
- Annotated problem areas still look essentially the same.
- Under-eye discoloration, scaling, or fine lines remain the first thing a client notices.
- Eye whites still look dull, beige, gray, or red when they were on the board.
- Neck or thumb/hand discoloration was flagged but barely changed.
- A chest, face, or skin blemish was flagged but remains obvious.
- The fix creates a new problem: bleached cheeks, broad blur, waxy skin, changed eye shape, or AI-looking smoothness.
- The result looks natural but not improved enough to justify a new version.

If a conservative local pass cannot meet this threshold, switch to hybrid-map or light regen. Do not keep iterating tiny invisible local changes.

## Light Retouching Mode

Use when the source is already close to final and needs subtle polish.

Prompt:

```text
Perform a conservative, non-regenerative retouch on this actor headshot.

Use the original image as the only canvas. Do not regenerate, replace, or invent facial features. Preserve identity, pose, expression, crop, hair shape, clothing texture, arms, hands, background, and lighting.

Inspect the image like a professional airbrusher and create an internal defect map. Prioritize tired eyes, under-eye fine lines and scaling, discoloration around the eyes, beige/off-color eye whites, small skin marks, neck discoloration, thumb/hand discoloration, and distracting flyaway hair.

Make changes noticeable enough to improve actor marketing value, but keep them natural. Preserve real skin texture, stubble, brows, lashes, moles, and natural facial structure.

Before finishing, compare before/after crops for both eyes, neck, thumb/hand, and full frame. Do not present the result if the marked issues are still obvious.
```

Operational notes:

- Preserve full source resolution.
- Use local masks or precise selections for small corrections, not global smoothing.
- Treat every correction as a layer-style operation: area, mask/selection, operation, strength, expected visible result.
- Keep pore texture, stubble, moles, and natural asymmetry.
- Use high-quality export settings and never overwrite the original.
- Reject the pass if the result is technically changed but not visibly improved.
- If material improvement requires stronger rescue work, stop pretending and choose hybrid-map or light regen.

## Hybrid Map Mode

Use as the default workflow for most actor headshot polish work. The purpose is to let image generation reveal a strong retouch target without letting it replace the real photograph by accident.

This mode is implemented by the deterministic `retoucher` pipeline. Generation proposes the target; code performs the alignment, masking, transfer, and QA. The steps below describe what the pipeline does; do not hand-apply them. Run `retoucher <source> --mode hybrid-map --out-dir <dir>`.

Workflow:

1. Use the clean original as the source of truth for identity, geometry, crop, wardrobe, hands, hair, lighting, and final resolution.
2. Use image generation to create a retouch map/proof from the clean original and any user annotations or crops. Treat annotated files as directional maps only.
3. Compare the map to the original and accept only the fixes that improve the retouch board without identity drift.
4. Transfer accepted fixes back onto the original through masks, local color/tone correction, healing/clone work, frequency-style texture cleanup, or other targeted operations.
5. Preserve original structure and believable detail. The final hybrid should look like the original photo professionally retouched, not a newly generated portrait.
6. QA the transferred result against the original and the imagegen map. Strengthen, switch to light regen, or label as proof if the mapped fixes cannot be reproduced cleanly.

Hybrid-map prompt (versioned in `retoucher/prompts.py`, which the pipeline imports; edit it there):

```text
Create a high-end actor headshot retouch map from this clean original.

Preserve the actor's exact identity, face shape, eye shape, expression, crop, pose, hair character, hands, wardrobe, background, and studio lighting. Do not create a new portrait.

Show the desired retouch direction clearly enough to guide final work on the original full-resolution file: healthier eyes, cleaner whites, reduced red/brown eyelid discoloration, smoother under-eye scaling and fine lines, cleaner small skin/chest marks, improved neck/thumb/hand discoloration when present, and only distracting flyaway reduction.

Keep real pores, stubble, asymmetry, fabric texture, and masculine character. Avoid waxy skin, broad cheek bleaching, painted eye whites, changed facial proportions, or AI-looking softness.
```

Transfer rules (the pipeline enforces these; do not perform them by hand):

- Apply the map to the original, not the original to the map. The original remains the final structure.
- Use tight masks around eye whites, lower lids, under-eyes, thumb/hand, neck, and blemishes. Do not wash out adjacent cheeks to hide eye problems.
- Fix the immediate red/brown eye area and scaly under-eye texture directly; do not substitute broad skin lightening.
- Retain lashes, eyelid edges, tear ducts, catchlights, iris shape, brow shape, stubble, moles, and skin texture.
- If the map fixes the problem but the transfer cannot do so visibly, escalate to light regen or present the imagegen result as a proof/final candidate with the quality tradeoff named.
- If the map changes face shape, eye shape, hand shape, crop, wardrobe, or identity, reject it and generate a stricter map.

## Light Regen Mode

Rare fallback only. Use when the deterministic hybrid-map transfer genuinely cannot fix the photo, especially around tired eyes, under-eye texture, discoloration, or skin tone. A fully regenerated image is what casting directors punish, so this is never the default. It can become the final candidate only when quality, resolution, and authenticity are good enough or the user explicitly accepts the tradeoff.

Prompt:

```text
Edit the clean unannotated actor headshot as the target image.

Use the annotated image only as a retouch map showing problem areas. Remove all annotation marks completely. Do not include black marker lines or any visible markup.

Preserve the actor's exact identity, facial structure, expression, pose, crop, camera angle, hairstyle, brows, lashes, stubble, moles, sweater, arms, hand shape, background, and original studio lighting. Do not create a new portrait. Do not beautify globally. This is a high-end actor headshot retouch, not a face replacement.

Retouch only the real photographic issues:
- reduce under-eye crepey texture, scaling, fine lines, puffiness, and dark brown/purple discoloration on both eyes
- brighten and neutralize the whites of the eyes while keeping them realistic
- reduce eyelid discoloration without changing eye shape
- clean small forehead, bridge, cheek, and skin marks
- reduce gray/brown/purple neck discoloration and harsh neck crease shadows
- reduce thumb/hand discoloration and dry crease texture
- tame only distracting flyaway hair
- keep natural pores, skin texture, stubble, and masculine character

The result should look like a world-class commercial actor/model retouch: rested, clean, polished, healthy, and marketable, while still unmistakably the same person and photograph.
```

Imagegen-specific safeguards:

- Use the clean original as the edit target.
- Use annotated images only as maps, never as output sources.
- Lock identity, eye shape, facial structure, hands, wardrobe, background, lighting, and crop.
- Do not accept an output that looks generated, airbrushed into wax, or subtly like a different person.
- Reject or label as proof if brows, lashes, skin pores, sweater texture, hands, or background show AI artifacts.
- If the output is lower resolution or loses real texture, call it a light-regen proof/final candidate, not a maximum-quality final, unless the user explicitly accepts that tradeoff.
- Require the same minimum viable edit threshold as local retouching: eyes, under-eyes, neck, and thumb/hand must be visibly improved when they are part of the retouch board.
- Never include black marker lines, annotation strokes, or old reference-file artifacts in the output.
- Do not reuse an older imagegen reference as a source file unless the user explicitly asks. Old references are direction only.

## Output Checklist

Before presenting output, inspect:

- Full frame: does the actor look more rested, sharp, healthy, and marketable?
- Both eyes: are fine-line scaling, puffiness, dull whites, and discoloration visibly improved?
- Eye authenticity: are whites realistic, not painted or glowing?
- Eye-area targeting: were red/brown lids and scaly under-eye texture addressed directly without bleaching the cheeks?
- Skin marks: are flagged chest, face, or visible blemishes materially reduced?
- Neck: is gray/brown/purple discoloration materially reduced?
- Thumb/hand: is discoloration and dry crease texture materially reduced?
- Face identity: no drift in expression, jaw, brows, lashes, eyelids, smile, or facial proportions.
- Wardrobe and hands: no AI artifacts or changed shape/texture.
- Export quality: maximum practical quality, original preserved, versioned output saved.
- Minimum viable edit threshold: the improvement is visible to a human eye without forensic comparison.

Reject and iterate when:

- The change is invisible in before/after crops.
- Skin is waxy or poreless.
- Cheeks or surrounding skin were bleached instead of fixing the eye area.
- Mask edges, color halos, or patch artifacts appear.
- Imagegen changed identity, clothing, hand shape, or realism.
- Black markup or annotation marks survived into the output.
- A low-resolution or detail-losing output is being mislabeled as final.
- The latest feedback item is addressed while older board items are forgotten.
