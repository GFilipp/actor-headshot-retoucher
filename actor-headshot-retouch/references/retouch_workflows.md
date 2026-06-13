# Retouch Workflows

Use this reference after the readiness gate passes. The job is commercial actor retouching, not generic beautification.

## Retouch Board

Create and maintain one full-image retouch board. Include both user annotations and an independent professional scan.

Minimum scan:

- Full-frame first impression: marketability, fatigue, distractions, crop, pose, wardrobe.
- Eyes: tiredness, under-eye scaling, crepey lines, puffiness, eyelid darkness, beige or dull eye whites.
- Skin: forehead, bridge, cheek, nose, neck, hands/thumbs, blotches, small marks, harsh color shifts.
- Hair: distracting flyaways only; preserve hairstyle and character.
- Wardrobe/background: preserve texture, lighting, and believable detail.

Do not solve only the newest complaint. Re-check the entire board after every major pass.

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
- Use local masks for small corrections, not global smoothing.
- Keep pore texture, stubble, moles, and natural asymmetry.
- Use high-quality export settings and never overwrite the original.
- Reject the pass if the result is technically changed but not visibly improved.

## Light Regen Mode

Use when deterministic retouching cannot cleanly fix the photo, especially around tired eyes, under-eye texture, discoloration, or skin tone.

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
- If the output is lower resolution or loses real texture, call it a light-regen proof/final candidate, not a maximum-quality final, unless the user explicitly accepts that tradeoff.

## Output Checklist

Before presenting output, inspect:

- Full frame: does the actor look more rested, sharp, healthy, and marketable?
- Both eyes: are fine-line scaling, puffiness, dull whites, and discoloration visibly improved?
- Eye authenticity: are whites realistic, not painted or glowing?
- Neck: is gray/brown/purple discoloration materially reduced?
- Thumb/hand: is discoloration and dry crease texture materially reduced?
- Face identity: no drift in expression, jaw, brows, lashes, eyelids, smile, or facial proportions.
- Wardrobe and hands: no AI artifacts or changed shape/texture.
- Export quality: maximum practical quality, original preserved, versioned output saved.

Reject and iterate when:

- The change is invisible in before/after crops.
- Skin is waxy or poreless.
- Mask edges, color halos, or patch artifacts appear.
- Imagegen changed identity, clothing, hand shape, or realism.
- A low-resolution or detail-losing output is being mislabeled as final.
