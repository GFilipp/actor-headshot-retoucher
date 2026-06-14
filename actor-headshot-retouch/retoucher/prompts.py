"""Retouch prompts, relocated from references/retouch_workflows.md.

These are the source of truth that the generator imports. The markdown
reference now points here so the wording is versioned in code and cannot drift
from what the pipeline actually sends.
"""
from __future__ import annotations

LIGHT_RETOUCH = """\
Perform a conservative, non-regenerative retouch on this actor headshot.

Use the original image as the only canvas. Do not regenerate, replace, or invent facial features. Preserve identity, pose, expression, crop, hair shape, clothing texture, arms, hands, background, and lighting.

Inspect the image like a professional airbrusher and create an internal defect map. Prioritize tired eyes, under-eye fine lines and scaling, discoloration around the eyes, beige/off-color eye whites, small skin marks, neck discoloration, thumb/hand discoloration, and distracting flyaway hair.

Make changes noticeable enough to improve actor marketing value, but keep them natural. Preserve real skin texture, stubble, brows, lashes, moles, and natural facial structure.

Before finishing, compare before/after crops for both eyes, neck, thumb/hand, and full frame. Do not present the result if the marked issues are still obvious.
"""

HYBRID_MAP = """\
Create a high-end actor headshot retouch map from this clean original.

Preserve the actor's exact identity, face shape, eye shape, expression, crop, pose, hair character, hands, wardrobe, background, and studio lighting. Do not create a new portrait.

Show the desired retouch direction clearly enough to guide final work on the original full-resolution file: healthier eyes, cleaner whites, reduced red/brown eyelid discoloration, smoother under-eye scaling and fine lines, cleaner small skin/chest marks, improved neck/thumb/hand discoloration when present, and only distracting flyaway reduction.

Keep real pores, stubble, asymmetry, fabric texture, and masculine character. Avoid waxy skin, broad cheek bleaching, painted eye whites, changed facial proportions, or AI-looking softness.
"""

LIGHT_REGEN = """\
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
"""

PROMPTS = {
    "light-retouch": LIGHT_RETOUCH,
    "hybrid-map": HYBRID_MAP,
    "light-regen": LIGHT_REGEN,
}


def prompt_for(mode: str) -> str:
    """Return the prompt text for a mode, defaulting to the hybrid-map map prompt."""
    return PROMPTS.get(mode, HYBRID_MAP)
