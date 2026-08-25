#!/usr/bin/env python3
"""Step-aware composite helper.

Build each selected Step (PNL/STRIP/UNIT) as an independent composite in the
root coordinate system, then merge the selected Step images. This makes the
Step visibility checkboxes map directly to both preview and saved PNG output.
"""
from __future__ import annotations

from typing import Iterable, Sequence

from PIL import Image, ImageChops

from odb_cam_renderer import CompositeLayer


def _composite_one_step(renderer, root_step: str, specs: Sequence[CompositeLayer], step: str,
                        margin_px: int = 0, background: int = 0) -> Image.Image:
    result = None
    for raw_spec in specs:
        spec = raw_spec.normalized()
        mask_image = renderer.render_hierarchy(root_step, spec.layer, {step}, margin_px)
        mask = mask_image.point(lambda value: 255 if value > 0 else 0, mode="L")
        if result is None:
            result = Image.new("L", mask_image.size, color=int(background))

        if spec.operation == "REPLACE":
            result.paste(spec.gv, mask=mask)
        elif spec.operation == "SUBTRACT":
            result.paste(0, mask=mask)
        else:
            contribution = Image.new("L", result.size, color=0)
            contribution.paste(spec.gv, mask=mask)
            result = ImageChops.lighter(result, contribution)

    if result is None:
        raise ValueError("At least one composite layer is required")
    return result


def render_selected_steps_composite(renderer, root_step: str, specs: Sequence[CompositeLayer],
                                    visible_steps: Iterable[str], margin_px: int = 0,
                                    background: int = 0) -> Image.Image:
    """Render checked Steps independently and merge them in the root coordinate system.

    Step order is deterministic: PNL -> STRIP -> UNIT -> any other selected Step.
    The merge uses pixel-wise maximum so each checked Step remains visible in the
    final grayscale CAM instead of being lost inside a single cross-Step mask.
    """
    visible = {str(step).lower() for step in visible_steps}
    if not visible:
        raise ValueError("At least one visible Step is required")
    if not specs:
        raise ValueError("At least one composite layer is required")

    ordered = [step for step in ("pnl", "strip", "unit") if step in visible]
    ordered.extend(sorted(visible.difference(ordered)))

    merged = None
    for step in ordered:
        step_image = _composite_one_step(renderer, root_step, specs, step, margin_px, background)
        merged = step_image if merged is None else ImageChops.lighter(merged, step_image)

    assert merged is not None
    return merged
