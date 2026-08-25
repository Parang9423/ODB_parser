#!/usr/bin/env python3
"""Memory-efficient checked-Step composite helpers.

The selected PNL/STRIP/UNIT feature data is rendered in one hierarchy pass so
large PNL exports do not allocate one full-size raster per Step. The selected
Step profiles are also burned into the returned grayscale image by default.
"""
from __future__ import annotations

from typing import Iterable, Sequence

from PIL import Image, ImageChops, ImageDraw

from odb_cam_renderer import CompositeLayer


def _draw_selected_profiles(renderer, image: Image.Image, root_step: str,
                            visible_steps: set[str], value: int = 255,
                            width_px: int = 1) -> None:
    """Rasterize selected Step profile outlines into the root image in-place."""
    if not all(hasattr(renderer, name) for name in ("profile_bounds", "collect_instances", "transformed_profile")):
        return
    xmin, _ymin, _xmax, ymax = renderer.profile_bounds(root_step)
    draw = ImageDraw.Draw(image)
    width = max(1, int(width_px))

    def px(point):
        x, y = point
        return (
            (x - xmin) * renderer.dpi_x,
            (ymax - y) * renderer.dpi_y,
        )

    for instance in renderer.collect_instances(root_step):
        if instance.step not in visible_steps:
            continue
        for _kind, points in renderer.transformed_profile(instance):
            if len(points) < 2:
                continue
            coords = [px(point) for point in points]
            if points[0] != points[-1]:
                coords.append(coords[0])
            draw.line(coords, fill=int(value), width=width)


def _legacy_test_fallback(renderer, root_step: str, specs: Sequence[CompositeLayer],
                          visible: set[str], margin_px: int, background: int) -> Image.Image:
    """Compatibility fallback for minimal/fake renderers used by unit tests."""
    merged = None
    ordered = [step for step in ("pnl", "strip", "unit") if step in visible]
    ordered.extend(sorted(visible.difference(ordered)))
    for step in ordered:
        step_result = None
        for raw_spec in specs:
            spec = raw_spec.normalized()
            mask_image = renderer.render_hierarchy(root_step, spec.layer, {step}, margin_px)
            mask = mask_image.point(lambda value: 255 if value > 0 else 0, mode="L")
            if step_result is None:
                step_result = Image.new("L", mask_image.size, color=int(background))
            if spec.operation == "REPLACE":
                step_result.paste(spec.gv, mask=mask)
            elif spec.operation == "SUBTRACT":
                step_result.paste(0, mask=mask)
            else:
                contribution = Image.new("L", step_result.size, 0)
                contribution.paste(spec.gv, mask=mask)
                step_result = ImageChops.lighter(step_result, contribution)
        merged = step_result if merged is None else ImageChops.lighter(merged, step_result)
    if merged is None:
        raise ValueError("At least one composite layer is required")
    return merged


def render_selected_steps_composite(renderer, root_step: str,
                                    specs: Sequence[CompositeLayer],
                                    visible_steps: Iterable[str],
                                    margin_px: int = 0,
                                    background: int = 0,
                                    include_profiles: bool = True,
                                    profile_value: int = 255,
                                    profile_width_px: int = 1) -> Image.Image:
    """Render checked Step features and profiles in the root PNL coordinate system.

    Production ``FastODBRenderer`` uses one hierarchy pass for substantially
    lower peak memory. PNL/STRIP/UNIT profile outlines are included by default,
    so the saved image reflects the same Step selections as the viewer.
    """
    visible = {str(step).lower() for step in visible_steps}
    if not visible:
        raise ValueError("At least one visible Step is required")
    if not specs:
        raise ValueError("At least one composite layer is required")

    if hasattr(renderer, "render_composite_hierarchy"):
        image = renderer.render_composite_hierarchy(
            root_step, specs, visible, margin_px=margin_px, background=background
        )
    else:
        image = _legacy_test_fallback(renderer, root_step, specs, visible, margin_px, background)

    if include_profiles:
        _draw_selected_profiles(
            renderer, image, root_step, visible,
            value=profile_value, width_px=profile_width_px,
        )
    return image
