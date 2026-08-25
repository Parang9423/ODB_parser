#!/usr/bin/env python3
"""Memory-efficient checked-Step composite helpers.

The selected PNL/STRIP/UNIT feature data is rendered in one hierarchy pass so
large PNL exports do not allocate one full-size raster per Step. Optionally the
selected Step profiles can be burned into the saved grayscale PNG as well.
"""
from __future__ import annotations

from typing import Iterable, Sequence

from PIL import Image, ImageDraw

from odb_cam_renderer import CompositeLayer


def _draw_selected_profiles(renderer, image: Image.Image, root_step: str,
                            visible_steps: set[str], value: int = 255,
                            width_px: int = 1) -> None:
    """Rasterize selected Step profile outlines into the root image in-place."""
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


def render_selected_steps_composite(renderer, root_step: str,
                                    specs: Sequence[CompositeLayer],
                                    visible_steps: Iterable[str],
                                    margin_px: int = 0,
                                    background: int = 0,
                                    include_profiles: bool = False,
                                    profile_value: int = 255,
                                    profile_width_px: int = 1) -> Image.Image:
    """Render all checked Steps in one root-coordinate composite.

    This delegates feature rendering to ``render_composite_hierarchy`` with the
    exact checked-Step set. Unlike the previous implementation it does not make
    separate PNL/STRIP/UNIT full-size images, which is important for high-DPI
    panel exports. When ``include_profiles`` is true, the checked Step profile
    outlines are rasterized into the returned PNG too.
    """
    visible = {str(step).lower() for step in visible_steps}
    if not visible:
        raise ValueError("At least one visible Step is required")
    if not specs:
        raise ValueError("At least one composite layer is required")

    image = renderer.render_composite_hierarchy(
        root_step,
        specs,
        visible,
        margin_px=margin_px,
        background=background,
    )
    if include_profiles:
        _draw_selected_profiles(
            renderer,
            image,
            root_step,
            visible,
            value=profile_value,
            width_px=profile_width_px,
        )
    return image
