from __future__ import annotations

from typing import List, Tuple

from aoi.contour_mapping import CropGeometry

Point = Tuple[float, float]


def rectangle_contour(
    geometry: CropGeometry,
    width_px: float,
    height_px: float,
    offset_px: Point = (0.0, 0.0),
) -> List[Point]:
    """Create a rectangular synthetic SEG contour around the crop center.

    The returned points use the same crop-pixel coordinate convention expected
    by contour_pixel_to_aoi_points()/contour_pixel_to_odb_points().
    """
    if width_px <= 0 or height_px <= 0:
        raise ValueError("synthetic contour width/height must be > 0")
    cx, cy = geometry.center_px
    ox, oy = map(float, offset_px)
    hx = float(width_px) / 2.0
    hy = float(height_px) / 2.0
    x = cx + ox
    y = cy + oy
    return [
        (x - hx, y - hy),
        (x + hx, y - hy),
        (x + hx, y + hy),
        (x - hx, y + hy),
    ]


def diamond_contour(
    geometry: CropGeometry,
    width_px: float,
    height_px: float,
    offset_px: Point = (0.0, 0.0),
) -> List[Point]:
    """Create a simple non-axis-aligned synthetic contour."""
    if width_px <= 0 or height_px <= 0:
        raise ValueError("synthetic contour width/height must be > 0")
    cx, cy = geometry.center_px
    ox, oy = map(float, offset_px)
    hx = float(width_px) / 2.0
    hy = float(height_px) / 2.0
    x = cx + ox
    y = cy + oy
    return [(x, y - hy), (x + hx, y), (x, y + hy), (x - hx, y)]
