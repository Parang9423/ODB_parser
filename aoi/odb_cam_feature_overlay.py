from __future__ import annotations

from typing import Iterable, Sequence

from PIL import Image, ImageDraw

from aoi.contour_mapping import AoiOdbTransform

Point = tuple[float, float]

_FEATURE_COLORS = {
    "P": (0, 255, 255),
    "L": (255, 255, 0),
    "S": (255, 0, 255),
}
_SEG_COLOR = (255, 0, 0)
_INTERSECTION_COLOR = (0, 255, 0)


def odb_point_to_cam_pixel(
    odb_x_mm: float,
    odb_y_mm: float,
    *,
    image_center_aoi_mm: Point,
    cam_size_px: tuple[int, int],
    resolution_um_per_px: float,
    transform: AoiOdbTransform,
) -> Point:
    """Map an ODB millimetre point directly into the matched CAM crop pixels.

    Inverse of the current AOI -> ODB SWAP_X+_Y- transform:
        AOI_X = ty - ODB_Y
        AOI_Y = ODB_X - tx

    The G and C crops share the same physical centre and pixel resolution, so
    CAM pixel coordinates can be calculated directly around the CAM centre.
    """
    if resolution_um_per_px <= 0:
        raise ValueError("resolution_um_per_px must be > 0")
    width, height = cam_size_px
    if width <= 0 or height <= 0:
        raise ValueError("CAM dimensions must be > 0")

    aoi_x_mm = transform.ty_mm - float(odb_y_mm)
    aoi_y_mm = float(odb_x_mm) - transform.tx_mm
    center_x_mm, center_y_mm = image_center_aoi_mm
    res_mm = resolution_um_per_px / 1000.0
    cam_cx = (width - 1) / 2.0
    cam_cy = (height - 1) / 2.0
    return (
        cam_cx + (aoi_x_mm - center_x_mm) / res_mm,
        cam_cy + (aoi_y_mm - center_y_mm) / res_mm,
    )


def geometry_to_cam_pixels(
    geometry,
    *,
    image_center_aoi_mm: Point,
    cam_size_px: tuple[int, int],
    resolution_um_per_px: float,
    transform: AoiOdbTransform,
    clip: bool = True,
):
    """Transform a Shapely ODB geometry (mm) into CAM pixel coordinates."""
    try:
        from shapely.geometry import box
        from shapely.ops import transform as shapely_transform
    except ImportError as exc:
        raise RuntimeError("Shapely is required for ODB feature overlays") from exc

    def mapper(x, y, z=None):
        # Shapely 2 may pass scalar values or array-like coordinate sequences.
        try:
            iterator = iter(x)
        except TypeError:
            px, py = odb_point_to_cam_pixel(
                x, y,
                image_center_aoi_mm=image_center_aoi_mm,
                cam_size_px=cam_size_px,
                resolution_um_per_px=resolution_um_per_px,
                transform=transform,
            )
            return (px, py) if z is None else (px, py, z)

        xs = list(iterator)
        ys = list(y)
        mapped = [
            odb_point_to_cam_pixel(
                ox, oy,
                image_center_aoi_mm=image_center_aoi_mm,
                cam_size_px=cam_size_px,
                resolution_um_per_px=resolution_um_per_px,
                transform=transform,
            )
            for ox, oy in zip(xs, ys)
        ]
        pxs = [p[0] for p in mapped]
        pys = [p[1] for p in mapped]
        return (pxs, pys) if z is None else (pxs, pys, z)

    mapped = shapely_transform(mapper, geometry)
    if clip and not mapped.is_empty:
        width, height = cam_size_px
        mapped = mapped.intersection(box(0.0, 0.0, float(width - 1), float(height - 1)))
    return mapped


def _draw_geometry_boundary(draw: ImageDraw.ImageDraw, geometry, color, width: int) -> None:
    if geometry is None or geometry.is_empty:
        return
    geom_type = geometry.geom_type
    if geom_type == "Polygon":
        exterior = [(float(x), float(y)) for x, y in geometry.exterior.coords]
        if len(exterior) >= 2:
            draw.line(exterior, fill=color, width=width, joint="curve")
        for ring in geometry.interiors:
            points = [(float(x), float(y)) for x, y in ring.coords]
            if len(points) >= 2:
                draw.line(points, fill=color, width=width, joint="curve")
        return
    if geom_type in {"MultiPolygon", "GeometryCollection", "MultiLineString"}:
        for child in geometry.geoms:
            _draw_geometry_boundary(draw, child, color, width)
        return
    if geom_type in {"LineString", "LinearRing"}:
        points = [(float(x), float(y)) for x, y in geometry.coords]
        if len(points) >= 2:
            draw.line(points, fill=color, width=width)


def draw_odb_feature_overlay(
    cam_image: Image.Image,
    *,
    seg_contours_px: Iterable[Sequence[Point]],
    feature_geometries: Iterable[tuple[str, object]],
    intersection_geometries: Iterable[object] = (),
    line_width: int = 2,
) -> Image.Image:
    """Draw SEG and intersecting ODB feature boundaries on a CAM image.

    Colors: SEG=red, P=cyan, L=yellow, S=magenta. Exact defect/feature
    intersection boundaries are green when supplied.
    """
    if line_width <= 0:
        raise ValueError("line_width must be > 0")
    output = cam_image.convert("RGB").copy()
    draw = ImageDraw.Draw(output)

    for primitive_type, geometry in feature_geometries:
        color = _FEATURE_COLORS.get(str(primitive_type).upper(), (255, 255, 255))
        _draw_geometry_boundary(draw, geometry, color, line_width)

    for geometry in intersection_geometries:
        _draw_geometry_boundary(draw, geometry, _INTERSECTION_COLOR, max(1, line_width))

    # Draw SEG last so the AI contour remains visible above ODB geometry.
    for contour in seg_contours_px:
        points = [(float(x), float(y)) for x, y in contour]
        if len(points) >= 2:
            draw.line(points + [points[0]], fill=_SEG_COLOR, width=line_width, joint="curve")
    return output
