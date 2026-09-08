from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

Point = Tuple[float, float]


@dataclass(frozen=True)
class CropGeometry:
    width_px: int
    height_px: int
    resolution_um_per_px_x: float
    resolution_um_per_px_y: float

    def __post_init__(self) -> None:
        if self.width_px <= 0 or self.height_px <= 0:
            raise ValueError("image width/height must be > 0")
        if self.resolution_um_per_px_x <= 0 or self.resolution_um_per_px_y <= 0:
            raise ValueError("pixel resolution must be > 0")

    @classmethod
    def isotropic(cls, width_px: int, height_px: int, resolution_um_per_px: float) -> "CropGeometry":
        return cls(width_px, height_px, resolution_um_per_px, resolution_um_per_px)

    @property
    def center_px(self) -> Point:
        # Geometric center for zero-based pixel coordinates: e.g. 100 px -> 49.5.
        return ((self.width_px - 1) / 2.0, (self.height_px - 1) / 2.0)

    @property
    def resolution_mm_per_px(self) -> Point:
        return (self.resolution_um_per_px_x / 1000.0, self.resolution_um_per_px_y / 1000.0)


@dataclass(frozen=True)
class AoiOdbTransform:
    """Current SWAP_X+_Y- AOI(mm) -> ODB(mm) affine transform.

    odb_x = aoi_y + tx
    odb_y = -aoi_x + ty

    tx/ty are product-specific translations derived from the ERT inspection
    frame and ODB STRIP-array geometry; they must not be hard-coded globally.
    """

    tx_mm: float
    ty_mm: float

    def point(self, aoi_x_mm: float, aoi_y_mm: float) -> Point:
        return (aoi_y_mm + self.tx_mm, -aoi_x_mm + self.ty_mm)


def contour_pixel_to_aoi_points(
    contour_px: Iterable[Point],
    image_center_aoi_mm: Point,
    geometry: CropGeometry,
) -> List[Point]:
    """Map arbitrary-size crop contour points to global AOI millimetres."""
    center_x_mm, center_y_mm = image_center_aoi_mm
    center_px_x, center_px_y = geometry.center_px
    res_x, res_y = geometry.resolution_mm_per_px

    return [
        (
            center_x_mm + (float(px) - center_px_x) * res_x,
            center_y_mm + (float(py) - center_px_y) * res_y,
        )
        for px, py in contour_px
    ]


def contour_pixel_to_odb_points(
    contour_px: Iterable[Point],
    image_center_aoi_mm: Point,
    geometry: CropGeometry,
    transform: AoiOdbTransform,
) -> List[Point]:
    """Map a SEG contour directly from crop pixels to ODB millimetres."""
    aoi_points = contour_pixel_to_aoi_points(contour_px, image_center_aoi_mm, geometry)
    return [transform.point(x, y) for x, y in aoi_points]


def contour_pixel_to_odb_polygon(
    contour_px: Iterable[Point],
    image_center_aoi_mm: Point,
    geometry: CropGeometry,
    transform: AoiOdbTransform,
):
    """Return a Shapely Polygon when Shapely is installed.

    Shapely is imported lazily so coordinate mapping itself has no mandatory
    geometry dependency.
    """
    points = contour_pixel_to_odb_points(contour_px, image_center_aoi_mm, geometry, transform)
    if len(points) < 3:
        raise ValueError("a contour polygon requires at least 3 points")
    try:
        from shapely.geometry import Polygon
    except ImportError as exc:
        raise RuntimeError("Shapely is required to create contour polygons") from exc
    polygon = Polygon(points)
    if polygon.is_empty or polygon.area <= 0:
        raise ValueError("contour produced an empty/zero-area polygon")
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    return polygon
