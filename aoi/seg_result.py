from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, Tuple

from aoi.contour_mapping import AoiOdbTransform, CropGeometry, contour_pixel_to_odb_polygon

Point = Tuple[float, float]
BBox = Tuple[float, float, float, float]


@dataclass(frozen=True)
class SegDetection:
    class_id: int
    class_name: str
    confidence: float
    bbox: BBox
    mask_contour: Tuple[Point, ...]

    def __post_init__(self) -> None:
        if not self.class_name:
            raise ValueError("class_name must not be empty")
        if not math.isfinite(self.confidence):
            raise ValueError("confidence must be finite")
        if len(self.bbox) != 4:
            raise ValueError("bbox must contain 4 values")
        if len(self.mask_contour) < 3:
            raise ValueError("mask_contour must contain at least 3 points")


@dataclass(frozen=True)
class SegResult:
    image: str
    width: int
    height: int
    detections: Tuple[SegDetection, ...]

    def __post_init__(self) -> None:
        if not self.image:
            raise ValueError("image must not be empty")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width/height must be > 0")

    def crop_geometry(
        self,
        resolution_um_per_px: float | Tuple[float, float],
    ) -> CropGeometry:
        if isinstance(resolution_um_per_px, tuple):
            if len(resolution_um_per_px) != 2:
                raise ValueError("resolution tuple must contain x/y values")
            return CropGeometry(
                self.width,
                self.height,
                float(resolution_um_per_px[0]),
                float(resolution_um_per_px[1]),
            )
        return CropGeometry.isotropic(self.width, self.height, float(resolution_um_per_px))


@dataclass(frozen=True)
class AoiImageFilenameInfo:
    image_name: str
    aoi_x_mm: float
    aoi_y_mm: float
    aoi_defect_class_id: int | None


_FILENAME_COORD_RE = re.compile(
    r"_(?P<y>-?\d+(?:\.\d+)?)_(?P<x>-?\d+(?:\.\d+)?)(?:_(?P<class_id>\d+))?(?:_[^.]*)?\.[^.]+$",
    re.IGNORECASE,
)


def parse_aoi_image_filename(image_name: str) -> AoiImageFilenameInfo:
    """Parse the known AOI filename suffix: ..._<Y>_<X>_<class>_SYSTEM.JPG.

    Coordinates are intentionally parsed from the right-hand suffix so product,
    lot, or unit identifiers may contain arbitrary underscores/dashes. The known
    convention is Y first, then X.
    """
    name = Path(image_name).name
    match = _FILENAME_COORD_RE.search(name)
    if not match:
        raise ValueError(f"unsupported AOI image filename format: {image_name}")
    class_id = match.group("class_id")
    return AoiImageFilenameInfo(
        image_name=name,
        aoi_x_mm=float(match.group("x")),
        aoi_y_mm=float(match.group("y")),
        aoi_defect_class_id=int(class_id) if class_id is not None else None,
    )


def _as_point(value: Sequence[Any], field: str) -> Point:
    if len(value) != 2:
        raise ValueError(f"{field} point must contain x/y")
    x, y = float(value[0]), float(value[1])
    if not math.isfinite(x) or not math.isfinite(y):
        raise ValueError(f"{field} point must be finite")
    return (x, y)


def seg_result_from_mapping(data: Mapping[str, Any]) -> SegResult:
    try:
        image = str(data["image"])
        width = int(data["width"])
        height = int(data["height"])
        raw_detections = data.get("detections", [])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid segmentation result header: {exc}") from exc

    if not isinstance(raw_detections, Sequence) or isinstance(raw_detections, (str, bytes)):
        raise ValueError("detections must be an array")

    detections = []
    for index, raw in enumerate(raw_detections):
        if not isinstance(raw, Mapping):
            raise ValueError(f"detections[{index}] must be an object")
        try:
            bbox_values = tuple(float(v) for v in raw["bbox"])
            contour_values = tuple(
                _as_point(point, f"detections[{index}].mask_contour")
                for point in raw["mask_contour"]
            )
            detection = SegDetection(
                class_id=int(raw["class_id"]),
                class_name=str(raw["class_name"]),
                confidence=float(raw["confidence"]),
                bbox=bbox_values,  # type: ignore[arg-type]
                mask_contour=contour_values,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid detections[{index}]: {exc}") from exc
        detections.append(detection)

    return SegResult(image=image, width=width, height=height, detections=tuple(detections))


def load_seg_result(path: str | Path) -> SegResult:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, Mapping):
        raise ValueError("segmentation result root must be a JSON object")
    return seg_result_from_mapping(data)


def detection_to_odb_polygon(
    result: SegResult,
    detection: SegDetection,
    image_center_aoi_mm: Point,
    resolution_um_per_px: float | Tuple[float, float],
    transform: AoiOdbTransform,
):
    """Convert one SEG detection contour into an ODB-coordinate polygon."""
    return contour_pixel_to_odb_polygon(
        contour_px=detection.mask_contour,
        image_center_aoi_mm=image_center_aoi_mm,
        geometry=result.crop_geometry(resolution_um_per_px),
        transform=transform,
    )


def iter_detection_odb_polygons(
    result: SegResult,
    image_center_aoi_mm: Point,
    resolution_um_per_px: float | Tuple[float, float],
    transform: AoiOdbTransform,
):
    """Yield (detection, polygon) pairs for every detection in one SEG result."""
    for detection in result.detections:
        yield detection, detection_to_odb_polygon(
            result,
            detection,
            image_center_aoi_mm,
            resolution_um_per_px,
            transform,
        )
