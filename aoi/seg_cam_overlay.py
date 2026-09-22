from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image, ImageDraw

Point = tuple[float, float]
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
_COORD_RE = re.compile(
    r"^(?P<prefix>[GCgc])_(?P<y>-?\d+(?:\.\d+)?)_(?P<x>-?\d+(?:\.\d+)?)"
)
_GENERATED_OUTPUT_SUFFIXES = (
    "_SEG_OVERLAY",
    "_ODB_CONTEXT",
    "_ODB_FEATURES",
    "_ODB_INTERSECTIONS",
)
_OUTPUT_REPORT_NAME = "seg_cam_overlay_report.json"


def parse_coordinate_key(path_or_name: str | Path) -> tuple[float, float]:
    """Return the first two G/C filename coordinates as a stable matching key.

    The project convention is G_<Y>_<X>_... for the AOI source image and
    C_<Y>_<X>_... for the corresponding CAM image.  The G/C prefix and the
    remaining suffix are intentionally ignored for matching.
    """
    stem = Path(path_or_name).stem
    match = _COORD_RE.match(stem)
    if match is None:
        raise ValueError(f"Unsupported G/C coordinate filename: {Path(path_or_name).name}")
    return float(match.group("y")), float(match.group("x"))


def image_role(path_or_name: str | Path) -> str | None:
    name = Path(path_or_name).name
    if not name:
        return None
    first = name[0].upper()
    if first == "G":
        return "source"
    if first == "C":
        return "cam"
    return None


def _is_within(path: Path, directory: Path) -> bool:
    """Return True when path is directory itself or one of its descendants."""
    try:
        path.resolve().relative_to(directory.resolve())
        return True
    except ValueError:
        return False


def _is_generated_output(path: Path) -> bool:
    """Return True for images produced by a previous overlay run."""
    stem_upper = path.stem.upper()
    if any(stem_upper.endswith(suffix) for suffix in _GENERATED_OUTPUT_SUFFIXES):
        return True
    # run() copies the original G image into its output folder. When a later
    # run selects a different --output, that old copy must not be rediscovered.
    if image_role(path) == "source" and (path.parent / _OUTPUT_REPORT_NAME).is_file():
        return True
    return False


def discover_gid_pairs(
    gids_dir: str | Path,
    *,
    exclude_dirs: Iterable[str | Path] = (),
) -> list[tuple[Path, Path]]:
    """Pair original G and C images recursively by their first two physical coordinates.

    Directories passed through ``exclude_dirs`` are completely excluded from
    discovery.  Generated ``*_SEG_OVERLAY`` files are also ignored as a safety
    net so previous outputs can never become new CAM inputs.
    """
    root = Path(gids_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"GIDS directory not found: {root}")

    excluded = tuple(Path(directory) for directory in exclude_dirs)
    sources: dict[tuple[float, float], list[Path]] = {}
    cams: dict[tuple[float, float], list[Path]] = {}
    for path in root.rglob("*"):
        if any(_is_within(path, directory) for directory in excluded):
            continue
        if not path.is_file() or path.suffix.lower() not in _IMAGE_EXTS:
            continue
        if _is_generated_output(path):
            continue
        role = image_role(path)
        if role is None:
            continue
        try:
            key = parse_coordinate_key(path)
        except ValueError:
            continue
        target = sources if role == "source" else cams
        target.setdefault(key, []).append(path)

    pairs: list[tuple[Path, Path]] = []
    for key in sorted(sources):
        source_rows = sorted(sources[key])
        cam_rows = sorted(cams.get(key, []))
        if len(source_rows) != 1:
            raise ValueError(f"Ambiguous G images for coordinate {key}: {source_rows}")
        if not cam_rows:
            continue
        if len(cam_rows) != 1:
            raise ValueError(f"Ambiguous C images for coordinate {key}: {cam_rows}")
        pairs.append((source_rows[0], cam_rows[0]))
    return pairs


def image_center(size: tuple[int, int]) -> Point:
    width, height = size
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    return (width - 1) / 2.0, (height - 1) / 2.0


def map_contour_by_shared_center(
    contour: Sequence[Point],
    source_size: tuple[int, int],
    cam_size: tuple[int, int],
) -> list[Point]:
    """Translate source-image contour coordinates into a larger CAM crop.

    Source and CAM crops share the same physical centre and pixel resolution.
    Therefore the contour is translated by the centre delta, not scaled by the
    ratio of image dimensions.  Example: 100x100 -> 200x200 adds (+50,+50).
    """
    source_cx, source_cy = image_center(source_size)
    cam_cx, cam_cy = image_center(cam_size)
    dx = cam_cx - source_cx
    dy = cam_cy - source_cy
    return [(float(x) + dx, float(y) + dy) for x, y in contour]


def draw_contours_on_cam(
    cam_image: Image.Image,
    contours: Iterable[Sequence[Point]],
    *,
    line_width: int = 2,
) -> Image.Image:
    """Return a copy of the CAM image with closed segmentation contours drawn."""
    if line_width <= 0:
        raise ValueError("line_width must be > 0")
    output = cam_image.convert("RGB").copy()
    draw = ImageDraw.Draw(output)
    for contour in contours:
        points = [(float(x), float(y)) for x, y in contour]
        if len(points) < 2:
            continue
        draw.line(points + [points[0]], fill=(255, 0, 0), width=line_width, joint="curve")
    return output
