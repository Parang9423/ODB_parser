#!/usr/bin/env python3
"""Parser for AOI ERT footer metadata used by the CAM alignment workflow.

The exact semantic meaning of the final geometry fields is still being
validated against the CAM Master. For that reason the last two values are
exposed as ``guide_reference_candidate`` rather than treated as a confirmed
alignment transform.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


@dataclass(frozen=True)
class ERTMetadata:
    path: Path
    region_values: Tuple[float, float, float, float]
    guide_reference_candidate: Tuple[float, float]
    item_code: str
    recipe_layer: str
    machine_or_scan_id: str
    panel_field: str
    resolution_um_per_px: float
    trailing_value: float
    timestamp: Optional[Tuple[int, int, int, int, int, int]] = None

    def roi_size_mm_for_pixels(self, width_px: int, height_px: int) -> Tuple[float, float]:
        scale_mm = self.resolution_um_per_px / 1000.0
        return width_px * scale_mm, height_px * scale_mm


def _split_csv(line: str):
    return [part.strip() for part in line.strip().split(",") if part.strip() != ""]


def _six_floats(line: str):
    values = _split_csv(line)
    if len(values) != 6:
        return None
    try:
        return tuple(float(value) for value in values)
    except ValueError:
        return None


def parse_ert(path: str | Path) -> ERTMetadata:
    path = Path(path)
    lines = [line.strip() for line in path.read_text(errors="replace").splitlines() if line.strip()]
    if len(lines) < 3:
        raise ValueError(f"ERT footer not found: {path}")

    geometry = None
    geometry_index = None
    for index in range(len(lines) - 3, max(-1, len(lines) - 12), -1):
        values = _six_floats(lines[index])
        if values is not None:
            geometry = values
            geometry_index = index
            break
    if geometry is None or geometry_index is None:
        raise ValueError(f"ERT 6-value geometry footer not found: {path}")

    if geometry_index + 1 >= len(lines):
        raise ValueError(f"ERT metadata footer not found after geometry row: {path}")
    meta = _split_csv(lines[geometry_index + 1])
    if len(meta) < 6:
        raise ValueError(f"ERT metadata footer has fewer than 6 fields: {path}")
    try:
        resolution = float(meta[4])
        trailing = float(meta[5])
    except ValueError as exc:
        raise ValueError(f"ERT resolution/footer numeric field is invalid: {path}") from exc
    if resolution <= 0:
        raise ValueError(f"ERT resolution must be positive: {resolution}")

    timestamp = None
    if geometry_index + 2 < len(lines):
        ts = _split_csv(lines[geometry_index + 2])
        if len(ts) >= 6:
            try:
                timestamp = tuple(int(value) for value in ts[:6])
            except ValueError:
                timestamp = None

    return ERTMetadata(
        path=path,
        region_values=(geometry[0], geometry[1], geometry[2], geometry[3]),
        guide_reference_candidate=(geometry[4], geometry[5]),
        item_code=meta[0],
        recipe_layer=meta[1],
        machine_or_scan_id=meta[2],
        panel_field=meta[3],
        resolution_um_per_px=resolution,
        trailing_value=trailing,
        timestamp=timestamp,
    )
