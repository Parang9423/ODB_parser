#!/usr/bin/env python3
"""Geometry diagnostics for SIGNAL surfaces inside a small ODB ROI.

This module is diagnostic-only.  It does not change production rasterization.
It answers the distinction needed during AOI/CAM calibration:
- does a SIGNAL surface bounding box merely overlap the ROI?
- does the actual polygon geometry overlap the ROI?
- is the ROI centre inside the filled I/H contour geometry?
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image

from hierarchy_renderer import FastODBRenderer
from odb_cam_renderer import Transform, arc_points
from render.roi import (
    _nonzero_pixels,
    _render_layer_mask,
    roi_bounds_in,
    select_roi_layers,
)

Point = tuple[float, float]
Bounds = tuple[float, float, float, float]


def _bbox(points: Sequence[Point]) -> Bounds:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def _bbox_intersects(a: Bounds, b: Bounds) -> bool:
    return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3])


def _point_in_polygon(point: Point, polygon: Sequence[Point]) -> bool:
    if len(polygon) < 3:
        return False
    x, y = point
    inside = False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)):
            x_cross = (xj - xi) * (y - yi) / ((yj - yi) or 1e-30) + xi
            if x < x_cross:
                inside = not inside
        j = i
    return inside


def _orientation(a: Point, b: Point, c: Point) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(a: Point, b: Point, p: Point, eps: float = 1e-12) -> bool:
    return (
        min(a[0], b[0]) - eps <= p[0] <= max(a[0], b[0]) + eps
        and min(a[1], b[1]) - eps <= p[1] <= max(a[1], b[1]) + eps
        and abs(_orientation(a, b, p)) <= eps
    )


def _segments_intersect(a: Point, b: Point, c: Point, d: Point) -> bool:
    o1, o2 = _orientation(a, b, c), _orientation(a, b, d)
    o3, o4 = _orientation(c, d, a), _orientation(c, d, b)
    if ((o1 > 0 > o2) or (o1 < 0 < o2)) and ((o3 > 0 > o4) or (o3 < 0 < o4)):
        return True
    return (
        _on_segment(a, b, c) or _on_segment(a, b, d)
        or _on_segment(c, d, a) or _on_segment(c, d, b)
    )


def _polygon_rect_intersects(polygon: Sequence[Point], rect: Bounds) -> bool:
    if len(polygon) < 3 or not _bbox_intersects(_bbox(polygon), rect):
        return False
    xmin, ymin, xmax, ymax = rect
    corners = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
    if any(_point_in_polygon(corner, polygon) for corner in corners):
        return True
    if any(xmin <= x <= xmax and ymin <= y <= ymax for x, y in polygon):
        return True
    rect_edges = list(zip(corners, corners[1:] + corners[:1]))
    poly_edges = list(zip(polygon, list(polygon[1:]) + [polygon[0]]))
    return any(_segments_intersect(a, b, c, d) for a, b in poly_edges for c, d in rect_edges)


def _surface_geometry_contains(point: Point, contours: Sequence[tuple[str, Sequence[Point]]]) -> bool:
    inside_island = any(kind.upper().startswith("I") and _point_in_polygon(point, pts) for kind, pts in contours)
    inside_hole = any(kind.upper().startswith("H") and _point_in_polygon(point, pts) for kind, pts in contours)
    return inside_island and not inside_hole


def _surface_geometry_intersects(rect: Bounds, contours: Sequence[tuple[str, Sequence[Point]]]) -> bool:
    # A useful diagnostic rather than a full polygon boolean engine: report overlap
    # with an island unless the ROI centre is demonstrably inside a hole.
    islands = [pts for kind, pts in contours if kind.upper().startswith("I")]
    return any(_polygon_rect_intersects(points, rect) for points in islands)


def _parse_surfaces(records: Sequence[str], transform: Transform):
    index = 0
    while index < len(records):
        raw = records[index].strip(); index += 1
        if not raw or raw.startswith(("#", "$")):
            continue
        tokens = raw.split()
        if tokens[0] != "S" or len(tokens) < 2:
            continue
        polarity = tokens[1].upper()
        contours: list[tuple[str, list[Point]]] = []
        current: list[Point] | None = None
        current_kind = "I"
        while index < len(records):
            sr = records[index].strip(); index += 1
            if not sr or sr.startswith("#"):
                continue
            vals = sr.split(); cmd = vals[0]
            try:
                if cmd == "OB" and len(vals) >= 4:
                    current = [(float(vals[1]), float(vals[2]))]
                    current_kind = vals[3].upper()
                elif cmd == "OS" and current is not None and len(vals) >= 3:
                    current.append((float(vals[1]), float(vals[2])))
                elif cmd == "OC" and current is not None and len(vals) >= 6:
                    end = (float(vals[1]), float(vals[2]))
                    center = (float(vals[3]), float(vals[4]))
                    current.extend(arc_points(current[-1], end, center, vals[5].upper().startswith("Y")))
                elif cmd == "OE":
                    if current:
                        contours.append((current_kind, [transform.apply(p) for p in current]))
                    current = None
                elif cmd == "SE":
                    break
            except (ValueError, IndexError):
                continue
        yield polarity, contours


def validate_signal_surfaces(job: Path, center_x_mm: float, center_y_mm: float,
                             resolution_um_per_px: float, recipe_layer: str,
                             width_px: int = 100, height_px: int = 100,
                             visible_steps: Sequence[str] = ("pnl", "strip", "unit"),
                             max_surfaces: int = 30) -> dict:
    selection = select_roi_layers(job, recipe_layer)
    renderer = FastODBRenderer.from_um_per_pixel(job, resolution_um_per_px, resolution_um_per_px)
    root_step = "pnl" if (job / "steps" / "pnl").is_dir() else next(
        p.name for p in (job / "steps").iterdir() if p.is_dir()
    )
    available = {p.name.lower() for p in (job / "steps").iterdir() if p.is_dir()}
    visible = {step.lower() for step in visible_steps if step.lower() in available}
    roi = roi_bounds_in(center_x_mm, center_y_mm, resolution_um_per_px, width_px, height_px)
    center = (center_x_mm / 25.4, center_y_mm / 25.4)

    rows: list[dict] = []
    bbox_hits = geometry_hits = center_hits = 0
    for instance in renderer.collect_instances(root_step):
        if instance.step not in visible:
            continue
        feature_file = renderer._step_dir(instance.step) / "layers" / selection.signal_layer.lower() / "features"
        _, records = renderer._feature_data(feature_file)
        if not records:
            continue
        for polarity, contours in _parse_surfaces(records, instance.transform):
            points = [point for _, contour in contours for point in contour]
            if not points:
                continue
            surface_bbox = _bbox(points)
            if not _bbox_intersects(surface_bbox, roi):
                continue
            bbox_hits += 1
            actual_intersection = _surface_geometry_intersects(roi, contours)
            center_inside = _surface_geometry_contains(center, contours)
            geometry_hits += int(actual_intersection)
            center_hits += int(center_inside)
            if len(rows) < max_surfaces:
                rows.append({
                    "step": instance.step.upper(),
                    "polarity": polarity,
                    "bbox_mm": [v * 25.4 for v in surface_bbox],
                    "bbox_intersects_roi": True,
                    "actual_polygon_intersects_roi": actual_intersection,
                    "roi_center_inside_filled_geometry": center_inside,
                    "contours": [
                        {
                            "kind": kind,
                            "vertices": len(points),
                            "bbox_mm": [v * 25.4 for v in _bbox(points)] if points else None,
                        }
                        for kind, points in contours
                    ],
                })

    return {
        "signal_layer": selection.signal_layer,
        "roi_center_mm": [center_x_mm, center_y_mm],
        "roi_size_px": [width_px, height_px],
        "roi_bounds_mm": [v * 25.4 for v in roi],
        "surface_bbox_hits": bbox_hits,
        "surface_actual_polygon_hits": geometry_hits,
        "surface_center_inside_hits": center_hits,
        "surfaces": rows,
    }


def render_signal_preview(job: Path, center_x_mm: float, center_y_mm: float,
                          resolution_um_per_px: float, recipe_layer: str,
                          width_px: int = 500, height_px: int = 500,
                          signal_gv: int = 255,
                          visible_steps: Sequence[str] = ("pnl", "strip", "unit")) -> tuple[Image.Image, dict]:
    selection = select_roi_layers(job, recipe_layer)
    renderer = FastODBRenderer.from_um_per_pixel(job, resolution_um_per_px, resolution_um_per_px)
    root_step = "pnl" if (job / "steps" / "pnl").is_dir() else next(
        p.name for p in (job / "steps").iterdir() if p.is_dir()
    )
    available = {p.name.lower() for p in (job / "steps").iterdir() if p.is_dir()}
    visible = tuple(step for step in visible_steps if step.lower() in available)
    bounds = roi_bounds_in(center_x_mm, center_y_mm, resolution_um_per_px, width_px, height_px)
    mask = _render_layer_mask(renderer, root_step, selection.signal_layer, visible, bounds, width_px, height_px)
    image = Image.new("L", (width_px, height_px), 0)
    image.paste(int(signal_gv), mask=mask.point(lambda value: 255 if value else 0, mode="L"))
    return image, {
        "signal_layer": selection.signal_layer,
        "size_px": [width_px, height_px],
        "physical_size_mm": [width_px * resolution_um_per_px / 1000.0, height_px * resolution_um_per_px / 1000.0],
        "nonzero_pixels": _nonzero_pixels(image),
    }
