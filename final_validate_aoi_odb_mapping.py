#!/usr/bin/env python3
"""Final AOI -> ODB validation using the fixed SWAP_X+_Y- mapping.

This validator intentionally performs no coordinate search or local alignment.
It discovers G_ AOI images under data/GIDS, resolves each image's ERT and ODB,
computes the product-specific ERT_YX -> STRIP_ARRAY translation, maps the AOI
physical coordinate with SWAP_X+_Y-, renders the CAM ROI at that exact point,
and compares it with the matching C_ reference image.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageOps

from aoi.coordinate_validation import ImageContext, discover_images, parse_image_context, resolve_resources
from aoi.ert import parse_ert
from hierarchy_renderer import FastODBRenderer
from odb_cam_renderer import contours_bounds, extract_input
from render.roi import render_roi_cam
from validation_utils import _find_reference, _fmt_seconds, _score_crop

ORIENTATION = "SWAP_X+_Y-"
MATRIX = ((0.0, 1.0), (-1.0, 0.0))


def _apply(point: tuple[float, float]) -> tuple[float, float]:
    x, y = point
    return y, -x


def _bounds(points: Iterable[tuple[float, float]]) -> list[float]:
    pts = list(points)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return [min(xs), min(ys), max(xs), max(ys)]


def _center(bounds: list[float]) -> tuple[float, float]:
    return ((bounds[0] + bounds[2]) / 2.0, (bounds[1] + bounds[3]) / 2.0)


def _corners(bounds: list[float]) -> list[tuple[float, float]]:
    return [(bounds[0], bounds[1]), (bounds[0], bounds[3]), (bounds[2], bounds[1]), (bounds[2], bounds[3])]


def _union(bounds_list: list[list[float]]) -> list[float]:
    return [min(b[0] for b in bounds_list), min(b[1] for b in bounds_list), max(b[2] for b in bounds_list), max(b[3] for b in bounds_list)]


def _ert_yx_frame(region_values: list[float]) -> list[float]:
    if len(region_values) < 4:
        raise ValueError("ERT inspection region requires four values")
    return [float(region_values[0]), float(region_values[1]), float(region_values[3]), float(region_values[2])]


def _translation(ert_yx_bounds: list[float], strip_array_bounds: list[float]) -> tuple[float, float]:
    transformed = _bounds(_apply(p) for p in _corners(ert_yx_bounds))
    ecx, ecy = _center(transformed)
    scx, scy = _center(strip_array_bounds)
    return scx - ecx, scy - ecy


def map_aoi_to_odb(aoi_x_mm: float, aoi_y_mm: float, tx_mm: float, ty_mm: float) -> tuple[float, float]:
    x, y = _apply((aoi_x_mm, aoi_y_mm))
    return x + tx_mm, y + ty_mm


def spatial_pick(rows: list[ImageContext], count: int) -> list[ImageContext]:
    if count <= 0 or len(rows) <= count:
        return rows
    xs = [r.x_mm for r in rows]
    ys = [r.y_mm for r in rows]
    cx = (min(xs) + max(xs)) / 2.0
    cy = (min(ys) + max(ys)) / 2.0
    targets = [(min(xs), min(ys)), (max(xs), min(ys)), (min(xs), max(ys)), (max(xs), max(ys)), (cx, cy)]
    selected: list[ImageContext] = []
    remaining = list(rows)
    for tx, ty in targets:
        if not remaining or len(selected) >= count:
            break
        best = min(remaining, key=lambda r: (r.x_mm - tx) ** 2 + (r.y_mm - ty) ** 2)
        selected.append(best)
        remaining.remove(best)
    while remaining and len(selected) < count:
        best = max(remaining, key=lambda r: min((r.x_mm-s.x_mm)**2 + (r.y_mm-s.y_mm)**2 for s in selected))
        selected.append(best)
        remaining.remove(best)
    return selected


def _strip_array_bounds(renderer: FastODBRenderer) -> list[float]:
    instances = renderer.collect_instances()
    strips = [i for i in instances if str(getattr(i, "step_name", "")).upper() == "STRIP"]
    if not strips:
        raise ValueError("No STRIP instances found")
    bounds = []
    for inst in strips:
        profile = renderer.load_profile(inst.step_name)
        local = contours_bounds(profile)
        if local is None:
            continue
        pts = [inst.transform.apply(x, y) for x, y in _corners(list(local))]
        bounds.append(_bounds(pts))
    if not bounds:
        raise ValueError("STRIP profiles have no usable bounds")
    return _union(bounds)


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate fixed AOI -> ODB mapping")
    ap.add_argument("root", type=Path, nargs="?", default=Path("data/GIDS"))
    ap.add_argument("--count", type=int, default=10)
    ap.add_argument("--output", type=Path, default=Path("final_aoi_odb_validation"))
    args = ap.parse_args()

    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    contexts = [parse_image_context(p) for p in discover_images(args.root.resolve())]
    contexts = [c for c in contexts if c is not None]
    selected = spatial_pick(contexts, args.count)
    results = []
    started = time.perf_counter()

    for idx, ctx in enumerate(selected, 1):
        resources = resolve_resources(ctx)
        if not resources.ert_path or not resources.odb_path:
            continue
        ert = parse_ert(resources.ert_path)
        job, tmp = extract_input(resources.odb_path)
        try:
            renderer = FastODBRenderer(job)
            strip_bounds = _strip_array_bounds(renderer)
            ert_bounds = _ert_yx_frame(ert.region_values)
            tx, ty = _translation(ert_bounds, strip_bounds)
            odb_x, odb_y = map_aoi_to_odb(ctx.x_mm, ctx.y_mm, tx, ty)
            ref_path = _find_reference(ctx.path)
            if ref_path is None:
                continue
            with Image.open(ref_path) as im:
                reference = ImageOps.grayscale(im).copy()
            resolution = float(ert.resolution_um_per_px)
            cam = render_roi_cam(job, odb_x, odb_y, reference.width, reference.height, resolution, resources.layer)
            if isinstance(cam, tuple):
                cam = cam[0]
            scores = _score_crop(reference, cam)
            results.append({
                "sample": idx, "g_image": str(ctx.path), "reference_c": str(ref_path),
                "odb_path": str(resources.odb_path), "layer": resources.layer,
                "aoi_x_mm": ctx.x_mm, "aoi_y_mm": ctx.y_mm,
                "odb_x_mm": odb_x, "odb_y_mm": odb_y,
                "resolution_um_per_px": resolution, "translation_mm": [tx, ty],
                **scores,
            })
        finally:
            if tmp is not None:
                tmp.cleanup()

    payload = {"orientation": ORIENTATION, "matrix": MATRIX, "results": results, "elapsed_seconds": time.perf_counter() - started}
    json_path = out / "final_aoi_odb_validation.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if results:
        with (out / "final_aoi_odb_validation.csv").open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader(); writer.writerows(results)
    print(f"Validated {len(results)} samples in {_fmt_seconds(payload['elapsed_seconds'])}")
    print(json_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
