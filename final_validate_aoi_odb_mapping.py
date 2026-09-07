#!/usr/bin/env python3
"""Final AOI -> ODB validation using the fixed SWAP_X+_Y- mapping.

This validator intentionally performs no coordinate search or local alignment.
It discovers G_ AOI images under data/GIDS, resolves each image's ERT and ODB,
computes the product-specific ERT_YX -> STRIP_ARRAY translation, maps the AOI
physical coordinate with SWAP_X+_Y-, renders the CAM ROI at that exact point,
and compares it with the matching C_ reference image.

The purpose is to verify that one rigid 1:1 mm transform works at spatially
separated locations and across additional files that were not used to select the
orientation.
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

from aoi.coordinate_validation import (
    ImageContext,
    discover_images,
    parse_image_context,
    resolve_resources,
)
from aoi.ert import parse_ert
from hierarchy_renderer import FastODBRenderer
from odb_cam_renderer import contours_bounds, extract_input
from render.roi import render_roi_cam
from search_local_coordinate_match import _find_reference, _fmt_seconds, _score_crop

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
    return [
        (bounds[0], bounds[1]),
        (bounds[0], bounds[3]),
        (bounds[2], bounds[1]),
        (bounds[2], bounds[3]),
    ]


def _union(bounds_list: list[list[float]]) -> list[float]:
    return [
        min(b[0] for b in bounds_list),
        min(b[1] for b in bounds_list),
        max(b[2] for b in bounds_list),
        max(b[3] for b in bounds_list),
    ]


def _instance_bounds_mm(renderer: FastODBRenderer, instance) -> list[float]:
    return [v * 25.4 for v in contours_bounds(renderer.transformed_profile(instance))]


def _ert_yx_frame(region_values: Iterable[float]) -> list[float]:
    rv = list(map(float, region_values))
    if len(rv) < 4:
        raise ValueError("ERT region_values must contain at least four values")
    # AOI image coordinates are X=filename second coordinate, Y=filename first
    # coordinate. ERT geometry was found to describe the same physical frame with
    # its axes stored in YX order, so swap the coordinate roles here.
    return [min(rv[1], rv[3]), min(rv[0], rv[2]), max(rv[1], rv[3]), max(rv[0], rv[2])]


def _translation(frame: list[float], strip_array: list[float]) -> tuple[float, float]:
    transformed_frame = _bounds(_apply(p) for p in _corners(frame))
    fc = _center(transformed_frame)
    tc = _center(strip_array)
    return tc[0] - fc[0], tc[1] - fc[1]


def map_aoi_to_odb(
    aoi_x_mm: float,
    aoi_y_mm: float,
    translation_mm: tuple[float, float],
) -> tuple[float, float]:
    """Apply fixed SWAP_X+_Y- and a product-specific translation."""
    tx, ty = translation_mm
    return aoi_y_mm + tx, -aoi_x_mm + ty


def _normalize_context(ctx: ImageContext, extents: tuple[float, float, float, float]) -> tuple[float, float]:
    xmin, ymin, xmax, ymax = extents
    dx = max(1e-9, xmax - xmin)
    dy = max(1e-9, ymax - ymin)
    return (ctx.x_mm - xmin) / dx, (ctx.y_mm - ymin) / dy


def spatial_pick(contexts: list[ImageContext], count: int) -> list[ImageContext]:
    """Pick spatially separated samples with deterministic farthest-point sampling."""
    if count <= 0 or count >= len(contexts):
        return list(contexts)
    xmin = min(c.x_mm for c in contexts)
    xmax = max(c.x_mm for c in contexts)
    ymin = min(c.y_mm for c in contexts)
    ymax = max(c.y_mm for c in contexts)
    extents = (xmin, ymin, xmax, ymax)
    norm = [_normalize_context(c, extents) for c in contexts]

    # Start at the point nearest the upper-left of AOI space, then repeatedly
    # choose the point farthest from every already selected point.
    first = min(range(len(contexts)), key=lambda i: (norm[i][0] ** 2 + norm[i][1] ** 2, str(contexts[i].image_path)))
    selected = [first]
    remaining = set(range(len(contexts))) - {first}
    while remaining and len(selected) < count:
        def nearest_distance_sq(i: int) -> float:
            x, y = norm[i]
            return min((x - norm[j][0]) ** 2 + (y - norm[j][1]) ** 2 for j in selected)

        nxt = max(remaining, key=lambda i: (nearest_distance_sq(i), str(contexts[i].image_path)))
        selected.append(nxt)
        remaining.remove(nxt)
    return [contexts[i] for i in selected]


def _load_excluded_images(path: Path | None) -> set[str]:
    if path is None or not path.is_file():
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    excluded: set[str] = set()
    for row in payload.get("results", []):
        ic = row.get("image_context", {})
        image = ic.get("image_path")
        if image:
            excluded.add(str(Path(image).resolve()).casefold())
    return excluded


def _save_compare(reference: Image.Image, cam: Image.Image, path: Path) -> None:
    ref = ImageOps.grayscale(reference)
    cmp = ImageOps.grayscale(cam)
    h = max(ref.height, cmp.height)
    canvas = Image.new("L", (ref.width + cmp.width, h), 0)
    canvas.paste(ref, (0, 0))
    canvas.paste(cmp, (ref.width, 0))
    canvas.save(path)


def _root_step(job: Path) -> str:
    steps = {p.name.lower() for p in (job / "steps").iterdir() if p.is_dir()}
    return "pnl" if "pnl" in steps else sorted(steps)[0]


def _strip_array_bounds(job: Path) -> list[float]:
    renderer = FastODBRenderer(job, 72.0)
    root = _root_step(job)
    bounds = [
        _instance_bounds_mm(renderer, instance)
        for instance in renderer.collect_instances(root)
        if instance.step.lower() == "strip"
    ]
    if not bounds:
        raise RuntimeError("No STRIP instances found in ODB hierarchy")
    return _union(bounds)


def main() -> int:
    parser = argparse.ArgumentParser(description="Final fixed SWAP_X+_Y- AOI/ODB validation")
    parser.add_argument("root", type=Path, help="Data root containing GIDS/, ERT/, and ODB/")
    parser.add_argument("--output", type=Path, default=Path("final_aoi_odb_validation"))
    parser.add_argument("--samples", type=int, default=10, help="Number of spatially separated files to validate")
    parser.add_argument("--exclude-json", type=Path, default=None, help="Optional coordinate_validation.json whose images should be excluded")
    parser.add_argument("--item", default=None, help="Optional item-revision filter")
    parser.add_argument("--layer", default=None, help="Optional layer filter")
    args = parser.parse_args()

    root = args.root.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    excluded = _load_excluded_images(args.exclude_json.resolve() if args.exclude_json else None)

    contexts: list[ImageContext] = []
    skipped_no_reference = 0
    for image in discover_images(root):
        try:
            ctx = parse_image_context(image, root / "GIDS")
        except Exception:
            continue
        if args.item and ctx.item_revision.casefold() != args.item.casefold():
            continue
        if args.layer and ctx.layer.casefold() != args.layer.casefold():
            continue
        if str(ctx.image_path.resolve()).casefold() in excluded:
            continue
        try:
            _find_reference(ctx.image_path)
        except FileNotFoundError:
            skipped_no_reference += 1
            continue
        contexts.append(ctx)

    if not contexts:
        raise FileNotFoundError("No eligible G_ images with matching C_ references were found")

    selected = spatial_pick(contexts, min(max(1, args.samples), len(contexts)))
    started = time.perf_counter()
    rows: list[dict] = []
    failures: list[dict] = []
    geometry_cache: dict[tuple[str, tuple[float, ...]], dict] = {}

    print(f"Orientation : {ORIENTATION}")
    print(f"Eligible    : {len(contexts)}")
    print(f"Selected    : {len(selected)}")
    print(f"Excluded    : {len(excluded)}")
    print(f"No C ref    : {skipped_no_reference}\n")

    for index, ctx in enumerate(selected, 1):
        try:
            resources = resolve_resources(root, ctx)
            ert = parse_ert(resources.ert_path)
            frame = _ert_yx_frame(ert.region_values)
            cache_key = (str(resources.odb_path.resolve()).casefold(), tuple(frame))
            if cache_key not in geometry_cache:
                job, tmp = extract_input(resources.odb_path)
                try:
                    strip_array = _strip_array_bounds(job)
                finally:
                    if tmp is not None:
                        tmp.cleanup()
                geometry_cache[cache_key] = {
                    "strip_array_bounds_mm": strip_array,
                    "translation_mm": _translation(frame, strip_array),
                }
            geom = geometry_cache[cache_key]
            tx, ty = geom["translation_mm"]
            odb_x, odb_y = map_aoi_to_odb(ctx.x_mm, ctx.y_mm, (tx, ty))

            c_path = _find_reference(ctx.image_path)
            with Image.open(c_path) as im:
                reference = ImageOps.grayscale(im)
                reference.load()
                reference = reference.copy()

            job, tmp = extract_input(resources.odb_path)
            try:
                t0 = time.perf_counter()
                cam, meta = render_roi_cam(
                    job,
                    odb_x,
                    odb_y,
                    float(ert.resolution_um_per_px),
                    ctx.layer,
                    width_px=reference.width,
                    height_px=reference.height,
                    signal_gv=255,
                    drill_gv=125,
                    return_components=False,
                )
                render_seconds = time.perf_counter() - t0
            finally:
                if tmp is not None:
                    tmp.cleanup()

            score, detail = _score_crop(cam, reference)
            stem = f"S{index:02d}_{ctx.image_path.stem}"
            ref_path = out / f"{stem}_REFERENCE_C.png"
            cam_path = out / f"{stem}_PREDICTED_CAM.png"
            compare_path = out / f"{stem}_COMPARE.png"
            reference.save(ref_path)
            cam.save(cam_path)
            _save_compare(reference, cam, compare_path)

            row = {
                "sample": index,
                "g_image": str(ctx.image_path),
                "reference_c": str(c_path),
                "item_revision": ctx.item_revision,
                "layer": ctx.layer,
                "lot": ctx.lot,
                "panel": ctx.panel,
                "aoi_x_mm": ctx.x_mm,
                "aoi_y_mm": ctx.y_mm,
                "odb_x_mm": odb_x,
                "odb_y_mm": odb_y,
                "translation_x_mm": tx,
                "translation_y_mm": ty,
                "score": score,
                "edge_score": detail.get("edge_score"),
                "occupancy_score": detail.get("occupancy_score"),
                "reference_mode": detail.get("reference_mode"),
                "ert_path": str(resources.ert_path),
                "odb_path": str(resources.odb_path),
                "resolution_um_per_px": float(ert.resolution_um_per_px),
                "ert_yx_frame_mm": frame,
                "strip_array_bounds_mm": geom["strip_array_bounds_mm"],
                "reference_output": str(ref_path),
                "cam_output": str(cam_path),
                "compare_output": str(compare_path),
                "render_seconds": render_seconds,
            }
            rows.append(row)
            print(
                f"[{index:02d}/{len(selected):02d}] {ctx.image_path.name} "
                f"AOI=({ctx.x_mm:.3f},{ctx.y_mm:.3f}) -> "
                f"ODB=({odb_x:.3f},{odb_y:.3f}) score={score:.4f} "
                f"edge={float(detail.get('edge_score', 0.0)):.4f} "
                f"render={_fmt_seconds(render_seconds)}"
            )
        except Exception as exc:
            failures.append({"image": str(ctx.image_path), "error": f"{type(exc).__name__}: {exc}"})
            print(f"[{index:02d}/{len(selected):02d}] FAIL {ctx.image_path.name}: {type(exc).__name__}: {exc}")

    if not rows:
        raise RuntimeError("All selected validation samples failed")

    scores = [float(row["score"]) for row in rows]
    edge_scores = [float(row["edge_score"] or 0.0) for row in rows]
    joint = math.exp(sum(math.log(max(1e-6, s)) for s in scores) / len(scores))
    summary = {
        "algorithm": "fixed direct CAM validation; no coordinate search",
        "orientation": ORIENTATION,
        "matrix_2x2": [list(MATRIX[0]), list(MATRIX[1])],
        "mapping": "ODB_X = AOI_Y + Tx; ODB_Y = -AOI_X + Ty",
        "translation_policy": "ERT_YX transformed frame center aligned to ODB STRIP_ARRAY; calculated per ODB/ERT geometry",
        "eligible_count": len(contexts),
        "selected_count": len(selected),
        "success_count": len(rows),
        "failure_count": len(failures),
        "mean_score": sum(scores) / len(scores),
        "joint_geometric_score": joint,
        "min_score": min(scores),
        "max_score": max(scores),
        "mean_edge_score": sum(edge_scores) / len(edge_scores),
        "elapsed_seconds": time.perf_counter() - started,
    }

    csv_path = out / "final_aoi_odb_validation.csv"
    simple_keys = [
        "sample", "g_image", "reference_c", "item_revision", "layer", "lot", "panel",
        "aoi_x_mm", "aoi_y_mm", "odb_x_mm", "odb_y_mm", "translation_x_mm",
        "translation_y_mm", "score", "edge_score", "occupancy_score", "reference_mode",
        "resolution_um_per_px", "render_seconds", "compare_output",
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=simple_keys)
        writer.writeheader()
        writer.writerows({k: row.get(k) for k in simple_keys} for row in rows)

    json_path = out / "final_aoi_odb_validation.json"
    json_path.write_text(
        json.dumps({"summary": summary, "results": rows, "failures": failures}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nFINAL SUMMARY")
    print(f" orientation : {ORIENTATION}")
    print(f" success     : {len(rows)}/{len(selected)}")
    print(f" mean score  : {summary['mean_score']:.4f}")
    print(f" joint score : {summary['joint_geometric_score']:.4f}")
    print(f" min score   : {summary['min_score']:.4f}")
    print(f" mean edge   : {summary['mean_edge_score']:.4f}")
    print(f" output      : {json_path}")
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
