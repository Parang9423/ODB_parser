#!/usr/bin/env python3
"""Diagnose AOI->ODB mapping by rendering the same ROI per hierarchy level.

Reads final_aoi_odb_validation.json and reuses its fixed ODB coordinates.
No coordinate search or transform adjustment is performed.
For each sample, renders the same physical ROI four ways:
  - PNL_ONLY
  - STRIP_ONLY
  - UNIT_ONLY
  - ALL (PNL+STRIP+UNIT)
Each image uses the same SIGNAL + applicable DRILL layer selection policy as
render_roi_cam, then is scored against the original C reference.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from PIL import Image, ImageChops, ImageOps

from hierarchy_renderer import FastODBRenderer
from odb_cam_renderer import extract_input
from render.roi import _render_layer_mask, roi_bounds_in, select_roi_layers
from search_local_coordinate_match import _score_crop

LEVELS = {
    "PNL_ONLY": ("pnl",),
    "STRIP_ONLY": ("strip",),
    "UNIT_ONLY": ("unit",),
    "ALL": ("pnl", "strip", "unit"),
}


def _root_step(job: Path) -> str:
    steps = sorted(p.name.lower() for p in (job / "steps").iterdir() if p.is_dir())
    if not steps:
        raise RuntimeError("ODB has no steps")
    return "pnl" if "pnl" in steps else steps[0]


def _available_visible_steps(job: Path, requested: tuple[str, ...]) -> tuple[str, ...]:
    available = {p.name.lower() for p in (job / "steps").iterdir() if p.is_dir()}
    return tuple(step for step in requested if step in available)


def _nonzero_pixels(image: Image.Image) -> int:
    hist = image.histogram()
    return int(sum(hist[1:])) if hist else 0


def _render_level_cam(
    job: Path,
    center_x_mm: float,
    center_y_mm: float,
    resolution_um_per_px: float,
    recipe_layer: str,
    width_px: int,
    height_px: int,
    visible_steps: tuple[str, ...],
    signal_gv: int = 255,
    drill_gv: int = 125,
) -> tuple[Image.Image, dict]:
    selection = select_roi_layers(job, recipe_layer)
    renderer = FastODBRenderer.from_um_per_pixel(job, resolution_um_per_px, resolution_um_per_px)
    root = _root_step(job)
    visible = _available_visible_steps(job, visible_steps)
    bounds = roi_bounds_in(center_x_mm, center_y_mm, resolution_um_per_px, width_px, height_px)

    if visible:
        signal_mask = _render_layer_mask(renderer, root, selection.signal_layer, visible, bounds, width_px, height_px)
    else:
        signal_mask = Image.new("L", (width_px, height_px), 0)
    signal_nonzero = _nonzero_pixels(signal_mask)
    result = Image.new("L", (width_px, height_px), 0)
    if signal_nonzero:
        result.paste(signal_gv, mask=signal_mask.point(lambda v: 255 if v else 0, mode="L"))

    drill_union = Image.new("L", (width_px, height_px), 0)
    drill_nonzero_by_layer: dict[str, int] = {}
    for drill_layer in selection.drill_layers:
        if visible:
            mask = _render_layer_mask(renderer, root, drill_layer, visible, bounds, width_px, height_px)
        else:
            mask = Image.new("L", (width_px, height_px), 0)
        count = _nonzero_pixels(mask)
        drill_nonzero_by_layer[drill_layer] = count
        if count:
            drill_union = ImageChops.lighter(drill_union, mask)

    drill_nonzero = _nonzero_pixels(drill_union)
    if drill_nonzero:
        result.paste(drill_gv, mask=drill_union.point(lambda v: 255 if v else 0, mode="L"))

    return result, {
        "visible_steps": [s.upper() for s in visible],
        "signal_layer": selection.signal_layer,
        "drill_layers": list(selection.drill_layers),
        "signal_nonzero_pixels": signal_nonzero,
        "drill_nonzero_pixels": drill_nonzero,
        "drill_nonzero_by_layer": drill_nonzero_by_layer,
        "final_nonzero_pixels": _nonzero_pixels(result),
        "renderer_stats": {
            "pads": renderer.stats.pads,
            "lines": renderer.stats.lines,
            "surfaces": renderer.stats.surfaces,
            "repeats": renderer.stats.repeats,
            "unsupported": renderer.stats.unsupported,
        },
    }


def _save_compare(reference: Image.Image, renders: list[tuple[str, Image.Image]], path: Path) -> None:
    ref = ImageOps.grayscale(reference)
    images = [("REFERENCE", ref)] + [(name, ImageOps.grayscale(img)) for name, img in renders]
    label_h = 18
    widths = [img.width for _, img in images]
    heights = [img.height for _, img in images]
    canvas = Image.new("L", (sum(widths), max(heights) + label_h), 0)
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(canvas)
    x = 0
    font = ImageFont.load_default()
    for (name, img), w in zip(images, widths):
        canvas.paste(img, (x, label_h))
        draw.text((x + 3, 3), name, fill=255, font=font)
        x += w
    canvas.save(path)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate fixed AOI->ODB coordinates by hierarchy-level ROI decomposition")
    ap.add_argument("validation_json", type=Path, help="final_aoi_odb_validation.json")
    ap.add_argument("--output", type=Path, default=Path("hierarchy_mapping_validation"))
    ap.add_argument("--samples", default="all", help="Comma-separated sample numbers, e.g. 1,4,5,8; default all")
    args = ap.parse_args()

    src = args.validation_json.resolve()
    payload = json.loads(src.read_text(encoding="utf-8"))
    rows = list(payload.get("results", []))
    if not rows:
        raise ValueError("Validation JSON has no results")

    if str(args.samples).strip().lower() != "all":
        wanted = {int(v.strip()) for v in str(args.samples).split(",") if v.strip()}
        rows = [row for row in rows if int(row["sample"]) in wanted]
    if not rows:
        raise ValueError("No requested samples found")

    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    results: list[dict] = []
    failures: list[dict] = []

    print("Hierarchy decomposition validation")
    print("Coordinate transform/search: disabled")
    print(f"Samples: {len(rows)}\n")

    for idx, row in enumerate(rows, 1):
        sample = int(row["sample"])
        try:
            reference_path = Path(row["reference_c"])
            odb_path = Path(row["odb_path"])
            recipe_layer = str(row["layer"])
            odb_x = float(row["odb_x_mm"])
            odb_y = float(row["odb_y_mm"])
            resolution = float(row["resolution_um_per_px"])

            with Image.open(reference_path) as im:
                reference = ImageOps.grayscale(im)
                reference.load()
                reference = reference.copy()

            job, tmp = extract_input(odb_path)
            try:
                level_rows: dict[str, dict] = {}
                rendered: list[tuple[str, Image.Image]] = []
                for level_name, visible_steps in LEVELS.items():
                    cam, meta = _render_level_cam(
                        job, odb_x, odb_y, resolution, recipe_layer,
                        reference.width, reference.height, visible_steps,
                    )
                    score, detail = _score_crop(cam, reference)
                    image_path = out / f"S{sample:02d}_{Path(row['g_image']).stem}_{level_name}.png"
                    cam.save(image_path)
                    rendered.append((level_name, cam))
                    level_rows[level_name] = {
                        "score": float(score),
                        "edge_score": float(detail.get("edge_score") or 0.0),
                        "occupancy_score": float(detail.get("occupancy_score") or 0.0),
                        "reference_mode": detail.get("reference_mode"),
                        "output": str(image_path),
                        **meta,
                    }
            finally:
                if tmp is not None:
                    tmp.cleanup()

            compare_path = out / f"S{sample:02d}_{Path(row['g_image']).stem}_HIERARCHY_COMPARE.png"
            _save_compare(reference, rendered, compare_path)
            best_level = max(level_rows, key=lambda name: level_rows[name]["score"])
            result = {
                "sample": sample,
                "g_image": row["g_image"],
                "reference_c": str(reference_path),
                "odb_path": str(odb_path),
                "layer": recipe_layer,
                "aoi_x_mm": row["aoi_x_mm"],
                "aoi_y_mm": row["aoi_y_mm"],
                "odb_x_mm": odb_x,
                "odb_y_mm": odb_y,
                "original_score": row.get("score"),
                "best_level": best_level,
                "best_score": level_rows[best_level]["score"],
                "levels": level_rows,
                "compare_output": str(compare_path),
            }
            results.append(result)
            scores = " ".join(f"{name}={level_rows[name]['score']:.4f}" for name in LEVELS)
            print(f"[{idx:02d}/{len(rows):02d}] S{sample:02d} {scores} best={best_level}")
        except Exception as exc:
            failures.append({"sample": sample, "g_image": row.get("g_image"), "error": f"{type(exc).__name__}: {exc}"})
            print(f"[{idx:02d}/{len(rows):02d}] S{sample:02d} FAIL {type(exc).__name__}: {exc}")

    summary_by_level = {}
    for level in LEVELS:
        vals = [float(r["levels"][level]["score"]) for r in results]
        nonzero = [int(r["levels"][level]["final_nonzero_pixels"]) for r in results]
        summary_by_level[level] = {
            "mean_score": _mean(vals),
            "max_score": max(vals) if vals else 0.0,
            "min_score": min(vals) if vals else 0.0,
            "nonblank_count": sum(v > 0 for v in nonzero),
        }

    output_payload = {
        "summary": {
            "algorithm": "hierarchy decomposition at fixed AOI->ODB coordinates; no search",
            "source_validation_json": str(src),
            "orientation": payload.get("summary", {}).get("orientation"),
            "mapping": payload.get("summary", {}).get("mapping"),
            "sample_count": len(rows),
            "success_count": len(results),
            "failure_count": len(failures),
            "levels": summary_by_level,
            "elapsed_seconds": time.perf_counter() - started,
        },
        "results": results,
        "failures": failures,
    }
    json_path = out / "hierarchy_mapping_validation.json"
    json_path.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nOutput: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
