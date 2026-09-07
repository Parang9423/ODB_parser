#!/usr/bin/env python3
"""Render a larger CAM context around one fixed AOI->ODB prediction.

This is a visual diagnostic, not a coordinate search. It reads an existing
final_aoi_odb_validation.json result, keeps its ODB coordinate unchanged, and
renders a larger physical window around that point so nearby UNIT geometry can
be inspected. The predicted point is marked at the exact image center.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from odb_cam_renderer import extract_input
from validate_hierarchy_mapping import _render_level_cam


def context_pixels(context_mm: float, resolution_um_per_px: float) -> int:
    if context_mm <= 0:
        raise ValueError("context_mm must be positive")
    if resolution_um_per_px <= 0:
        raise ValueError("resolution_um_per_px must be positive")
    return max(1, int(round(context_mm * 1000.0 / resolution_um_per_px)))


def _mark_center(image: Image.Image, text: str) -> Image.Image:
    out = ImageOps.grayscale(image).copy()
    draw = ImageDraw.Draw(out)
    cx = (out.width - 1) / 2.0
    cy = (out.height - 1) / 2.0
    arm = max(12, min(out.width, out.height) // 60)
    draw.line((cx - arm, cy, cx + arm, cy), fill=255, width=3)
    draw.line((cx, cy - arm, cx, cy + arm), fill=255, width=3)
    r = max(5, arm // 3)
    draw.ellipse((cx-r, cy-r, cx+r, cy+r), outline=255, width=2)
    font = ImageFont.load_default()
    box_w = min(max(140, len(text) * 7), max(140, out.width - 10))
    draw.rectangle((5, 5, min(out.width-1, 5+box_w), 27), fill=0, outline=255)
    draw.text((9, 9), text, fill=255, font=font)
    return out


def _thumbnail_reference(reference: Image.Image, height: int) -> Image.Image:
    ref = ImageOps.grayscale(reference)
    if ref.height == height:
        return ref.copy()
    scale = height / max(1, ref.height)
    return ref.resize((max(1, int(round(ref.width * scale))), height), Image.Resampling.NEAREST)


def _save_compare(reference: Image.Image, context: Image.Image, path: Path) -> None:
    preview_h = min(1200, max(300, context.height))
    if context.height > preview_h:
        scale = preview_h / context.height
        ctx = context.resize((max(1, int(round(context.width * scale))), preview_h), Image.Resampling.NEAREST)
    else:
        ctx = context.copy()
    ref = _thumbnail_reference(reference, ctx.height)
    gap = 16
    canvas = Image.new("L", (ref.width + gap + ctx.width, ctx.height + 22), 0)
    draw = ImageDraw.Draw(canvas)
    draw.text((3, 4), "REFERENCE", fill=255, font=ImageFont.load_default())
    draw.text((ref.width + gap + 3, 4), "LARGE UNIT CONTEXT - center = predicted ODB", fill=255, font=ImageFont.load_default())
    canvas.paste(ref, (0, 22))
    canvas.paste(ctx, (ref.width + gap, 22))
    canvas.save(path)


def main() -> int:
    ap = argparse.ArgumentParser(description="Render a large fixed-coordinate UNIT CAM context for visual residual diagnosis")
    ap.add_argument("validation_json", type=Path, help="final_aoi_odb_validation.json")
    ap.add_argument("--sample", type=int, default=3, help="Sample number from validation JSON; default 3")
    ap.add_argument("--context-mm", type=float, default=5.0, help="Square context size in mm; default 5 mm")
    ap.add_argument("--output", type=Path, default=Path("local_context_validation"))
    ap.add_argument("--level", choices=["unit", "all"], default="unit", help="Render UNIT_ONLY or ALL hierarchy")
    args = ap.parse_args()

    src = args.validation_json.resolve()
    payload = json.loads(src.read_text(encoding="utf-8"))
    row = next((r for r in payload.get("results", []) if int(r["sample"]) == args.sample), None)
    if row is None:
        raise ValueError(f"Sample {args.sample} not found in validation JSON")

    resolution = float(row["resolution_um_per_px"])
    size_px = context_pixels(args.context_mm, resolution)
    if size_px * size_px > 25_000_000:
        raise ValueError(
            f"Requested context is {size_px}x{size_px} ({size_px*size_px:,} px). "
            "Use a smaller --context-mm or coarser source resolution."
        )

    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    reference_path = Path(row["reference_c"])
    odb_path = Path(row["odb_path"])
    with Image.open(reference_path) as im:
        reference = ImageOps.grayscale(im)
        reference.load()
        reference = reference.copy()

    visible = ("unit",) if args.level == "unit" else ("pnl", "strip", "unit")
    print(f"Sample       : S{args.sample:02d}")
    print(f"ODB point    : ({float(row['odb_x_mm']):.6f}, {float(row['odb_y_mm']):.6f}) mm")
    print(f"Context      : {args.context_mm:.3f} x {args.context_mm:.3f} mm")
    print(f"Resolution   : {resolution:.3f} um/px")
    print(f"Raster       : {size_px} x {size_px} px")
    print(f"Hierarchy    : {args.level.upper()}\n")

    job, tmp = extract_input(odb_path)
    started = time.perf_counter()
    try:
        cam, meta = _render_level_cam(
            job,
            float(row["odb_x_mm"]),
            float(row["odb_y_mm"]),
            resolution,
            str(row["layer"]),
            size_px,
            size_px,
            visible,
        )
    finally:
        if tmp is not None:
            tmp.cleanup()

    label = f"S{args.sample:02d} ODB=({float(row['odb_x_mm']):.3f},{float(row['odb_y_mm']):.3f}) mm"
    marked = _mark_center(cam, label)
    stem = f"S{args.sample:02d}_{Path(row['g_image']).stem}_{args.level.upper()}_{args.context_mm:g}MM"
    raw_path = out / f"{stem}_RAW.png"
    marked_path = out / f"{stem}_MARKED.png"
    compare_path = out / f"{stem}_COMPARE.png"
    cam.save(raw_path)
    marked.save(marked_path)
    _save_compare(reference, marked, compare_path)

    result = {
        "sample": args.sample,
        "g_image": row["g_image"],
        "reference_c": str(reference_path),
        "odb_path": str(odb_path),
        "layer": row["layer"],
        "aoi_x_mm": row["aoi_x_mm"],
        "aoi_y_mm": row["aoi_y_mm"],
        "odb_x_mm": row["odb_x_mm"],
        "odb_y_mm": row["odb_y_mm"],
        "resolution_um_per_px": resolution,
        "context_mm": args.context_mm,
        "size_px": [size_px, size_px],
        "hierarchy": args.level.upper(),
        "center_pixel": [(size_px - 1) / 2.0, (size_px - 1) / 2.0],
        "context_bounds_mm": [
            float(row["odb_x_mm"]) - args.context_mm / 2.0,
            float(row["odb_y_mm"]) - args.context_mm / 2.0,
            float(row["odb_x_mm"]) + args.context_mm / 2.0,
            float(row["odb_y_mm"]) + args.context_mm / 2.0,
        ],
        "render_metadata": meta,
        "raw_output": str(raw_path),
        "marked_output": str(marked_path),
        "compare_output": str(compare_path),
        "elapsed_seconds": time.perf_counter() - started,
        "note": "No coordinate search or adjustment. Crosshair marks the original predicted ODB coordinate.",
    }
    json_path = out / f"{stem}.json"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Nonzero      : {meta['final_nonzero_pixels']:,}")
    print(f"Marked image : {marked_path}")
    print(f"Compare      : {compare_path}")
    print(f"JSON         : {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
