#!/usr/bin/env python3
"""Compare AOI coordinates using the existing full PNL render path.

This intentionally differs from render/roi.py: first render the complete checked-step
CAM at the ERT physical resolution, then convert AOI mm to pixel coordinates and crop.
It is a diagnostic for the hypothesis that AOI (0,0) already equals full CAM image
pixel (0,0).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageChops

from odb_cam_renderer import CompositeLayer, extract_input
from hierarchy_renderer import FastODBRenderer
from render.composite import render_selected_steps_composite
from render.roi import select_roi_layers


def nonzero(image: Image.Image) -> int:
    hist = image.histogram()
    return int(sum(hist[1:]))


def crop_centered(image: Image.Image, x_px: float, y_px: float, width: int, height: int) -> Image.Image:
    # Preserve exact requested size even at an image edge.
    left = int(round(x_px - width / 2.0))
    top = int(round(y_px - height / 2.0))
    out = Image.new("L", (width, height), 0)
    src_left, src_top = max(0, left), max(0, top)
    src_right, src_bottom = min(image.width, left + width), min(image.height, top + height)
    if src_right > src_left and src_bottom > src_top:
        region = image.crop((src_left, src_top, src_right, src_bottom))
        out.paste(region, (src_left - left, src_top - top))
    return out


def first_detail(payload: dict) -> dict:
    rows = list(payload.get("results", []))
    if not rows:
        raise ValueError("coordinate_validation.json has no results")
    return rows[0]


def main() -> int:
    ap = argparse.ArgumentParser(description="Render full CAM at ERT resolution, then crop using AOI mm->pixel conversion")
    ap.add_argument("validation_json", type=Path)
    ap.add_argument("--output", type=Path, default=Path("full_render_compare"))
    ap.add_argument("--size", type=int, default=100)
    ap.add_argument("--signal-gv", type=int, default=255)
    ap.add_argument("--drill-gv", type=int, default=125)
    args = ap.parse_args()

    payload = json.loads(args.validation_json.resolve().read_text(encoding="utf-8"))
    detail = first_detail(payload)
    info = detail["image_context"]
    resources = detail["resources"]
    ert = detail["ert"]
    resolution = float(ert["resolution_um_per_px"])
    mm_per_px = resolution / 1000.0
    aoi_x_mm = float(info["x_mm"])
    aoi_y_mm = float(info["y_mm"])
    x_px = aoi_x_mm / mm_per_px
    y_px = aoi_y_mm / mm_per_px

    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    job, temp_dir = extract_input(Path(resources["odb_path"]))
    try:
        selection = select_roi_layers(job, str(info["layer"]))
        renderer = FastODBRenderer.from_um_per_pixel(job, resolution, resolution)
        available = {p.name.lower() for p in (job / "steps").iterdir() if p.is_dir()}
        root = "pnl" if "pnl" in available else sorted(available)[0]
        visible = [s for s in ("pnl", "strip", "unit") if s in available]

        # Render signal and drill separately so the comparison report tells us which contributes.
        signal = render_selected_steps_composite(
            renderer, root, [CompositeLayer(selection.signal_layer, args.signal_gv, "REPLACE")],
            visible, include_profiles=False, background=0,
        )
        drill = Image.new("L", signal.size, 0)
        rendered_drills = []
        for layer in selection.drill_layers:
            mask = render_selected_steps_composite(
                renderer, root, [CompositeLayer(layer, 255, "REPLACE")],
                visible, include_profiles=False, background=0,
            )
            if nonzero(mask):
                drill = ImageChops.lighter(drill, mask)
                rendered_drills.append(layer)

        final = Image.new("L", signal.size, 0)
        final.paste(args.signal_gv, mask=signal.point(lambda v: 255 if v else 0, mode="L"))
        if nonzero(drill):
            final.paste(args.drill_gv, mask=drill.point(lambda v: 255 if v else 0, mode="L"))

        signal_crop = crop_centered(signal, x_px, y_px, args.size, args.size)
        drill_crop = crop_centered(drill, x_px, y_px, args.size, args.size)
        final_crop = crop_centered(final, x_px, y_px, args.size, args.size)
        stem = Path(info["image_path"]).stem
        signal_crop.save(out / f"{stem}_FULL_SIGNAL.png")
        drill_crop.save(out / f"{stem}_FULL_DRILL.png")
        final_crop.save(out / f"{stem}_FULL_CAM.png")

        report = {
            "source_image": info["image_path"],
            "resolution_um_per_px": resolution,
            "mm_per_px": mm_per_px,
            "aoi_coordinate_mm": [aoi_x_mm, aoi_y_mm],
            "pixel_coordinate_direct": [x_px, y_px],
            "full_render_size_px": list(final.size),
            "inside_full_render": 0 <= x_px < final.width and 0 <= y_px < final.height,
            "signal_layer": selection.signal_layer,
            "drill_layers_rendered": rendered_drills,
            "visible_steps": [s.upper() for s in visible],
            "crop_size_px": [args.size, args.size],
            "crop_nonzero": {
                "signal": nonzero(signal_crop),
                "drill": nonzero(drill_crop),
                "final": nonzero(final_crop),
            },
            "note": "This test deliberately assumes AOI (0,0) == full rendered CAM image top-left pixel (0,0).",
        }
        (out / "full_render_compare.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"AOI mm       : X={aoi_x_mm:.6f}, Y={aoi_y_mm:.6f}")
        print(f"Resolution   : {resolution:g} um/px ({mm_per_px:g} mm/px)")
        print(f"Direct pixel : X={x_px:.3f}, Y={y_px:.3f}")
        print(f"Full CAM     : {final.width} x {final.height} px")
        print(f"Inside CAM   : {report['inside_full_render']}")
        print(f"Layer        : {selection.signal_layer}")
        print(f"Crop nonzero : signal={report['crop_nonzero']['signal']} drill={report['crop_nonzero']['drill']} final={report['crop_nonzero']['final']}")
        print(f"Output       : {out}")
        return 0
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
