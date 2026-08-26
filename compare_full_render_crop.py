#!/usr/bin/env python3
"""Compare AOI coordinates against the existing full-PNL render convention.

The old diagnostic physically allocated the complete CAM at ERT resolution and then
cropped it, which can require many GB of RAM.  This version is mathematically
identical but memory-safe:

1. Compute where AOI mm would land in the hypothetical full CAM pixel raster.
2. Convert that pixel location back to the corresponding ODB PNL physical point,
   using the same full-render origin (PNL xmin, ymax) and Y-down image convention.
3. Render only the requested 100x100 ROI around that ODB point.

Thus it tests exactly the same "AOI (0,0) == full rendered CAM top-left" hypothesis
without ever allocating the full image.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from odb_cam_renderer import extract_input
from hierarchy_renderer import FastODBRenderer
from render.roi import render_roi_cam, select_roi_layers


def first_detail(payload: dict) -> dict:
    rows = list(payload.get("results", []))
    if not rows:
        raise ValueError("coordinate_validation.json has no results")
    return rows[0]


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Emulate full CAM render + AOI mm->pixel crop without allocating the full CAM"
    )
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
    if resolution <= 0:
        raise ValueError("ERT resolution must be positive")
    mm_per_px = resolution / 1000.0
    dpi = 25400.0 / resolution

    aoi_x_mm = float(info["x_mm"])
    aoi_y_mm = float(info["y_mm"])
    x_px = aoi_x_mm / mm_per_px
    y_px = aoi_y_mm / mm_per_px

    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    job, temp_dir = extract_input(Path(resources["odb_path"]))
    try:
        selection = select_roi_layers(job, str(info["layer"]))
        renderer = FastODBRenderer.from_um_per_pixel(job, resolution, resolution)
        available = {p.name.lower() for p in (job / "steps").iterdir() if p.is_dir()}
        root = "pnl" if "pnl" in available else sorted(available)[0]
        xmin, ymin, xmax, ymax = renderer.profile_bounds(root)

        # Hypothetical complete raster dimensions, calculated only; no image is allocated.
        full_width_px = max(1, int(math.ceil((xmax - xmin) * dpi)))
        full_height_px = max(1, int(math.ceil((ymax - ymin) * dpi)))
        inside_full = 0 <= x_px < full_width_px and 0 <= y_px < full_height_px

        # Full renderer pixel convention:
        #   px_x = (odb_x - xmin) * dpi
        #   px_y = (ymax - odb_y) * dpi
        # Invert it for the AOI-derived pixel location.
        mapped_odb_x_in = xmin + x_px / dpi
        mapped_odb_y_in = ymax - y_px / dpi
        mapped_odb_x_mm = mapped_odb_x_in * 25.4
        mapped_odb_y_mm = mapped_odb_y_in * 25.4

        cam, meta = render_roi_cam(
            job=job,
            center_x_mm=mapped_odb_x_mm,
            center_y_mm=mapped_odb_y_mm,
            resolution_um_per_px=resolution,
            recipe_layer=str(info["layer"]),
            width_px=args.size,
            height_px=args.size,
            signal_gv=args.signal_gv,
            drill_gv=args.drill_gv,
        )

        stem = Path(info["image_path"]).stem
        cam_path = out / f"{stem}_FULL_RENDER_EQUIV_CAM.png"
        cam.save(cam_path, format="PNG", compress_level=1, optimize=False)

        report = {
            "source_image": info["image_path"],
            "resolution_um_per_px": resolution,
            "mm_per_px": mm_per_px,
            "dpi_equivalent": dpi,
            "aoi_coordinate_mm": [aoi_x_mm, aoi_y_mm],
            "pixel_coordinate_direct": [x_px, y_px],
            "pnl_profile_bounds_mm": [xmin * 25.4, ymin * 25.4, xmax * 25.4, ymax * 25.4],
            "hypothetical_full_render_size_px": [full_width_px, full_height_px],
            "inside_hypothetical_full_render": inside_full,
            "mapped_odb_coordinate_mm": [mapped_odb_x_mm, mapped_odb_y_mm],
            "signal_layer": selection.signal_layer,
            "drill_layers_considered": list(selection.drill_layers),
            "crop_size_px": [args.size, args.size],
            "output_cam": str(cam_path),
            "render_metadata": meta,
            "mapping_equations": {
                "x_px": "AOI_X_mm / mm_per_px",
                "y_px": "AOI_Y_mm / mm_per_px",
                "odb_x": "PNL_X_MIN + x_px / dpi",
                "odb_y": "PNL_Y_MAX - y_px / dpi",
            },
            "note": (
                "Equivalent to rendering the entire PNL at ERT resolution and cropping at "
                "(AOI_X/mm_per_px, AOI_Y/mm_per_px), but only the final ROI is allocated."
            ),
        }
        (out / "full_render_compare.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        print(f"AOI mm        : X={aoi_x_mm:.6f}, Y={aoi_y_mm:.6f}")
        print(f"Resolution    : {resolution:g} um/px ({mm_per_px:g} mm/px)")
        print(f"Direct pixel  : X={x_px:.3f}, Y={y_px:.3f}")
        print(f"Full CAM(calc): {full_width_px} x {full_height_px} px")
        print(f"Inside CAM    : {inside_full}")
        print(f"PNL bounds mm : X=({xmin*25.4:.3f},{xmax*25.4:.3f}) Y=({ymin*25.4:.3f},{ymax*25.4:.3f})")
        print(f"Mapped ODB mm : X={mapped_odb_x_mm:.6f}, Y={mapped_odb_y_mm:.6f}")
        print(f"Layer         : {selection.signal_layer}")
        print(
            "Crop nonzero  : "
            f"signal={meta.get('signal_nonzero_pixels', '?')} "
            f"drill={meta.get('drill_nonzero_pixels', '?')} "
            f"final={meta.get('final_nonzero_pixels', '?')}"
        )
        print(f"Output        : {cam_path}")
        return 0
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
