#!/usr/bin/env python3
"""Fast coarse-to-fine AOI(panel-mm) -> ODB coordinate search.

The AOI filename X/Y values are panel physical millimetres and are used directly
as the initial search centre.

Unlike the previous implementation, each stage renders one larger CAM patch and
evaluates all candidate centres by cropping that patch in memory. The final
200x200 CAM is rendered only once after the best coordinate is selected.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from PIL import Image, ImageChops, ImageFilter, ImageOps

from odb_cam_renderer import extract_input
from render.roi import render_roi_cam


def _find_reference(g_path: Path) -> Path:
    suffix = g_path.name[2:] if g_path.name[:2].upper() == "G_" else g_path.name
    wanted = "C_" + Path(suffix).stem
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
        candidate = g_path.with_name(wanted + ext)
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Matching C_ reference not found for {g_path.name}")


def _axis_offsets(radius_mm: float, step_mm: float) -> list[float]:
    if radius_mm < 0 or step_mm <= 0:
        raise ValueError("radius must be >= 0 and step must be > 0")
    count = int(math.floor(radius_mm / step_mm + 1e-9))
    values = [i * step_mm for i in range(-count, count + 1)]
    if radius_mm > 0 and (not values or abs(values[0] + radius_mm) > 1e-9):
        values.insert(0, -radius_mm)
    if radius_mm > 0 and abs(values[-1] - radius_mm) > 1e-9:
        values.append(radius_mm)
    return sorted({round(v, 9) for v in values})


def _grid(center_x: float, center_y: float, radius_mm: float, step_mm: float) -> list[tuple[float, float]]:
    offsets = _axis_offsets(radius_mm, step_mm)
    return [(center_x + dx, center_y + dy) for dy in offsets for dx in offsets]


def _inside_panel(x_mm: float, y_mm: float, bounds_mm: list[float]) -> bool:
    xmin, ymin, xmax, ymax = map(float, bounds_mm)
    return xmin <= x_mm <= xmax and ymin <= y_mm <= ymax


def _parse_levels(values: list[str]) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for raw in values:
        try:
            radius_text, step_text = raw.split(":", 1)
            radius, step = float(radius_text), float(step_text)
        except Exception as exc:
            raise ValueError(f"Invalid search level {raw!r}; expected RADIUS:STEP in mm") from exc
        if radius < 0 or step <= 0:
            raise ValueError(f"Invalid search level {raw!r}; radius>=0 and step>0 required")
        out.append((radius, step))
    if not out:
        raise ValueError("At least one search level is required")
    return out


def _fmt_seconds(value: float) -> str:
    value = max(0.0, float(value))
    if value < 60:
        return f"{value:.1f}s"
    if value < 3600:
        return f"{value / 60:.1f}m"
    return f"{value / 3600:.2f}h"


def _stage_resolution(native_um: float, step_mm: float) -> float:
    """Use coarse rasterization for wide searches, native resolution near the end."""
    return max(float(native_um), min(20.0, float(step_mm) * 20.0))


def _resize_reference(reference: Image.Image, native_um: float, stage_um: float) -> Image.Image:
    scale = float(native_um) / float(stage_um)
    width = max(8, int(round(reference.width * scale)))
    height = max(8, int(round(reference.height * scale)))
    return ImageOps.grayscale(reference).resize((width, height), Image.Resampling.BILINEAR)


def _binary(image: Image.Image, invert: bool = False) -> Image.Image:
    gray = ImageOps.autocontrast(ImageOps.grayscale(image))
    if invert:
        gray = ImageOps.invert(gray)
    return gray.point(lambda v: 255 if v >= 128 else 0, mode="L")


def _edge_mask(binary: Image.Image) -> Image.Image:
    return binary.filter(ImageFilter.FIND_EDGES).point(lambda v: 255 if v >= 32 else 0, mode="L")


def _count_on(binary: Image.Image) -> int:
    hist = binary.histogram()
    return int(hist[255]) if hist else 0


def _occupancy(binary: Image.Image) -> float:
    return _count_on(binary) / float(max(1, binary.width * binary.height))


def _tolerant_edge_dice(a: Image.Image, b: Image.Image) -> float:
    aa, bb = a.convert("L"), b.convert("L")
    na, nb = _count_on(aa), _count_on(bb)
    if na + nb == 0:
        return 0.0
    da = aa.filter(ImageFilter.MaxFilter(3))
    db = bb.filter(ImageFilter.MaxFilter(3))
    overlap_a = _count_on(ImageChops.multiply(aa, db))
    overlap_b = _count_on(ImageChops.multiply(bb, da))
    return min(1.0, (overlap_a + overlap_b) / float(na + nb))


def _score_crop(cam_crop: Image.Image, reference_stage: Image.Image) -> tuple[float, dict]:
    """Score geometry while strongly rejecting meaningless fully-solid crops."""
    cam_bin = _binary(cam_crop, invert=False)
    cam_edge = _edge_mask(cam_bin)
    cam_occ = _occupancy(cam_bin)

    best_score = -1.0
    best_detail: dict = {}
    for mode, invert in (("normal", False), ("inverted", True)):
        ref_bin = _binary(reference_stage, invert=invert)
        ref_edge = _edge_mask(ref_bin)
        ref_occ = _occupancy(ref_bin)

        occupancy_score = max(0.0, 1.0 - abs(cam_occ - ref_occ))
        edge_score = _tolerant_edge_dice(cam_edge, ref_edge)
        score = 0.35 * occupancy_score + 0.65 * edge_score

        # A crop that is essentially all copper/background cannot win merely
        # because its outer image border creates a strong FIND_EDGES response.
        cam_solid = cam_occ <= 0.005 or cam_occ >= 0.995
        ref_solid = ref_occ <= 0.005 or ref_occ >= 0.995
        if cam_solid and not ref_solid:
            score *= 0.02

        if score > best_score:
            best_score = score
            best_detail = {
                "reference_mode": mode,
                "occupancy_score": round(occupancy_score, 6),
                "edge_score": round(edge_score, 6),
                "cam_occupancy": round(cam_occ, 6),
                "reference_occupancy": round(ref_occ, 6),
                "cam_solid_rejected": bool(cam_solid and not ref_solid),
            }
    return best_score, best_detail


def _patch_geometry(radius_mm: float, reference_stage: Image.Image, resolution_um: float) -> tuple[int, int]:
    ref_w_mm = reference_stage.width * resolution_um / 1000.0
    ref_h_mm = reference_stage.height * resolution_um / 1000.0
    width_mm = 2.0 * radius_mm + ref_w_mm + 2.0 * resolution_um / 1000.0
    height_mm = 2.0 * radius_mm + ref_h_mm + 2.0 * resolution_um / 1000.0
    return (
        max(reference_stage.width, int(math.ceil(width_mm * 1000.0 / resolution_um))),
        max(reference_stage.height, int(math.ceil(height_mm * 1000.0 / resolution_um))),
    )


def _candidate_crop(
    patch: Image.Image,
    patch_center_x_mm: float,
    patch_center_y_mm: float,
    candidate_x_mm: float,
    candidate_y_mm: float,
    resolution_um: float,
    crop_size: tuple[int, int],
) -> Image.Image | None:
    """Map physical +X right / +Y up to raster +X right / +Y down."""
    px_per_mm = 1000.0 / float(resolution_um)
    cx = patch.width / 2.0 + (candidate_x_mm - patch_center_x_mm) * px_per_mm
    cy = patch.height / 2.0 - (candidate_y_mm - patch_center_y_mm) * px_per_mm
    w, h = crop_size
    left = int(round(cx - w / 2.0))
    top = int(round(cy - h / 2.0))
    right, bottom = left + w, top + h
    if left < 0 or top < 0 or right > patch.width or bottom > patch.height:
        return None
    return patch.crop((left, top, right, bottom))


def main() -> int:
    parser = argparse.ArgumentParser(description="Fast search of ODB coordinates around AOI panel-mm coordinates")
    parser.add_argument("validation_json", type=Path)
    parser.add_argument("--output", type=Path, default=Path("local_coordinate_search"))
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument(
        "--level", action="append", default=None, metavar="RADIUS:STEP",
        help="Search level in mm. Repeat for coarse-to-fine search. Default: 10:2, 2:0.5, 0.5:0.1, 0.1:0.02",
    )
    parser.add_argument("--save-search-patches", action="store_true")
    args = parser.parse_args()

    levels = _parse_levels(args.level or ["10:2", "2:0.5", "0.5:0.1", "0.1:0.02"])
    payload = json.loads(args.validation_json.resolve().read_text(encoding="utf-8"))
    details = list(payload.get("results", []))[: max(1, args.limit)]
    if not details:
        raise ValueError("coordinate_validation.json has no results")

    output_root = args.output.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = output_root / "local_coordinate_search.json"
    report = {
        "coordinate_assumption": "filename X/Y are panel physical millimetres and are used as the initial search centre",
        "algorithm": "one large CAM render per stage; candidate crops scored in memory; one final native CAM render",
        "search_levels_mm": [{"radius": r, "step": s} for r, s in levels],
        "images": [],
    }

    for image_no, detail in enumerate(details, 1):
        started = time.perf_counter()
        info, resources, ert = detail["image_context"], detail["resources"], detail["ert"]
        g_path = Path(info["image_path"])
        c_path = _find_reference(g_path)
        aoi_x, aoi_y = float(info["x_mm"]), float(info["y_mm"])
        native_um = float(ert["resolution_um_per_px"])
        panel_bounds = list(detail["pnl_bounds_mm"])

        image_dir = output_root / f"{image_no:02d}_{g_path.stem}"
        image_dir.mkdir(parents=True, exist_ok=True)
        with Image.open(c_path) as src:
            reference = ImageOps.grayscale(src)
            reference.load()
        reference_path = image_dir / "REFERENCE_C.png"
        reference.save(reference_path)

        print(f"[{image_no}] {g_path.name}", flush=True)
        print(f"    AOI panel mm : X={aoi_x:.6f}, Y={aoi_y:.6f}", flush=True)
        print(f"    reference    : {reference.width}x{reference.height}px @ {native_um:g} um/px", flush=True)

        job, temp_dir = extract_input(Path(resources["odb_path"]))
        current_x, current_y = aoi_x, aoi_y
        stages: list[dict] = []
        try:
            for stage_no, (radius, step) in enumerate(levels, 1):
                stage_started = time.perf_counter()
                stage_um = _stage_resolution(native_um, step)
                stage_ref = _resize_reference(reference, native_um, stage_um)
                patch_w, patch_h = _patch_geometry(radius, stage_ref, stage_um)
                center_in = (current_x, current_y)

                print(
                    f"    stage {stage_no}/{len(levels)}: center=({current_x:.3f},{current_y:.3f}) "
                    f"radius={radius:g}mm step={step:g}mm raster={stage_um:g}um/px "
                    f"patch={patch_w}x{patch_h}px",
                    flush=True,
                )

                patch, patch_meta = render_roi_cam(
                    job, current_x, current_y, stage_um, str(info["layer"]),
                    width_px=patch_w, height_px=patch_h,
                    signal_gv=255, drill_gv=125, return_components=False,
                )
                if args.save_search_patches:
                    patch.save(image_dir / f"STAGE_{stage_no}_SEARCH_PATCH.png")

                candidates = [p for p in _grid(current_x, current_y, radius, step)
                              if _inside_panel(p[0], p[1], panel_bounds)]
                stage_best_score = -1.0
                stage_best_xy = center_in
                stage_best_detail: dict = {}
                scored = 0
                skipped_crop = 0

                for candidate_x, candidate_y in candidates:
                    crop = _candidate_crop(
                        patch, current_x, current_y, candidate_x, candidate_y,
                        stage_um, stage_ref.size,
                    )
                    if crop is None:
                        skipped_crop += 1
                        continue
                    score, score_detail = _score_crop(crop, stage_ref)
                    scored += 1
                    if score > stage_best_score:
                        stage_best_score = score
                        stage_best_xy = (candidate_x, candidate_y)
                        stage_best_detail = score_detail

                if scored == 0:
                    raise RuntimeError(f"Stage {stage_no} produced no scoreable candidates")

                current_x, current_y = stage_best_xy
                elapsed = time.perf_counter() - stage_started
                stage_row = {
                    "stage": stage_no,
                    "radius_mm": radius,
                    "step_mm": step,
                    "raster_resolution_um_per_px": stage_um,
                    "search_center_input_mm": [center_in[0], center_in[1]],
                    "patch_size_px": [patch_w, patch_h],
                    "patch_final_nonzero": int(patch_meta["final_nonzero_pixels"]),
                    "candidate_count": len(candidates),
                    "scored_candidates": scored,
                    "skipped_crops": skipped_crop,
                    "best_odb_mm": [current_x, current_y],
                    "best_score": round(stage_best_score, 6),
                    "score_detail": stage_best_detail,
                    "elapsed_seconds": round(elapsed, 3),
                }
                stages.append(stage_row)
                print(
                    f"      best=({current_x:.6f},{current_y:.6f}) score={stage_best_score:.6f} "
                    f"occ={stage_best_detail.get('cam_occupancy')} edge={stage_best_detail.get('edge_score')} "
                    f"elapsed={_fmt_seconds(elapsed)}",
                    flush=True,
                )

                interim = {
                    "status": "running",
                    "aoi_panel_mm": [aoi_x, aoi_y],
                    "current_best_odb_mm": [current_x, current_y],
                    "completed_stages": stages,
                }
                (image_dir / "INTERIM.json").write_text(
                    json.dumps(interim, ensure_ascii=False, indent=2), encoding="utf-8"
                )

            final_cam, final_meta = render_roi_cam(
                job, current_x, current_y, native_um, str(info["layer"]),
                width_px=reference.width, height_px=reference.height,
                signal_gv=255, drill_gv=125, return_components=False,
            )
        finally:
            if temp_dir is not None:
                temp_dir.cleanup()

        final_score, final_score_detail = _score_crop(final_cam, reference)
        best_path = image_dir / "BEST_CAM.png"
        final_cam.save(best_path)
        dx, dy = current_x - aoi_x, current_y - aoi_y
        row = {
            "g_image": str(g_path),
            "c_reference": str(c_path),
            "reference_output": str(reference_path),
            "best_cam_output": str(best_path),
            "aoi_panel_mm": [aoi_x, aoi_y],
            "best_odb_mm": [current_x, current_y],
            "delta_mm": [dx, dy],
            "distance_from_aoi_mm": math.hypot(dx, dy),
            "resolution_um_per_px": native_um,
            "reference_size_px": [reference.width, reference.height],
            "final_score": round(final_score, 6),
            "final_score_detail": final_score_detail,
            "signal_layer": final_meta["signal_layer"],
            "physical_signal_layer": final_meta["physical_signal_layer"],
            "drill_layers": list(final_meta["drill_layers_considered"]),
            "signal_nonzero": int(final_meta["signal_nonzero_pixels"]),
            "drill_nonzero": int(final_meta["drill_nonzero_pixels"]),
            "final_nonzero": int(final_meta["final_nonzero_pixels"]),
            "panel_bounds_mm": panel_bounds,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "stages": stages,
        }
        report["images"].append(row)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

        print(f"    BEST ODB mm  : X={current_x:.6f}, Y={current_y:.6f}", flush=True)
        print(f"    delta mm     : dX={dx:.6f}, dY={dy:.6f}, distance={math.hypot(dx, dy):.6f}", flush=True)
        print(f"    final score  : {final_score:.6f} {final_score_detail}", flush=True)
        print(f"    total elapsed: {_fmt_seconds(row['elapsed_seconds'])}", flush=True)
        print(f"    output       : {best_path}", flush=True)

    print(f"Report: {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
