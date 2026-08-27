#!/usr/bin/env python3
"""Coarse-to-fine AOI(panel-mm) -> ODB coordinate search using C_ CAM references.

The filename X/Y values are treated as panel physical millimetres.  They are the
initial search centre, not pixels and not a corner-normalised coordinate.

For each G/C pair the tool:
1. reads AOI X/Y mm from coordinate_validation.json,
2. uses that panel-mm point as the initial ODB search centre,
3. performs a configurable coarse-to-fine local search,
4. renders only the native C reference size (normally 200x200),
5. keeps only REFERENCE_C.png, BEST_CAM.png and compact JSON metadata.

No SIGNAL-only, DRILL-only, mask, surface-debug or 100x100 production render is
created here.  This tool is strictly for coordinate calibration.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageFilter, ImageOps

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


def _binary_edges(image: Image.Image) -> Image.Image:
    image = ImageOps.autocontrast(ImageOps.grayscale(image))
    edge = image.filter(ImageFilter.FIND_EDGES)
    hist = edge.histogram()
    total = sum(hist)
    target = total * 0.82
    acc = 0
    threshold = 32
    for value, count in enumerate(hist):
        acc += count
        if acc >= target:
            threshold = max(16, value)
            break
    return edge.point(lambda value: 255 if value >= threshold else 0, mode="1")


def _dice(a: Image.Image, b: Image.Image) -> float:
    from PIL import ImageChops

    aa, bb = a.convert("1"), b.convert("1")
    ha, hb = aa.histogram(), bb.histogram()
    na, nb = ha[255], hb[255]
    if na + nb == 0:
        return 0.0
    overlap = ImageChops.logical_and(aa, bb).histogram()[255]
    return (2.0 * overlap) / float(na + nb)


def _score(cam: Image.Image, reference_edges: Image.Image) -> float:
    if cam.size != reference_edges.size:
        raise ValueError(f"CAM/reference size mismatch: {cam.size} != {reference_edges.size}")
    return _dice(_binary_edges(cam), reference_edges)


def _axis_offsets(radius_mm: float, step_mm: float) -> list[float]:
    if radius_mm < 0 or step_mm <= 0:
        raise ValueError("radius must be >= 0 and step must be > 0")
    count = int(math.floor(radius_mm / step_mm + 1e-9))
    values = [i * step_mm for i in range(-count, count + 1)]
    # Keep the exact radius boundary when radius is not an integer multiple of step.
    if radius_mm > 0 and (not values or abs(values[0] + radius_mm) > 1e-9):
        values.insert(0, -radius_mm)
    if radius_mm > 0 and abs(values[-1] - radius_mm) > 1e-9:
        values.append(radius_mm)
    # Stable unique values after floating-point construction.
    return sorted({round(v, 9) for v in values})


def _grid(center_x: float, center_y: float, radius_mm: float, step_mm: float) -> Iterable[tuple[float, float]]:
    offsets = _axis_offsets(radius_mm, step_mm)
    for dy in offsets:
        for dx in offsets:
            yield center_x + dx, center_y + dy


def _inside_panel(x_mm: float, y_mm: float, bounds_mm: list[float], margin_mm: float = 0.0) -> bool:
    xmin, ymin, xmax, ymax = map(float, bounds_mm)
    return (
        xmin - margin_mm <= x_mm <= xmax + margin_mm
        and ymin - margin_mm <= y_mm <= ymax + margin_mm
    )


def _parse_levels(values: list[str]) -> list[tuple[float, float]]:
    levels: list[tuple[float, float]] = []
    for raw in values:
        try:
            radius_text, step_text = raw.split(":", 1)
            radius, step = float(radius_text), float(step_text)
        except Exception as exc:
            raise ValueError(f"Invalid search level {raw!r}; expected RADIUS:STEP in mm") from exc
        if radius < 0 or step <= 0:
            raise ValueError(f"Invalid search level {raw!r}; radius>=0 and step>0 required")
        levels.append((radius, step))
    if not levels:
        raise ValueError("At least one search level is required")
    return levels


def main() -> int:
    parser = argparse.ArgumentParser(description="Search ODB coordinates around AOI panel-mm coordinates")
    parser.add_argument("validation_json", type=Path)
    parser.add_argument("--output", type=Path, default=Path("local_coordinate_search"))
    parser.add_argument("--limit", type=int, default=1, help="Number of G/C pairs to search")
    parser.add_argument(
        "--level", action="append", default=None, metavar="RADIUS:STEP",
        help="Search level in mm. Repeat for coarse-to-fine search. Default: 10:2, 2:0.5, 0.5:0.1, 0.1:0.02",
    )
    args = parser.parse_args()

    levels = _parse_levels(args.level or ["10:2", "2:0.5", "0.5:0.1", "0.1:0.02"])
    payload = json.loads(args.validation_json.resolve().read_text(encoding="utf-8"))
    details = list(payload.get("results", []))[: max(1, args.limit)]
    if not details:
        raise ValueError("coordinate_validation.json has no results")

    output_root = args.output.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    report = {
        "coordinate_assumption": "filename X/Y are panel physical millimetres and are used as the initial search centre",
        "search_levels_mm": [{"radius": r, "step": s} for r, s in levels],
        "images": [],
    }

    for image_no, detail in enumerate(details, 1):
        info, resources, ert = detail["image_context"], detail["resources"], detail["ert"]
        g_path = Path(info["image_path"])
        c_path = _find_reference(g_path)
        aoi_x, aoi_y = float(info["x_mm"]), float(info["y_mm"])
        resolution = float(ert["resolution_um_per_px"])
        panel_bounds = list(detail["pnl_bounds_mm"])

        image_dir = output_root / f"{image_no:02d}_{g_path.stem}"
        image_dir.mkdir(parents=True, exist_ok=True)
        with Image.open(c_path) as reference_source:
            reference = ImageOps.grayscale(reference_source)
            reference.load()
        reference_path = image_dir / "REFERENCE_C.png"
        reference.save(reference_path)
        reference_edges = _binary_edges(reference)

        job, temp_dir = extract_input(Path(resources["odb_path"]))
        current_x, current_y = aoi_x, aoi_y
        best_cam: Image.Image | None = None
        best_meta: dict | None = None
        best_score = -1.0
        stages: list[dict] = []
        try:
            for stage_no, (radius, step) in enumerate(levels, 1):
                stage_best_score = -1.0
                stage_best_x, stage_best_y = current_x, current_y
                stage_best_cam: Image.Image | None = None
                stage_best_meta: dict | None = None
                tested = 0
                skipped_outside_panel = 0

                for candidate_x, candidate_y in _grid(current_x, current_y, radius, step):
                    if not _inside_panel(candidate_x, candidate_y, panel_bounds):
                        skipped_outside_panel += 1
                        continue
                    tested += 1
                    cam, meta = render_roi_cam(
                        job,
                        candidate_x,
                        candidate_y,
                        resolution,
                        str(info["layer"]),
                        width_px=reference.width,
                        height_px=reference.height,
                        signal_gv=255,
                        drill_gv=125,
                        return_components=False,
                    )
                    score = _score(cam, reference_edges) if int(meta["final_nonzero_pixels"]) else 0.0
                    if score > stage_best_score:
                        stage_best_score = score
                        stage_best_x, stage_best_y = candidate_x, candidate_y
                        stage_best_cam = cam.copy()
                        stage_best_meta = dict(meta)

                if stage_best_cam is None:
                    raise RuntimeError(f"Search stage {stage_no} produced no in-panel candidates")

                current_x, current_y = stage_best_x, stage_best_y
                best_score = stage_best_score
                best_cam = stage_best_cam
                best_meta = stage_best_meta
                stages.append({
                    "stage": stage_no,
                    "radius_mm": radius,
                    "step_mm": step,
                    "search_center_input_mm": [
                        aoi_x if stage_no == 1 else stages[-1]["best_odb_mm"][0],
                        aoi_y if stage_no == 1 else stages[-1]["best_odb_mm"][1],
                    ],
                    "tested_candidates": tested,
                    "skipped_outside_panel": skipped_outside_panel,
                    "best_odb_mm": [stage_best_x, stage_best_y],
                    "best_score": round(stage_best_score, 6),
                })
        finally:
            if temp_dir is not None:
                temp_dir.cleanup()

        if best_cam is None or best_meta is None:
            raise RuntimeError("Coordinate search did not produce a result")

        best_path = image_dir / "BEST_CAM.png"
        best_cam.save(best_path)
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
            "resolution_um_per_px": resolution,
            "reference_size_px": [reference.width, reference.height],
            "score": round(best_score, 6),
            "signal_layer": best_meta["signal_layer"],
            "physical_signal_layer": best_meta["physical_signal_layer"],
            "drill_layers": list(best_meta["drill_layers_considered"]),
            "signal_nonzero": int(best_meta["signal_nonzero_pixels"]),
            "drill_nonzero": int(best_meta["drill_nonzero_pixels"]),
            "final_nonzero": int(best_meta["final_nonzero_pixels"]),
            "panel_bounds_mm": panel_bounds,
            "stages": stages,
        }
        report["images"].append(row)
        print(f"[{image_no}] {g_path.name}")
        print(f"    AOI panel mm : X={aoi_x:.6f}, Y={aoi_y:.6f}")
        print(f"    BEST ODB mm  : X={current_x:.6f}, Y={current_y:.6f}")
        print(f"    delta mm     : dX={dx:.6f}, dY={dy:.6f}, distance={math.hypot(dx, dy):.6f}")
        print(f"    score        : {best_score:.6f}")
        print(f"    layers       : signal={best_meta['signal_layer']} drill={best_meta['drill_layers_considered']}")
        print(f"    output       : {best_path}")

    report_path = output_root / "local_coordinate_search.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
