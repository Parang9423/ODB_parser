#!/usr/bin/env python3
"""Lightweight AOI -> ODB coordinate hypothesis search.

Calibration assumptions:
- G image filename coordinates provide the AOI physical centre in mm.
- C_ CAM reference images share the same centre and physical resolution as G,
  but are normally 200x200 while G is normally 100x100.
- Coordinate-search comparison must therefore render ODB directly at the native
  C reference dimensions. No image resize is used for scoring.

This script intentionally performs only the work required for coordinate search:
1. load/save the C reference image,
2. render one final SIGNAL(255)+DRILL(125) composite for each coordinate hypothesis,
3. score that final composite against C,
4. record compact layer/coordinate/render metadata.

Surface decomposition, wide previews, masks, SIGNAL-only/DRILL-only images and
100x100 production crops belong to diagnostics/production tools and are not run here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageFilter, ImageOps

from odb_cam_renderer import extract_input
from render.roi import render_roi_cam


def _find_reference(g_path: Path) -> Path:
    suffix = g_path.name[2:] if g_path.name[:2].upper() == "G_" else g_path.name
    wanted_stem = "C_" + Path(suffix).stem
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
        candidate = g_path.with_name(wanted_stem + ext)
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Matching C_ reference not found for {g_path.name}")


def _candidate_points(x: float, y: float, bounds: list[float]) -> list[tuple[str, float, float, str]]:
    xmin, ymin, xmax, ymax = map(float, bounds)
    return [
        ("DIRECT_XY", x, y, "ODB=(X,Y)"),
        ("DIRECT_YX", y, x, "ODB=(Y,X)"),
        ("LEFT_BOTTOM_XY", xmin + x, ymin + y, "origin=left-bottom, +X,+Y"),
        ("LEFT_TOP_XY", xmin + x, ymax - y, "origin=left-top, +X,-Y"),
        ("RIGHT_BOTTOM_XY", xmax - x, ymin + y, "origin=right-bottom, -X,+Y"),
        ("RIGHT_TOP_XY", xmax - x, ymax - y, "origin=right-top, -X,-Y"),
        ("LEFT_BOTTOM_YX", xmin + y, ymin + x, "swapped; origin=left-bottom"),
        ("LEFT_TOP_YX", xmin + y, ymax - x, "swapped; origin=left-top"),
        ("RIGHT_BOTTOM_YX", xmax - y, ymin + x, "swapped; origin=right-bottom"),
        ("RIGHT_TOP_YX", xmax - y, ymax - x, "swapped; origin=right-top"),
    ]


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


def _best_reference_orientation(cam: Image.Image, reference: Image.Image) -> tuple[float, str]:
    if cam.size != reference.size:
        raise ValueError(f"Physical-scale comparison requires equal pixel size: CAM={cam.size}, C={reference.size}")
    cam_edge = _binary_edges(cam)
    normal_edge = _binary_edges(reference)
    inverted_edge = _binary_edges(ImageOps.invert(ImageOps.grayscale(reference)))
    normal_score = _dice(cam_edge, normal_edge)
    inverted_score = _dice(cam_edge, inverted_edge)
    if normal_score >= inverted_score:
        return normal_score, "normal"
    return inverted_score, "inverted"


def main() -> int:
    parser = argparse.ArgumentParser(description="Lightweight AOI->ODB coordinate convention search")
    parser.add_argument("validation_json", type=Path)
    parser.add_argument("--output", type=Path, default=Path("coordinate_search"))
    parser.add_argument("--limit", type=int, default=5, help="Number of G/C pairs to test")
    args = parser.parse_args()

    payload = json.loads(args.validation_json.resolve().read_text(encoding="utf-8"))
    details = list(payload.get("results", []))[: max(1, args.limit)]
    if not details:
        raise ValueError("coordinate_validation.json has no results")

    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    report: dict = {"mode": "lightweight_coordinate_search", "images": [], "aggregate": {}}
    aggregate: dict[str, list[float]] = {}

    for image_no, detail in enumerate(details, 1):
        info = detail["image_context"]
        resources = detail["resources"]
        ert = detail["ert"]
        g_path = Path(info["image_path"])
        c_path = _find_reference(g_path)
        bounds = list(detail["pnl_bounds_mm"])
        resolution = float(ert["resolution_um_per_px"])
        x_mm, y_mm = float(info["x_mm"]), float(info["y_mm"])

        image_out = out / f"{image_no:02d}_{g_path.stem}"
        image_out.mkdir(parents=True, exist_ok=True)
        with Image.open(c_path) as reference_source:
            reference_source.load()
            reference = ImageOps.grayscale(reference_source)
        reference_path = image_out / "REFERENCE_C.png"
        reference.save(reference_path)
        reference_size = reference.size

        job, temp_dir = extract_input(Path(resources["odb_path"]))
        candidates: list[dict] = []
        try:
            for name, odb_x, odb_y, description in _candidate_points(x_mm, y_mm, bounds):
                try:
                    cam, meta = render_roi_cam(
                        job,
                        odb_x,
                        odb_y,
                        resolution,
                        str(info["layer"]),
                        width_px=reference_size[0],
                        height_px=reference_size[1],
                        signal_gv=255,
                        drill_gv=125,
                        return_components=False,
                    )
                    candidate_path = image_out / f"{name}.png"
                    cam.save(candidate_path)
                    score, reference_mode = _best_reference_orientation(cam, reference)
                    if int(meta["final_nonzero_pixels"]) == 0:
                        score = 0.0

                    row = {
                        "name": name,
                        "description": description,
                        "aoi_x_mm": x_mm,
                        "aoi_y_mm": y_mm,
                        "odb_x_mm": odb_x,
                        "odb_y_mm": odb_y,
                        "resolution_um_per_px": resolution,
                        "comparison_size_px": list(reference_size),
                        "score": round(score, 6),
                        "reference_mode": reference_mode,
                        "physical_signal_layer": int(meta["physical_signal_layer"]),
                        "signal_layer": meta["signal_layer"],
                        "drill_layers_selected": list(meta["drill_layers_considered"]),
                        "drill_layers_excluded": list(meta["drill_layers_excluded"]),
                        "drill_layers_rendered": list(meta["drill_layers_rendered"]),
                        "signal_nonzero": int(meta["signal_nonzero_pixels"]),
                        "drill_nonzero": int(meta["drill_nonzero_pixels"]),
                        "final_nonzero": int(meta["final_nonzero_pixels"]),
                        "reference_image": str(reference_path),
                        "comparison_image": str(candidate_path),
                    }
                    candidates.append(row)
                    aggregate.setdefault(name, []).append(score)
                except Exception as exc:
                    candidates.append({
                        "name": name,
                        "description": description,
                        "aoi_x_mm": x_mm,
                        "aoi_y_mm": y_mm,
                        "odb_x_mm": odb_x,
                        "odb_y_mm": odb_y,
                        "score": 0.0,
                        "error": f"{type(exc).__name__}: {exc}",
                    })
                    aggregate.setdefault(name, []).append(0.0)
        finally:
            if temp_dir is not None:
                temp_dir.cleanup()

        candidates.sort(key=lambda row: float(row.get("score", 0.0)), reverse=True)
        top = candidates[0]
        report["images"].append({
            "g_image": str(g_path),
            "c_reference": str(c_path),
            "aoi_x_mm": x_mm,
            "aoi_y_mm": y_mm,
            "resolution_um_per_px": resolution,
            "comparison_size_px": list(reference_size),
            "ranking": candidates,
        })

        print(f"[{image_no}] {g_path.name}")
        print(f"    C reference={reference_size[0]}x{reference_size[1]} px; no resize")
        print(
            f"    best={top['name']} score={top.get('score', 0):.4f} "
            f"ODB=({top.get('odb_x_mm'):.3f},{top.get('odb_y_mm'):.3f}) "
            f"signal={top.get('signal_nonzero','-')} drill={top.get('drill_nonzero','-')}"
        )
        if "signal_layer" in top:
            print(
                f"    layers: signal={top['signal_layer']} (L{top['physical_signal_layer']}) "
                f"drill={top['drill_layers_selected']}"
            )

    aggregate_rows = []
    for name, scores in aggregate.items():
        aggregate_rows.append({
            "name": name,
            "mean_score": round(sum(scores) / len(scores), 6),
            "tested_images": len(scores),
            "scores": [round(score, 6) for score in scores],
        })
    aggregate_rows.sort(key=lambda row: row["mean_score"], reverse=True)
    report["aggregate"] = {
        "ranking": aggregate_rows,
        "best": aggregate_rows[0] if aggregate_rows else None,
    }

    report_path = out / "coordinate_search.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if aggregate_rows:
        best = aggregate_rows[0]
        print(f"Aggregate best: {best['name']} mean={best['mean_score']:.4f} across {best['tested_images']} image(s)")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
