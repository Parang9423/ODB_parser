#!/usr/bin/env python3
"""Rank AOI->ODB coordinate hypotheses against existing C_ CAM reference images.

For each G_<Y>_<X>_<idx> validation item, find the matching C_ image, render a
100x100 ODB SIGNAL+DRILL ROI for several coordinate conventions, and rank the
candidates by simple structural similarity. Each candidate also saves SIGNAL-only
and DRILL-only images plus symbol/primitive diagnostics so raster interpretation
can be checked independently of coordinate matching.
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
        p = g_path.with_name(wanted_stem + ext)
        if p.is_file():
            return p
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


def _binary_edges(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    im = ImageOps.grayscale(image).resize(size, Image.Resampling.BILINEAR)
    im = ImageOps.autocontrast(im)
    edge = im.filter(ImageFilter.FIND_EDGES)
    hist = edge.histogram()
    total = sum(hist)
    target = total * 0.82
    acc = 0
    threshold = 32
    for i, n in enumerate(hist):
        acc += n
        if acc >= target:
            threshold = max(16, i)
            break
    return edge.point(lambda v: 255 if v >= threshold else 0, mode="1")


def _dice(a: Image.Image, b: Image.Image) -> float:
    aa = a.convert("1"); bb = b.convert("1")
    ha, hb = aa.histogram(), bb.histogram()
    na, nb = ha[255], hb[255]
    if na + nb == 0:
        return 0.0
    from PIL import ImageChops
    both = ImageChops.logical_and(aa, bb).histogram()[255]
    return (2.0 * both) / float(na + nb)


def _best_reference_orientation(cam: Image.Image, reference: Image.Image) -> tuple[float, str]:
    size = cam.size
    cam_edge = _binary_edges(cam, size)
    raw = _binary_edges(reference, size)
    inv = _binary_edges(ImageOps.invert(ImageOps.grayscale(reference)), size)
    s_raw, s_inv = _dice(cam_edge, raw), _dice(cam_edge, inv)
    return (s_raw, "normal") if s_raw >= s_inv else (s_inv, "inverted")


def _save_candidate_components(image_out: Path, name: str, cam: Image.Image, components: dict) -> dict:
    candidate_dir = image_out / name
    candidate_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "composite": candidate_dir / "COMPOSITE.png",
        "signal_only": candidate_dir / "SIGNAL_ONLY.png",
        "drill_only": candidate_dir / "DRILL_ONLY.png",
        "signal_mask": candidate_dir / "SIGNAL_MASK.png",
        "drill_mask": candidate_dir / "DRILL_MASK.png",
    }
    cam.save(paths["composite"])
    components["signal"].save(paths["signal_only"])
    components["drill"].save(paths["drill_only"])
    components["signal_mask"].save(paths["signal_mask"])
    components["drill_mask"].save(paths["drill_mask"])
    drill_layer_paths = {}
    for layer, mask in components.get("drill_layer_masks", {}).items():
        safe = layer.replace("/", "_").replace("\\", "_")
        path = candidate_dir / f"DRILL_{safe}_MASK.png"
        mask.save(path)
        drill_layer_paths[layer] = str(path)
    return {key: str(value) for key, value in paths.items()} | {"drill_layer_masks": drill_layer_paths}


def main() -> int:
    ap = argparse.ArgumentParser(description="Search AOI->ODB coordinate conventions using C_ CAM images as reference")
    ap.add_argument("validation_json", type=Path)
    ap.add_argument("--output", type=Path, default=Path("coordinate_search"))
    ap.add_argument("--limit", type=int, default=5, help="Number of G/C pairs to test")
    ap.add_argument("--size", type=int, default=100)
    args = ap.parse_args()

    payload = json.loads(args.validation_json.resolve().read_text(encoding="utf-8"))
    details = list(payload.get("results", []))[: max(1, args.limit)]
    if not details:
        raise ValueError("coordinate_validation.json has no results")
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    report = {"images": [], "aggregate": {}}
    aggregate: dict[str, list[float]] = {}

    for image_no, detail in enumerate(details, 1):
        info, resources, ert = detail["image_context"], detail["resources"], detail["ert"]
        g_path = Path(info["image_path"])
        c_path = _find_reference(g_path)
        bounds = list(detail["pnl_bounds_mm"])
        resolution = float(ert["resolution_um_per_px"])
        x, y = float(info["x_mm"]), float(info["y_mm"])
        image_out = out / f"{image_no:02d}_{g_path.stem}"; image_out.mkdir(parents=True, exist_ok=True)
        reference = Image.open(c_path); reference.load()
        reference.resize((args.size, args.size), Image.Resampling.BILINEAR).save(image_out / "REFERENCE_C.png")

        job, temp_dir = extract_input(Path(resources["odb_path"]))
        candidates = []
        try:
            for name, odb_x, odb_y, description in _candidate_points(x, y, bounds):
                try:
                    cam, meta, components = render_roi_cam(
                        job, odb_x, odb_y, resolution, str(info["layer"]),
                        width_px=args.size, height_px=args.size, signal_gv=255, drill_gv=125,
                        return_components=True,
                    )
                    score, ref_mode = _best_reference_orientation(cam, reference)
                    if int(meta["final_nonzero_pixels"]) == 0:
                        score = 0.0
                    output_files = _save_candidate_components(image_out, name, cam, components)
                    # Keep the old flat composite too for quick visual browsing.
                    cam.save(image_out / f"{name}.png")
                    row = {
                        "name": name, "description": description,
                        "odb_x_mm": odb_x, "odb_y_mm": odb_y,
                        "score": round(score, 6), "reference_mode": ref_mode,
                        "physical_signal_layer": int(meta["physical_signal_layer"]),
                        "signal_layer": meta["signal_layer"],
                        "drill_layers_selected": list(meta["drill_layers_considered"]),
                        "drill_layers_excluded": list(meta["drill_layers_excluded"]),
                        "drill_layers_rendered": list(meta["drill_layers_rendered"]),
                        "signal_nonzero": int(meta["signal_nonzero_pixels"]),
                        "drill_nonzero": int(meta["drill_nonzero_pixels"]),
                        "final_nonzero": int(meta["final_nonzero_pixels"]),
                        "output_files": output_files,
                        "feature_diagnostics": meta.get("feature_diagnostics", {}),
                    }
                    candidate_dir = image_out / name
                    (candidate_dir / "render_detail.json").write_text(
                        json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8"
                    )
                    candidates.append(row); aggregate.setdefault(name, []).append(score)
                except Exception as exc:
                    candidates.append({"name": name, "description": description, "odb_x_mm": odb_x, "odb_y_mm": odb_y, "error": f"{type(exc).__name__}: {exc}", "score": 0.0})
                    aggregate.setdefault(name, []).append(0.0)
        finally:
            if temp_dir is not None:
                temp_dir.cleanup()
            reference.close()
        candidates.sort(key=lambda r: float(r.get("score", 0.0)), reverse=True)
        report["images"].append({
            "g_image": str(g_path), "c_reference": str(c_path), "aoi_x_mm": x, "aoi_y_mm": y,
            "resolution_um_per_px": resolution, "ranking": candidates,
        })
        top = candidates[0]
        print(f"[{image_no}] {g_path.name}")
        print(
            f"    best={top['name']} score={top.get('score', 0):.4f} "
            f"ODB=({top.get('odb_x_mm'):.3f},{top.get('odb_y_mm'):.3f}) "
            f"signal={top.get('signal_nonzero','-')} drill={top.get('drill_nonzero','-')}"
        )
        if "signal_layer" in top:
            print(
                f"    layers: signal={top['signal_layer']} (L{top['physical_signal_layer']}) "
                f"drill={top['drill_layers_selected']} excluded={top['drill_layers_excluded']}"
            )
            signal_diag = top.get("feature_diagnostics", {}).get("signal", {})
            counts = signal_diag.get("roi_primitive_counts", {})
            print(
                f"    signal primitives: P={counts.get('pads',0)} L={counts.get('lines',0)} S={counts.get('surfaces',0)}"
            )
            drill_diags = top.get("feature_diagnostics", {}).get("drill", {})
            drill_counts = {
                layer: diag.get("roi_primitive_counts", {}) for layer, diag in drill_diags.items()
            }
            print(f"    drill primitives: {drill_counts}")

    agg_rows = []
    for name, scores in aggregate.items():
        agg_rows.append({"name": name, "mean_score": round(sum(scores) / len(scores), 6), "tested_images": len(scores), "scores": [round(s, 6) for s in scores]})
    agg_rows.sort(key=lambda r: r["mean_score"], reverse=True)
    report["aggregate"] = {"ranking": agg_rows, "best": agg_rows[0] if agg_rows else None}
    report_path = out / "coordinate_search.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if agg_rows:
        print(f"Aggregate best: {agg_rows[0]['name']} mean={agg_rows[0]['mean_score']:.4f} across {agg_rows[0]['tested_images']} image(s)")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
