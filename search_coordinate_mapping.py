#!/usr/bin/env python3
"""Rank AOI->ODB coordinate hypotheses against existing C_ CAM reference images.

Important calibration rule:
- G images are normally 100x100.
- C reference CAM images use the same centre and physical resolution but are
  normally 200x200.

Therefore C images must never be resized to 100x100 for comparison.  Each ODB
candidate is rendered once at the requested G crop size and once at the native C
reference size.  The native-size render is used for similarity ranking.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageFilter, ImageOps

from odb_cam_renderer import extract_input
from render.roi import render_roi_cam
from render.surface_validation import render_signal_preview, validate_signal_surfaces


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


def _center_crop_or_pad(image: Image.Image, width: int, height: int) -> Image.Image:
    """Return a same-resolution crop around the exact image centre; never resize."""
    source = ImageOps.grayscale(image)
    left = int(round(source.width / 2.0 - width / 2.0))
    top = int(round(source.height / 2.0 - height / 2.0))
    out = Image.new("L", (width, height), 0)
    src_left, src_top = max(0, left), max(0, top)
    src_right, src_bottom = min(source.width, left + width), min(source.height, top + height)
    if src_right > src_left and src_bottom > src_top:
        region = source.crop((src_left, src_top, src_right, src_bottom))
        out.paste(region, (src_left - left, src_top - top))
    return out


def _binary_edges(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    im = ImageOps.grayscale(image)
    if im.size != size:
        # Used only as a guard for unusual reference sizes.  Normal calibration
        # paths render ODB directly at the native C image dimensions.
        im = im.resize(size, Image.Resampling.BILINEAR)
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
    if cam.size != reference.size:
        raise ValueError(f"Physical-scale comparison requires equal pixel size: CAM={cam.size}, C={reference.size}")
    size = cam.size
    cam_edge = _binary_edges(cam, size)
    raw = _binary_edges(reference, size)
    inv = _binary_edges(ImageOps.invert(ImageOps.grayscale(reference)), size)
    s_raw, s_inv = _dice(cam_edge, raw), _dice(cam_edge, inv)
    return (s_raw, "normal") if s_raw >= s_inv else (s_inv, "inverted")


def _save_candidate_components(image_out: Path, name: str, cam: Image.Image, components: dict) -> tuple[Path, dict]:
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
    return candidate_dir, ({key: str(value) for key, value in paths.items()} | {"drill_layer_masks": drill_layer_paths})


def _save_reference_size_components(candidate_dir: Path, cam: Image.Image, components: dict) -> dict:
    paths = {
        "composite_reference_size": candidate_dir / "COMPOSITE_REFERENCE_SIZE.png",
        "signal_reference_size": candidate_dir / "SIGNAL_REFERENCE_SIZE.png",
        "drill_reference_size": candidate_dir / "DRILL_REFERENCE_SIZE.png",
    }
    cam.save(paths["composite_reference_size"])
    components["signal"].save(paths["signal_reference_size"])
    components["drill"].save(paths["drill_reference_size"])
    return {key: str(value) for key, value in paths.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description="Search AOI->ODB coordinate conventions using native-size C CAM references")
    ap.add_argument("validation_json", type=Path)
    ap.add_argument("--output", type=Path, default=Path("coordinate_search"))
    ap.add_argument("--limit", type=int, default=5, help="Number of G/C pairs to test")
    ap.add_argument("--size", type=int, default=100, help="Output size required for G inference crop")
    ap.add_argument("--wide-signal-size", type=int, default=500, help="Diagnostic SIGNAL-only preview size")
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
        reference_gray = ImageOps.grayscale(reference)
        reference_gray.save(image_out / "REFERENCE_C.png")
        reference_center = _center_crop_or_pad(reference_gray, args.size, args.size)
        reference_center.save(image_out / f"REFERENCE_C_CENTER_{args.size}.png")
        reference_size = reference_gray.size

        job, temp_dir = extract_input(Path(resources["odb_path"]))
        candidates = []
        try:
            for name, odb_x, odb_y, description in _candidate_points(x, y, bounds):
                try:
                    # Required production/G size.
                    cam, meta, components = render_roi_cam(
                        job, odb_x, odb_y, resolution, str(info["layer"]),
                        width_px=args.size, height_px=args.size, signal_gv=255, drill_gv=125,
                        return_components=True,
                    )
                    candidate_dir, output_files = _save_candidate_components(image_out, name, cam, components)

                    # Native C reference size (normally 200x200). Same centre and same
                    # resolution means this is the only physically valid direct comparison.
                    cam_ref, meta_ref, components_ref = render_roi_cam(
                        job, odb_x, odb_y, resolution, str(info["layer"]),
                        width_px=reference_size[0], height_px=reference_size[1],
                        signal_gv=255, drill_gv=125, return_components=True,
                    )
                    output_files.update(_save_reference_size_components(candidate_dir, cam_ref, components_ref))
                    score, ref_mode = _best_reference_orientation(cam_ref, reference_gray)
                    if int(meta_ref["final_nonzero_pixels"]) == 0:
                        score = 0.0

                    surface_validation = validate_signal_surfaces(
                        job, odb_x, odb_y, resolution, str(info["layer"]),
                        width_px=args.size, height_px=args.size,
                    )
                    wide_signal, wide_meta = render_signal_preview(
                        job, odb_x, odb_y, resolution, str(info["layer"]),
                        width_px=args.wide_signal_size, height_px=args.wide_signal_size,
                    )
                    wide_path = candidate_dir / f"SIGNAL_WIDE_{args.wide_signal_size}.png"
                    wide_signal.save(wide_path)
                    output_files["signal_wide"] = str(wide_path)

                    # Keep old flat 100x100 composite for quick browsing.
                    cam.save(image_out / f"{name}.png")
                    row = {
                        "name": name, "description": description,
                        "odb_x_mm": odb_x, "odb_y_mm": odb_y,
                        "score": round(score, 6), "reference_mode": ref_mode,
                        "g_target_size_px": [args.size, args.size],
                        "c_reference_size_px": list(reference_size),
                        "comparison_render_size_px": list(cam_ref.size),
                        "comparison_preserves_physical_scale": True,
                        "physical_signal_layer": int(meta["physical_signal_layer"]),
                        "signal_layer": meta["signal_layer"],
                        "drill_layers_selected": list(meta["drill_layers_considered"]),
                        "drill_layers_excluded": list(meta["drill_layers_excluded"]),
                        "drill_layers_rendered": list(meta["drill_layers_rendered"]),
                        "signal_nonzero": int(meta["signal_nonzero_pixels"]),
                        "drill_nonzero": int(meta["drill_nonzero_pixels"]),
                        "final_nonzero": int(meta["final_nonzero_pixels"]),
                        "reference_size_signal_nonzero": int(meta_ref["signal_nonzero_pixels"]),
                        "reference_size_drill_nonzero": int(meta_ref["drill_nonzero_pixels"]),
                        "reference_size_final_nonzero": int(meta_ref["final_nonzero_pixels"]),
                        "wide_signal": wide_meta,
                        "surface_validation": surface_validation,
                        "output_files": output_files,
                        "feature_diagnostics": meta.get("feature_diagnostics", {}),
                    }
                    (candidate_dir / "render_detail.json").write_text(
                        json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8"
                    )
                    candidates.append(row); aggregate.setdefault(name, []).append(score)
                except Exception as exc:
                    candidates.append({
                        "name": name, "description": description,
                        "odb_x_mm": odb_x, "odb_y_mm": odb_y,
                        "error": f"{type(exc).__name__}: {exc}", "score": 0.0,
                    })
                    aggregate.setdefault(name, []).append(0.0)
        finally:
            if temp_dir is not None:
                temp_dir.cleanup()
            reference.close()
        candidates.sort(key=lambda r: float(r.get("score", 0.0)), reverse=True)
        report["images"].append({
            "g_image": str(g_path), "c_reference": str(c_path),
            "aoi_x_mm": x, "aoi_y_mm": y,
            "resolution_um_per_px": resolution,
            "g_target_size_px": [args.size, args.size],
            "c_reference_size_px": list(reference_size),
            "reference_handling": "native-size comparison; centre crop only for 100x100 visual check; no resize",
            "ranking": candidates,
        })
        top = candidates[0]
        print(f"[{image_no}] {g_path.name}")
        print(
            f"    C reference={reference_size[0]}x{reference_size[1]} px; G target={args.size}x{args.size} px; no resize"
        )
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
            surface = top.get("surface_validation", {})
            print(
                f"    surfaces: bbox_hits={surface.get('surface_bbox_hits',0)} "
                f"polygon_hits={surface.get('surface_actual_polygon_hits',0)} "
                f"center_inside={surface.get('surface_center_inside_hits',0)}"
            )
            print(
                f"    wide SIGNAL {args.wide_signal_size}px nonzero="
                f"{top.get('wide_signal',{}).get('nonzero_pixels','-')}"
            )

    agg_rows = []
    for name, scores in aggregate.items():
        agg_rows.append({
            "name": name,
            "mean_score": round(sum(scores) / len(scores), 6),
            "tested_images": len(scores),
            "scores": [round(s, 6) for s in scores],
        })
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
