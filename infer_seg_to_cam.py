#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from PIL import Image

from app_core import inspect_job
from aoi.contour_mapping import AoiOdbTransform, CropGeometry, contour_pixel_to_odb_polygon
from aoi.odb_cam_feature_overlay import draw_odb_feature_overlay, geometry_to_cam_pixels
from aoi.seg_cam_overlay import discover_gid_pairs, draw_contours_on_cam, map_contour_by_shared_center, parse_coordinate_key
from hierarchy_renderer import FastODBRenderer
from odb.feature_query import DefectContourQuery, extract_vector_features
from odb_cam_renderer import extract_input
from render.roi import _feature_diagnostics, select_roi_layers


def _resolve_model(model_arg: Path | None, models_dir: Path) -> Path:
    if model_arg is not None:
        model = model_arg.expanduser().resolve()
        if not model.is_file():
            raise FileNotFoundError(f"Model file not found: {model}")
        return model
    candidates = sorted(models_dir.glob("*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No .pt model found in {models_dir}. Put one YOLOv8 SEG model in models/ or pass --model explicitly.")
    if len(candidates) > 1:
        raise ValueError(f"Multiple .pt models found in {models_dir}: {candidates}. Pass --model explicitly.")
    return candidates[0].resolve()


def _load_yolo(model_path: Path):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Ultralytics is required. Install it with: pip install ultralytics") from exc
    return YOLO(str(model_path))


def _to_python_contour(segment: Any) -> list[tuple[float, float]]:
    if hasattr(segment, "tolist"):
        segment = segment.tolist()
    return [(float(row[0]), float(row[1])) for row in segment if len(row) >= 2]


def _class_name(names: Any, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def _select_pairs(pairs, *, image=None, limit=None):
    selected = pairs
    if image:
        requested = Path(image).name.casefold()
        selected = [pair for pair in selected if pair[0].name.casefold() == requested]
        if not selected:
            raise ValueError(f"Requested G image was not found among matched G/C pairs: {image}")
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        selected = selected[:limit]
    return selected


def _prepare_odb(odb_input: Path, recipe_layer: str, root_step: str):
    job, temp_dir = extract_input(odb_input)
    selection = select_roi_layers(job, recipe_layer)
    renderer = FastODBRenderer(job, dpi=100.0)
    features = extract_vector_features(renderer, root_step, [selection.signal_layer], positive_only=False)
    return temp_dir, job, renderer, selection.signal_layer, features, DefectContourQuery(features)


def _safe_layer_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(name))


def _primitive_total(diag: dict) -> int:
    counts = diag.get("roi_primitive_counts", {})
    return int(counts.get("pads", 0)) + int(counts.get("lines", 0)) + int(counts.get("surfaces", 0))


def _context_layer_sweep(
    *,
    job: Path,
    renderer: FastODBRenderer,
    root_step: str,
    cam_fov_polygon,
    center_aoi_mm,
    cam_size,
    resolution_um_per_px: float,
    transform: AoiOdbTransform,
    cam_image: Image.Image,
    seg_contours,
    output_dir: Path,
    cam_stem: str,
    line_width: int,
):
    """Inspect every ODB matrix layer at the fixed CAM FOV.

    This is diagnostic/visualization only. It never changes the signal-layer
    DefectContourQuery used for exact defect/spec intersections.
    """
    info = inspect_job(job)
    available_steps = tuple(
        p.name.lower() for p in (job / "steps").iterdir() if p.is_dir()
    )
    bounds_in = tuple(float(v) / 25.4 for v in cam_fov_polygon.bounds)

    rows = []
    combined: dict[str, tuple[str, object]] = {}
    fov_features = []
    for layer in info.layers:
        diag = _feature_diagnostics(
            renderer,
            root_step,
            layer.name,
            available_steps,
            bounds_in,
            resolution_um_per_px,
            max_samples=0,
        )
        primitive_count = _primitive_total(diag)
        row = {
            "name": layer.name,
            "type": layer.layer_type,
            "context": layer.context,
            "side": layer.side,
            "polarity": layer.polarity,
            "prefilter_primitive_count": primitive_count,
            "fov_feature_count": 0,
            "overlay_image": None,
        }
        if primitive_count > 0:
            layer_features = extract_vector_features(
                renderer, root_step, [layer.name], positive_only=False
            )
            layer_hits = DefectContourQuery(layer_features).query(cam_fov_polygon)
            row["fov_feature_count"] = len(layer_hits)
            fov_features.extend(hit.feature for hit in layer_hits)

            layer_overlay_features: dict[str, tuple[str, object]] = {}
            for hit in layer_hits:
                feature_cam = geometry_to_cam_pixels(
                    hit.feature.geometry,
                    image_center_aoi_mm=center_aoi_mm,
                    cam_size_px=cam_size,
                    resolution_um_per_px=resolution_um_per_px,
                    transform=transform,
                )
                key = f"{layer.name}:{hit.feature.feature_id}"
                layer_overlay_features[key] = (hit.feature.primitive_type, feature_cam)
                combined[key] = (hit.feature.primitive_type, feature_cam)

            if layer_overlay_features:
                layer_overlay = draw_odb_feature_overlay(
                    cam_image,
                    seg_contours_px=seg_contours,
                    feature_geometries=layer_overlay_features.values(),
                    line_width=line_width,
                )
                layer_path = output_dir / f"{cam_stem}_ODB_CONTEXT_{_safe_layer_name(layer.name)}.png"
                layer_overlay.save(layer_path, format="PNG")
                row["overlay_image"] = str(layer_path)
        rows.append(row)

    return rows, combined, fov_features


def _feature_hit_row(hit) -> dict:
    """Serialize one exact feature/SEG-contour positive-area intersection."""
    return {
        "feature_id": hit.feature.feature_id,
        "layer": hit.feature.layer,
        "step": hit.feature.step,
        "depth": hit.feature.depth,
        "primitive_type": hit.feature.primitive_type,
        "symbol": hit.feature.symbol,
        "polarity": hit.feature.polarity,
        "feature_bounds_mm": [float(v) for v in hit.feature.bounds],
        "intersection_area_mm2": float(hit.intersection_area),
        # Percentage denominator is the SEG contour area, not the feature area.
        "contour_overlap_pct": float(hit.defect_overlap_pct),
    }


def _cam_fov_odb_polygon(cam_size, center_aoi_mm, resolution_um_per_px, transform):
    """Return the CAM field-of-view polygon in ODB coordinates."""
    geometry = CropGeometry.isotropic(cam_size[0], cam_size[1], resolution_um_per_px)
    contour = [
        (0.0, 0.0),
        (float(cam_size[0] - 1), 0.0),
        (float(cam_size[0] - 1), float(cam_size[1] - 1)),
        (0.0, float(cam_size[1] - 1)),
    ]
    return contour_pixel_to_odb_polygon(contour, center_aoi_mm, geometry, transform)


def run(
    gids_dir: Path,
    output_dir: Path,
    model_path: Path,
    *,
    conf: float = 0.25,
    device: str | None = None,
    line_width: int = 2,
    image: str | None = None,
    limit: int | None = None,
    odb_input: Path | None = None,
    recipe_layer: str | None = None,
    resolution_um_per_px: float | None = None,
    tx_mm: float | None = None,
    ty_mm: float | None = None,
    root_step: str = "pnl",
    context_all_layers: bool = False,
) -> dict:
    if not 0.0 <= conf <= 1.0:
        raise ValueError("conf must be between 0 and 1")

    odb_enabled = odb_input is not None
    if odb_enabled:
        if not recipe_layer:
            raise ValueError("--recipe-layer is required with --odb-input")
        if resolution_um_per_px is None or resolution_um_per_px <= 0:
            raise ValueError("--resolution-um-per-px > 0 is required with --odb-input")
        if tx_mm is None or ty_mm is None:
            raise ValueError("--tx-mm and --ty-mm are required with --odb-input")

    discovered_pairs = discover_gid_pairs(gids_dir, exclude_dirs=(output_dir,))
    if not discovered_pairs:
        raise ValueError(f"No matching G/C image pairs found under {gids_dir}")
    pairs = _select_pairs(discovered_pairs, image=image, limit=limit)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = _load_yolo(model_path)

    temp_dir = None
    job = None
    renderer = None
    signal_layer = None
    features = []
    query = None
    if odb_enabled:
        temp_dir, job, renderer, signal_layer, features, query = _prepare_odb(odb_input, recipe_layer, root_step)
        print(f"ODB signal layer: {signal_layer}")
        print(f"ODB features    : {len(features)}")

    items: list[dict] = []
    try:
        for source_path, cam_path in pairs:
            with Image.open(source_path) as src_im:
                source_size = src_im.size
            with Image.open(cam_path) as cam_im:
                cam_size = cam_im.size
                cam_copy = cam_im.copy()

            source_output_path = output_dir / source_path.name
            shutil.copy2(source_path, source_output_path)

            kwargs = {"source": str(source_path), "conf": conf, "verbose": False}
            if device:
                kwargs["device"] = device
            results = model.predict(**kwargs)
            if not results:
                raise RuntimeError(f"YOLO returned no result object for {source_path}")
            result = results[0]

            cam_contours = []
            detections = []
            defect_polygons = []
            overlay_features: dict[str, tuple[str, object]] = {}
            overlay_intersections = []
            context_features: dict[str, tuple[str, object]] = {}
            context_feature_count = 0
            context_layer_rows = []
            all_layer_overlay_features: dict[str, tuple[str, object]] = {}
            all_layer_overlay_intersections = []

            masks = getattr(result, "masks", None)
            segments = [] if masks is None else list(masks.xy)
            boxes = getattr(result, "boxes", None)
            cls_values = [] if boxes is None else boxes.cls.detach().cpu().tolist()
            conf_values = [] if boxes is None else boxes.conf.detach().cpu().tolist()
            names = getattr(result, "names", {})

            # Filename convention is G_<AOI_Y>_<AOI_X>_...
            aoi_y_mm, aoi_x_mm = parse_coordinate_key(source_path)
            center_aoi_mm = (aoi_x_mm, aoi_y_mm)
            aoi_odb_transform = None
            crop_geometry = None
            if odb_enabled:
                aoi_odb_transform = AoiOdbTransform(tx_mm=tx_mm, ty_mm=ty_mm)
                crop_geometry = CropGeometry.isotropic(source_size[0], source_size[1], resolution_um_per_px)

                # Visualization-only context query. Every ODB feature intersecting
                # the CAM FOV is drawn so coordinate registration can be inspected
                # even when the SEG defect lies entirely in empty/SPACE geometry.
                # This context set is deliberately NOT used for defect/spec logic.
                cam_fov_polygon = _cam_fov_odb_polygon(
                    cam_size, center_aoi_mm, resolution_um_per_px, aoi_odb_transform
                )
                if not context_all_layers:
                    context_hits = query.query(cam_fov_polygon)
                    context_feature_count = len(context_hits)
                    for context_hit in context_hits:
                        context_cam = geometry_to_cam_pixels(
                            context_hit.feature.geometry,
                            image_center_aoi_mm=center_aoi_mm,
                            cam_size_px=cam_size,
                            resolution_um_per_px=resolution_um_per_px,
                            transform=aoi_odb_transform,
                        )
                        context_features[context_hit.feature.feature_id] = (
                            context_hit.feature.primitive_type, context_cam
                        )

            for index, segment in enumerate(segments):
                source_contour = _to_python_contour(segment)
                if len(source_contour) < 3:
                    continue
                cam_contour = map_contour_by_shared_center(source_contour, source_size, cam_size)
                cam_contours.append(cam_contour)
                class_id = int(cls_values[index]) if index < len(cls_values) else -1
                confidence = float(conf_values[index]) if index < len(conf_values) else None

                row = {
                    "index": index,
                    "class_id": class_id,
                    "class_name": _class_name(names, class_id),
                    "confidence": confidence,
                    "source_contour_px": [[x, y] for x, y in source_contour],
                    "cam_contour_px": [[x, y] for x, y in cam_contour],
                }

                if odb_enabled:
                    defect_polygon = contour_pixel_to_odb_polygon(source_contour, center_aoi_mm, crop_geometry, aoi_odb_transform)
                    hits = query.query(defect_polygon)
                    hit_rows = []
                    for hit in sorted(hits, key=lambda h: h.defect_overlap_pct, reverse=True):
                        feature_cam = geometry_to_cam_pixels(
                            hit.feature.geometry,
                            image_center_aoi_mm=center_aoi_mm,
                            cam_size_px=cam_size,
                            resolution_um_per_px=resolution_um_per_px,
                            transform=aoi_odb_transform,
                        )
                        intersection_cam = geometry_to_cam_pixels(
                            hit.intersection_geometry,
                            image_center_aoi_mm=center_aoi_mm,
                            cam_size_px=cam_size,
                            resolution_um_per_px=resolution_um_per_px,
                            transform=aoi_odb_transform,
                        )
                        overlay_features[hit.feature.feature_id] = (hit.feature.primitive_type, feature_cam)
                        overlay_intersections.append(intersection_cam)
                        hit_row = _feature_hit_row(hit)
                        # Backward-compatible alias for existing consumers.
                        hit_row["defect_overlap_pct"] = hit_row["contour_overlap_pct"]
                        hit_rows.append(hit_row)
                    defect_polygons.append(defect_polygon)
                    row["odb"] = {
                        "defect_polygon_bounds_mm": [float(v) for v in defect_polygon.bounds],
                        "defect_area_mm2": float(defect_polygon.area),
                        "hit_count": len(hits),
                        "features": hit_rows,
                    }
                detections.append(row)

            # Run the broad layer diagnostic only after SEG contours are known so
            # every per-layer context image contains the same red SEG overlay.
            # Exact defect intersections above intentionally remain signal-layer only.
            if odb_enabled and context_all_layers:
                print("ODB context layer sweep: prefiltering all matrix layers at fixed CAM FOV...", flush=True)
                context_layer_rows, context_features, all_fov_features = _context_layer_sweep(
                    job=job,
                    renderer=renderer,
                    root_step=root_step,
                    cam_fov_polygon=cam_fov_polygon,
                    center_aoi_mm=center_aoi_mm,
                    cam_size=cam_size,
                    resolution_um_per_px=resolution_um_per_px,
                    transform=aoi_odb_transform,
                    cam_image=cam_copy,
                    seg_contours=cam_contours,
                    output_dir=output_dir,
                    cam_stem=cam_path.stem,
                    line_width=line_width,
                )
                context_feature_count = sum(row["fov_feature_count"] for row in context_layer_rows)

                # Exact SEG-contour -> feature coverage across every matrix layer.
                # The percentage denominator is each SEG contour's own area.
                all_layer_query = DefectContourQuery(all_fov_features)
                all_layer_overlay_features: dict[str, tuple[str, object]] = {}
                all_layer_overlay_intersections = []
                for detection, defect_polygon in zip(detections, defect_polygons):
                    contour_hits = sorted(
                        all_layer_query.query(defect_polygon),
                        key=lambda h: h.defect_overlap_pct,
                        reverse=True,
                    )
                    coverage_rows = []
                    for hit in contour_hits:
                        coverage_rows.append(_feature_hit_row(hit))
                        key = f"{hit.feature.layer}:{hit.feature.feature_id}"
                        all_layer_overlay_features[key] = (
                            hit.feature.primitive_type,
                            geometry_to_cam_pixels(
                                hit.feature.geometry,
                                image_center_aoi_mm=center_aoi_mm,
                                cam_size_px=cam_size,
                                resolution_um_per_px=resolution_um_per_px,
                                transform=aoi_odb_transform,
                            ),
                        )
                        all_layer_overlay_intersections.append(
                            geometry_to_cam_pixels(
                                hit.intersection_geometry,
                                image_center_aoi_mm=center_aoi_mm,
                                cam_size_px=cam_size,
                                resolution_um_per_px=resolution_um_per_px,
                                transform=aoi_odb_transform,
                            )
                        )
                    detection["odb_all_layers"] = {
                        "contour_area_mm2": float(defect_polygon.area),
                        "hit_count": len(contour_hits),
                        "features": coverage_rows,
                        "sum_contour_overlap_pct": float(
                            sum(hit.defect_overlap_pct for hit in contour_hits)
                        ),
                        "percentage_note": (
                            "Each contour_overlap_pct = intersection_area / SEG contour area * 100. "
                            "Percentages can sum above 100 when geometries from different ODB layers overlap."
                        ),
                    }

                active = [row for row in context_layer_rows if row["fov_feature_count"] > 0]
                for row in active:
                    print(
                        f"  {row['name']} [{row['type']}] FOV features={row['fov_feature_count']}",
                        flush=True,
                    )
                if not active:
                    print("  No supported P/L/S geometry found in CAM FOV on any matrix layer.", flush=True)

            seg_overlay = draw_contours_on_cam(cam_copy, cam_contours, line_width=line_width)
            seg_output_path = output_dir / f"{cam_path.stem}_SEG_OVERLAY.png"
            seg_overlay.save(seg_output_path, format="PNG")

            odb_context_output_path = None
            odb_output_path = None
            if odb_enabled:
                context_overlay = draw_odb_feature_overlay(
                    cam_copy,
                    seg_contours_px=cam_contours,
                    feature_geometries=context_features.values(),
                    line_width=line_width,
                )
                odb_context_output_path = output_dir / f"{cam_path.stem}_ODB_CONTEXT.png"
                context_overlay.save(odb_context_output_path, format="PNG")

                odb_overlay = draw_odb_feature_overlay(
                    cam_copy,
                    seg_contours_px=cam_contours,
                    feature_geometries=overlay_features.values(),
                    line_width=line_width,
                )
                odb_output_path = output_dir / f"{cam_path.stem}_ODB_FEATURES.png"
                odb_overlay.save(odb_output_path, format="PNG")

                intersection_overlay = draw_odb_feature_overlay(
                    cam_copy,
                    seg_contours_px=cam_contours,
                    feature_geometries=overlay_features.values(),
                    intersection_geometries=overlay_intersections,
                    line_width=line_width,
                )
                intersection_output_path = output_dir / f"{cam_path.stem}_ODB_INTERSECTIONS.png"
                intersection_overlay.save(intersection_output_path, format="PNG")

                all_layers_intersection_output_path = None
                if context_all_layers:
                    all_layers_intersection_overlay = draw_odb_feature_overlay(
                        cam_copy,
                        seg_contours_px=cam_contours,
                        feature_geometries=all_layer_overlay_features.values(),
                        intersection_geometries=all_layer_overlay_intersections,
                        line_width=line_width,
                    )
                    all_layers_intersection_output_path = output_dir / f"{cam_path.stem}_ODB_CONTOUR_FEATURE_COVERAGE_ALL_LAYERS.png"
                    all_layers_intersection_overlay.save(all_layers_intersection_output_path, format="PNG")
            else:
                intersection_output_path = None
                all_layers_intersection_output_path = None

            items.append({
                "source_image": str(source_path),
                "copied_source_image": str(source_output_path),
                "cam_image": str(cam_path),
                "seg_overlay_image": str(seg_output_path),
                "odb_context_overlay_image": str(odb_context_output_path) if odb_context_output_path else None,
                "odb_context_feature_count": context_feature_count if odb_enabled else None,
                "odb_context_layers": context_layer_rows if odb_enabled and context_all_layers else None,
                "odb_features_overlay_image": str(odb_output_path) if odb_output_path else None,
                "odb_intersections_overlay_image": str(intersection_output_path) if intersection_output_path else None,
                "odb_all_layers_contour_coverage_overlay_image": str(all_layers_intersection_output_path) if all_layers_intersection_output_path else None,
                "source_size_px": list(source_size),
                "cam_size_px": list(cam_size),
                "aoi_center_mm": [aoi_x_mm, aoi_y_mm],
                "detection_count": len(detections),
                "detections": detections,
            })

        report = {
            "model": str(model_path),
            "gids_dir": str(gids_dir),
            "output_dir": str(output_dir),
            "discovered_pair_count": len(discovered_pairs),
            "processed_pair_count": len(pairs),
            "selection": {"image": image, "limit": limit},
            "confidence_threshold": conf,
            "odb": None if not odb_enabled else {
                "input": str(odb_input),
                "recipe_layer": recipe_layer,
                "signal_layer": signal_layer,
                "root_step": root_step,
                "feature_count": len(features),
                "resolution_um_per_px": resolution_um_per_px,
                "orientation": "SWAP_X+_Y-",
                "tx_mm": tx_mm,
                "ty_mm": ty_mm,
                "context_all_layers": context_all_layers,
                "exact_defect_query_layer": signal_layer,
                "overlay_legend": {"SEG": "red", "P": "cyan", "L": "yellow", "S": "magenta", "intersection": "green"},
                "overlay_policy": {
                    "ODB_CONTEXT": "all matrix layers intersecting the CAM field of view when --context-all-layers is enabled; otherwise selected signal layer only; visualization only",
                    "ODB_FEATURES": "ODB features with positive-area exact SEG-contour intersection",
                    "ODB_INTERSECTIONS": "exact SEG/ODB intersection geometry on the selected recipe signal layer",
                    "ODB_CONTOUR_FEATURE_COVERAGE_ALL_LAYERS": "exact positive-area intersections between each SEG contour and FOV features from every matrix layer; each feature reports intersection_area / contour_area * 100",
                },
            },
            "items": items,
        }
        report_path = output_dir / "seg_cam_overlay_report.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Model           : {model_path}")
        print(f"Discovered pairs: {len(discovered_pairs)}")
        print(f"Processed pairs : {len(pairs)}")
        print(f"Output          : {output_dir}")
        print(f"Report          : {report_path}")
        return report
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run YOLOv8 SEG on G images and optionally overlay exact intersecting ODB features on matched C CAM crops.")
    parser.add_argument("--gids-dir", type=Path, default=Path("data/GIDS"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("data/GIDS/seg_cam_overlay"))
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default=None)
    parser.add_argument("--line-width", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--image", default=None)
    parser.add_argument("--odb-input", type=Path, default=None, help="ODB++ .tgz/.tar.gz or extracted job directory")
    parser.add_argument("--recipe-layer", default=None, help="AOI recipe layer, e.g. L1_TU")
    parser.add_argument("--resolution-um-per-px", type=float, default=None)
    parser.add_argument("--tx-mm", type=float, default=None)
    parser.add_argument("--ty-mm", type=float, default=None)
    parser.add_argument("--root-step", default="pnl")
    parser.add_argument(
        "--context-all-layers",
        action="store_true",
        help="Diagnostic only: sweep every ODB matrix layer in the fixed CAM FOV and save per-layer context overlays. Exact defect/spec intersection remains on the selected recipe signal layer.",
    )
    args = parser.parse_args()

    model_path = _resolve_model(args.model, args.models_dir)
    run(
        args.gids_dir, args.output, model_path,
        conf=args.conf, device=args.device, line_width=args.line_width,
        image=args.image, limit=args.limit,
        odb_input=args.odb_input, recipe_layer=args.recipe_layer,
        resolution_um_per_px=args.resolution_um_per_px,
        tx_mm=args.tx_mm, ty_mm=args.ty_mm, root_step=args.root_step,
        context_all_layers=args.context_all_layers,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
