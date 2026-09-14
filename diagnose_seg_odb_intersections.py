#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from aoi.contour_mapping import AoiOdbTransform
from aoi.seg_result import iter_detection_odb_polygons, load_seg_result, parse_aoi_image_filename
from hierarchy_renderer import FastODBRenderer
from odb.feature_query import DefectContourQuery, extract_vector_features
from odb_cam_renderer import extract_input
from render.roi import select_roi_layers


def _bounds_list(geometry) -> list[float]:
    return [float(v) for v in geometry.bounds]


def diagnose(
    seg_json: str | Path,
    odb_input: str | Path,
    recipe_layer: str,
    resolution_um_per_px: float,
    tx_mm: float,
    ty_mm: float,
    root_step: str = "pnl",
) -> dict:
    result = load_seg_result(seg_json)
    filename = parse_aoi_image_filename(result.image)
    center_aoi_mm = (filename.aoi_x_mm, filename.aoi_y_mm)
    transform = AoiOdbTransform(tx_mm=tx_mm, ty_mm=ty_mm)

    job, temp_dir = extract_input(Path(odb_input))
    try:
        selection = select_roi_layers(job, recipe_layer)
        signal_layer = selection.signal_layer
        renderer = FastODBRenderer(job, dpi=100.0)
        features = extract_vector_features(
            renderer=renderer,
            root_step=root_step,
            layers=[signal_layer],
            positive_only=True,
        )
        query = DefectContourQuery(features)

        detections = []
        for index, (detection, polygon) in enumerate(
            iter_detection_odb_polygons(
                result=result,
                image_center_aoi_mm=center_aoi_mm,
                resolution_um_per_px=resolution_um_per_px,
                transform=transform,
            )
        ):
            hits = query.query(polygon)
            detections.append({
                "index": index,
                "ai_class_id": detection.class_id,
                "ai_class_name": detection.class_name,
                "confidence": detection.confidence,
                "bbox_px": list(detection.bbox),
                "contour_point_count": len(detection.mask_contour),
                "defect_polygon_bounds_mm": _bounds_list(polygon),
                "defect_area_mm2": float(polygon.area),
                "hit_count": len(hits),
                "hits": [
                    {
                        "feature_id": hit.feature.feature_id,
                        "layer": hit.feature.layer,
                        "step": hit.feature.step,
                        "depth": hit.feature.depth,
                        "primitive_type": hit.feature.primitive_type,
                        "symbol": hit.feature.symbol,
                        "polarity": hit.feature.polarity,
                        "feature_bounds_mm": list(hit.feature.bounds),
                        "intersection_area_mm2": hit.intersection_area,
                        "defect_overlap_pct": hit.defect_overlap_pct,
                    }
                    for hit in sorted(hits, key=lambda row: row.defect_overlap_pct, reverse=True)
                ],
            })

        return {
            "seg_json": str(Path(seg_json)),
            "image": result.image,
            "image_size_px": [result.width, result.height],
            "aoi_center_mm": [filename.aoi_x_mm, filename.aoi_y_mm],
            "aoi_defect_class_id": filename.aoi_defect_class_id,
            "resolution_um_per_px": resolution_um_per_px,
            "aoi_to_odb": {
                "orientation": "SWAP_X+_Y-",
                "tx_mm": tx_mm,
                "ty_mm": ty_mm,
            },
            "root_step": root_step,
            "recipe_layer": recipe_layer,
            "odb_signal_layer": signal_layer,
            "extracted_feature_count": len(features),
            "detections": detections,
        }
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Map SEG JSON contours to ODB coordinates and report exact intersecting ODB features."
    )
    parser.add_argument("seg_json", type=Path)
    parser.add_argument("odb_input", type=Path, help="ODB++ .tgz/.tar.gz or extracted job directory")
    parser.add_argument("--recipe-layer", required=True, help="AOI physical layer recipe, e.g. L5-BD")
    parser.add_argument("--resolution-um-per-px", required=True, type=float)
    parser.add_argument("--tx-mm", required=True, type=float)
    parser.add_argument("--ty-mm", required=True, type=float)
    parser.add_argument("--root-step", default="pnl")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.resolution_um_per_px <= 0:
        parser.error("--resolution-um-per-px must be > 0")

    payload = diagnose(
        seg_json=args.seg_json,
        odb_input=args.odb_input,
        recipe_layer=args.recipe_layer,
        resolution_um_per_px=args.resolution_um_per_px,
        tx_mm=args.tx_mm,
        ty_mm=args.ty_mm,
        root_step=args.root_step,
    )
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"Saved: {args.output}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
