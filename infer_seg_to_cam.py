#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image

from aoi.seg_cam_overlay import (
    discover_gid_pairs,
    draw_contours_on_cam,
    map_contour_by_shared_center,
)


def _resolve_model(model_arg: Path | None, models_dir: Path) -> Path:
    if model_arg is not None:
        model = model_arg.expanduser().resolve()
        if not model.is_file():
            raise FileNotFoundError(f"Model file not found: {model}")
        return model

    candidates = sorted(models_dir.glob("*.pt"))
    if not candidates:
        raise FileNotFoundError(
            f"No .pt model found in {models_dir}. Put one YOLOv8 SEG model in models/ "
            "or pass --model explicitly."
        )
    if len(candidates) > 1:
        raise ValueError(
            f"Multiple .pt models found in {models_dir}: {candidates}. Pass --model explicitly."
        )
    return candidates[0].resolve()


def _load_yolo(model_path: Path):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Ultralytics is required for YOLOv8 SEG inference. Install it in the active environment: "
            "pip install ultralytics"
        ) from exc
    return YOLO(str(model_path))


def _to_python_contour(segment: Any) -> list[tuple[float, float]]:
    if hasattr(segment, "tolist"):
        segment = segment.tolist()
    points: list[tuple[float, float]] = []
    for row in segment:
        if len(row) < 2:
            continue
        points.append((float(row[0]), float(row[1])))
    return points


def _class_name(names: Any, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def run(
    gids_dir: Path,
    output_dir: Path,
    model_path: Path,
    *,
    conf: float = 0.25,
    device: str | None = None,
    line_width: int = 2,
) -> dict:
    if not 0.0 <= conf <= 1.0:
        raise ValueError("conf must be between 0 and 1")

    pairs = discover_gid_pairs(gids_dir)
    if not pairs:
        raise ValueError(f"No matching G/C image pairs found under {gids_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    model = _load_yolo(model_path)
    items: list[dict] = []

    for source_path, cam_path in pairs:
        with Image.open(source_path) as src_im:
            source_size = src_im.size
        with Image.open(cam_path) as cam_im:
            cam_size = cam_im.size
            cam_copy = cam_im.copy()

        kwargs = {
            "source": str(source_path),
            "conf": conf,
            "verbose": False,
        }
        if device:
            kwargs["device"] = device
        results = model.predict(**kwargs)
        if not results:
            raise RuntimeError(f"YOLO returned no result object for {source_path}")
        result = results[0]

        source_contours: list[list[tuple[float, float]]] = []
        cam_contours: list[list[tuple[float, float]]] = []
        detections: list[dict] = []

        masks = getattr(result, "masks", None)
        segments = [] if masks is None else list(masks.xy)
        boxes = getattr(result, "boxes", None)
        cls_values = [] if boxes is None else boxes.cls.detach().cpu().tolist()
        conf_values = [] if boxes is None else boxes.conf.detach().cpu().tolist()
        names = getattr(result, "names", {})

        for index, segment in enumerate(segments):
            source_contour = _to_python_contour(segment)
            if len(source_contour) < 3:
                continue
            cam_contour = map_contour_by_shared_center(source_contour, source_size, cam_size)
            source_contours.append(source_contour)
            cam_contours.append(cam_contour)

            class_id = int(cls_values[index]) if index < len(cls_values) else -1
            confidence = float(conf_values[index]) if index < len(conf_values) else None
            detections.append({
                "index": index,
                "class_id": class_id,
                "class_name": _class_name(names, class_id),
                "confidence": confidence,
                "source_contour_px": [[x, y] for x, y in source_contour],
                "cam_contour_px": [[x, y] for x, y in cam_contour],
            })

        overlay = draw_contours_on_cam(cam_copy, cam_contours, line_width=line_width)
        output_path = output_dir / f"{cam_path.stem}_SEG_OVERLAY.png"
        overlay.save(output_path, format="PNG")

        items.append({
            "source_image": str(source_path),
            "cam_image": str(cam_path),
            "output_image": str(output_path),
            "source_size_px": list(source_size),
            "cam_size_px": list(cam_size),
            "mapping": "shared_center_same_pixel_resolution",
            "center_offset_px": [
                (cam_size[0] - source_size[0]) / 2.0,
                (cam_size[1] - source_size[1]) / 2.0,
            ],
            "detection_count": len(detections),
            "detections": detections,
        })

    report = {
        "model": str(model_path),
        "gids_dir": str(gids_dir),
        "output_dir": str(output_dir),
        "pair_count": len(pairs),
        "confidence_threshold": conf,
        "items": items,
    }
    report_path = output_dir / "seg_cam_overlay_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Model        : {model_path}")
    print(f"Matched pairs: {len(pairs)}")
    print(f"Output       : {output_dir}")
    print(f"Report       : {report_path}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run YOLOv8 segmentation on G_* AOI images, match C_* CAM images by the first two "
            "filename coordinates, and draw the segmentation contours on the CAM crops."
        )
    )
    parser.add_argument("--gids-dir", type=Path, default=Path("data/GIDS"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("data/GIDS/seg_cam_overlay"))
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default=None, help="Ultralytics device, e.g. 0, cpu, cuda:0")
    parser.add_argument("--line-width", type=int, default=2)
    args = parser.parse_args()

    model_path = _resolve_model(args.model, args.models_dir)
    run(
        args.gids_dir,
        args.output,
        model_path,
        conf=args.conf,
        device=args.device,
        line_width=args.line_width,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
