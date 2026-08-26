#!/usr/bin/env python3
"""Generate 100x100 ODB CAM crops from coordinate_validation.json."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from odb_cam_renderer import extract_input
from render.roi import render_roi_cam


def _safe_rel_parts(image_path: Path) -> tuple[str, ...]:
    parts = image_path.parts
    lowered = [part.casefold() for part in parts]
    try:
        idx = lowered.index("gids")
        return tuple(parts[idx + 1 : -1])
    except ValueError:
        return tuple()


def _find_hypothesis(detail: dict, name: str) -> dict:
    for row in detail.get("hypotheses", []):
        if str(row.get("hypothesis", "")).upper() == name.upper():
            return row
    raise KeyError(f"Hypothesis {name!r} not found for {detail.get('image_context', {}).get('image_path')}")


def generate_from_validation_json(validation_json: str | Path, output_dir: str | Path,
                                  hypothesis: str = "DIRECT_LOCAL",
                                  width_px: int = 100, height_px: int = 100,
                                  signal_gv: int = 255, drill_gv: int = 125,
                                  limit: int | None = None) -> int:
    src = Path(validation_json).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    payload = json.loads(src.read_text(encoding="utf-8"))
    details = list(payload.get("results", []))
    if limit is not None:
        details = details[: max(0, limit)]
    if not details:
        raise ValueError(f"No validation results found in {src}")

    odb_cache: dict[str, tuple[Path, object | None]] = {}
    generated: list[dict] = []
    failures: list[dict] = []
    try:
        for detail in details:
            image_info = detail.get("image_context", {})
            resources = detail.get("resources", {})
            ert = detail.get("ert", {})
            image_path = Path(image_info.get("image_path", ""))
            try:
                row = _find_hypothesis(detail, hypothesis)
                odb_path = str(resources["odb_path"])
                if odb_path not in odb_cache:
                    job, temp_dir = extract_input(Path(odb_path))
                    odb_cache[odb_path] = (job, temp_dir)
                job = odb_cache[odb_path][0]

                resolution = float(ert["resolution_um_per_px"])
                recipe_layer = str(image_info["layer"])
                center_x_mm = float(row["odb_x_mm"])
                center_y_mm = float(row["odb_y_mm"])
                cam, meta = render_roi_cam(
                    job=job,
                    center_x_mm=center_x_mm,
                    center_y_mm=center_y_mm,
                    resolution_um_per_px=resolution,
                    recipe_layer=recipe_layer,
                    width_px=width_px,
                    height_px=height_px,
                    signal_gv=signal_gv,
                    drill_gv=drill_gv,
                )

                rel_dir = _safe_rel_parts(image_path)
                target_dir = output.joinpath(*rel_dir) if rel_dir else output
                target_dir.mkdir(parents=True, exist_ok=True)
                out_name = f"{image_path.stem}_ODB_CAM_{hypothesis}.png"
                out_path = target_dir / out_name
                cam.save(out_path, format="PNG", compress_level=1, optimize=False)
                generated.append({
                    "source_image": str(image_path),
                    "output_cam": str(out_path),
                    "hypothesis": hypothesis,
                    "coordinate_mm": [center_x_mm, center_y_mm],
                    **meta,
                })
            except Exception as exc:
                failures.append({
                    "image": str(image_path),
                    "error": f"{type(exc).__name__}: {exc}",
                })
    finally:
        for _job, temp_dir in odb_cache.values():
            if temp_dir is not None:
                temp_dir.cleanup()

    report = output / "cam_crop_generation.json"
    report.write_text(
        json.dumps({"generated": generated, "failures": failures}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Validation rows : {len(details)}")
    print(f"CAM generated   : {len(generated)}")
    print(f"Failures        : {len(failures)}")
    print(f"Report          : {report}")
    return 0 if not failures else 2


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate ODB CAM ROI crops from coordinate validation JSON")
    parser.add_argument("validation_json", type=Path, help="coordinate_validation.json")
    parser.add_argument("--output", type=Path, default=Path("cam_crops"))
    parser.add_argument("--hypothesis", default="DIRECT_LOCAL")
    parser.add_argument("--width", type=int, default=100)
    parser.add_argument("--height", type=int, default=100)
    parser.add_argument("--signal-gv", type=int, default=255)
    parser.add_argument("--drill-gv", type=int, default=125)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    return generate_from_validation_json(
        args.validation_json,
        args.output,
        hypothesis=args.hypothesis,
        width_px=args.width,
        height_px=args.height,
        signal_gv=args.signal_gv,
        drill_gv=args.drill_gv,
        limit=args.limit,
    )


if __name__ == "__main__":
    raise SystemExit(main())
