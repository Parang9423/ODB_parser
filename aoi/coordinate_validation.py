#!/usr/bin/env python3
"""Validation pipeline for testing whether AOI image coordinates directly match ODB geometry.

This module intentionally does *not* apply an alignment transform.  It resolves an
AOI image to its ERT/ODB resources, parses the filename Y/X coordinate and ERT
resolution, then tests a small set of explicit coordinate hypotheses against ODB
PNL/STRIP/UNIT profiles.  The goal is to prove the coordinate convention before
production inference code is built around it.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

from aoi.ert import ERTMetadata, parse_ert
from hierarchy_renderer import FastODBRenderer
from odb_cam_renderer import extract_input

_IMAGE_RE = re.compile(
    r"^[^_]+_(?P<y>[+-]?\d+(?:\.\d+)?)_(?P<x>[+-]?\d+(?:\.\d+)?)(?:_(?P<index>[^.]+))?\.png$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ImageContext:
    image_path: Path
    item_revision: str
    layer: str
    lot: str
    panel: str
    x_mm: float
    y_mm: float
    image_index: str = ""


@dataclass(frozen=True)
class ResourceContext:
    ert_path: Path
    odb_path: Path


@dataclass(frozen=True)
class CoordinateHypothesis:
    name: str
    x_in: float
    y_in: float
    description: str


@dataclass(frozen=True)
class StepHit:
    step: str
    instance_index: int
    inside: bool
    local_x_mm: float
    local_y_mm: float


@dataclass(frozen=True)
class ValidationResult:
    image: str
    item_revision: str
    layer: str
    lot: str
    panel: str
    ert: str
    odb: str
    resolution_um_per_px: float
    aoi_x_mm: float
    aoi_y_mm: float
    roi_width_mm: float
    roi_height_mm: float
    hypothesis: str
    hypothesis_description: str
    odb_x_mm: float
    odb_y_mm: float
    pnl_inside: bool
    strip_hits: int
    unit_hits: int
    deepest_step: str


def parse_image_context(image_path: str | Path, gids_root: str | Path) -> ImageContext:
    image = Path(image_path).resolve()
    root = Path(gids_root).resolve()
    try:
        rel = image.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Image is not under GIDS root: {image}") from exc
    if len(rel.parts) < 5:
        raise ValueError(
            "Expected GIDS/<item-revision>/<layer>/<lot>/<panel>/<image>.png; "
            f"got relative path: {rel}"
        )
    item_revision, layer, lot, panel = rel.parts[-5:-1]
    match = _IMAGE_RE.match(image.name)
    if not match:
        raise ValueError(f"Image filename does not match G_<Y>_<X>_<index>.png style: {image.name}")
    return ImageContext(
        image_path=image,
        item_revision=item_revision,
        layer=layer,
        lot=lot,
        panel=panel,
        x_mm=float(match.group("x")),
        y_mm=float(match.group("y")),
        image_index=match.group("index") or "",
    )


def _casefold_child(parent: Path, name: str) -> Optional[Path]:
    if not parent.is_dir():
        return None
    wanted = name.casefold()
    for child in parent.iterdir():
        if child.name.casefold() == wanted:
            return child
    return None


def resolve_resources(root: str | Path, context: ImageContext) -> ResourceContext:
    root = Path(root).resolve()
    # ERT root may be ROOT/ERT/<item> or directly ROOT/<item>, matching the test layout discussion.
    ert_base = root / "ERT" if (root / "ERT").is_dir() else root
    item_dir = _casefold_child(ert_base, context.item_revision)
    if item_dir is None:
        raise FileNotFoundError(f"ERT item directory not found for {context.item_revision} under {ert_base}")
    layer_dir = _casefold_child(item_dir, context.layer)
    if layer_dir is None:
        raise FileNotFoundError(f"ERT layer directory not found: {context.layer} under {item_dir}")
    lot_dir = _casefold_child(layer_dir, context.lot)
    if lot_dir is None:
        raise FileNotFoundError(f"ERT lot directory not found: {context.lot} under {layer_dir}")
    ert_files = sorted(p for p in lot_dir.rglob("*") if p.is_file() and p.suffix.casefold() == ".ert")
    if not ert_files:
        raise FileNotFoundError(f"No ERT file found under {lot_dir}")

    # Prefer a panel token match when filenames expose it; otherwise only accept an unambiguous single ERT.
    panel_key = context.panel.casefold()
    matched = [p for p in ert_files if panel_key in p.stem.casefold()]
    if len(matched) == 1:
        ert = matched[0]
    elif len(ert_files) == 1:
        ert = ert_files[0]
    else:
        raise RuntimeError(
            f"Multiple ERT files found for lot {context.lot} but panel {context.panel!r} could not be resolved: "
            + ", ".join(p.name for p in ert_files[:10])
        )

    odb_dir = root / "ODB"
    if not odb_dir.is_dir():
        raise FileNotFoundError(f"ODB directory not found: {odb_dir}")
    candidates = [
        p for p in odb_dir.iterdir()
        if p.is_file()
        and p.name.casefold() in {
            f"{context.item_revision}.tgz".casefold(),
            f"{context.item_revision}.tar.gz".casefold(),
        }
    ]
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected exactly one ODB archive for {context.item_revision}; found {len(candidates)} under {odb_dir}"
        )
    return ResourceContext(ert_path=ert, odb_path=candidates[0])


def coordinate_hypotheses(x_mm: float, y_mm: float, pnl_bounds_in: Sequence[float]) -> list[CoordinateHypothesis]:
    """Return explicit hypotheses; no hidden calibration is applied.

    DIRECT_LOCAL is the user's current hypothesis: AOI (0,0) equals ODB local (0,0).
    The other variants are diagnostics only, useful for detecting whether the AOI origin
    instead corresponds to a PNL profile corner or whether Y direction is inverted.
    """
    xmin, ymin, xmax, ymax = map(float, pnl_bounds_in)
    x_in, y_in = x_mm / 25.4, y_mm / 25.4
    return [
        CoordinateHypothesis("DIRECT_LOCAL", x_in, y_in, "AOI X/Y mm == ODB PNL local X/Y mm"),
        CoordinateHypothesis("PNL_MIN_Y_UP", xmin + x_in, ymin + y_in, "AOI origin == PNL profile min corner, +Y"),
        CoordinateHypothesis("PNL_MIN_Y_DOWN", xmin + x_in, ymax - y_in, "AOI origin == PNL top-left, Y downward"),
    ]


def _invert_transform(transform, point_in: tuple[float, float]) -> tuple[float, float]:
    det = transform.a * transform.d - transform.b * transform.c
    if abs(det) < 1e-12:
        raise ValueError("Non-invertible STEP transform")
    px = point_in[0] - transform.tx
    py = point_in[1] - transform.ty
    return (
        (transform.d * px - transform.b * py) / det,
        (-transform.c * px + transform.a * py) / det,
    )


def _point_in_bounds(point: tuple[float, float], bounds: Sequence[float], eps: float = 1e-9) -> bool:
    x, y = point
    xmin, ymin, xmax, ymax = bounds
    return xmin - eps <= x <= xmax + eps and ymin - eps <= y <= ymax + eps


def step_hits(renderer: FastODBRenderer, root_step: str, point_in: tuple[float, float]) -> list[StepHit]:
    hits: list[StepHit] = []
    counters: dict[str, int] = {}
    for instance in renderer.collect_instances(root_step):
        index = counters.get(instance.step, 0)
        counters[instance.step] = index + 1
        local = _invert_transform(instance.transform, point_in)
        try:
            bounds = renderer.profile_bounds(instance.step)
        except Exception:
            continue
        inside = _point_in_bounds(local, bounds)
        if inside:
            hits.append(StepHit(
                step=instance.step.upper(),
                instance_index=index,
                inside=True,
                local_x_mm=local[0] * 25.4,
                local_y_mm=local[1] * 25.4,
            ))
    return hits


def validate_image(root: str | Path, image_path: str | Path) -> tuple[list[ValidationResult], dict]:
    root = Path(root).resolve()
    gids_root = root / "GIDS"
    context = parse_image_context(image_path, gids_root)
    resources = resolve_resources(root, context)
    ert: ERTMetadata = parse_ert(resources.ert_path)

    job, temp_dir = extract_input(resources.odb_path)
    try:
        renderer = FastODBRenderer(job, 72.0)
        root_step = "pnl" if "pnl" in {p.name.lower() for p in (job / "steps").iterdir() if p.is_dir()} else context.layer.lower()
        pnl_bounds = renderer.profile_bounds(root_step)
        roi_w, roi_h = ert.roi_size_mm_for_pixels(100, 100)
        results: list[ValidationResult] = []
        detail: dict = {
            "image_context": {**asdict(context), "image_path": str(context.image_path)},
            "resources": {"ert_path": str(resources.ert_path), "odb_path": str(resources.odb_path)},
            "ert": {
                "resolution_um_per_px": ert.resolution_um_per_px,
                "region_values": list(ert.region_values),
                "guide_reference_candidate": list(ert.guide_reference_candidate),
            },
            "pnl_bounds_mm": [value * 25.4 for value in pnl_bounds],
            "hypotheses": [],
        }
        for hypothesis in coordinate_hypotheses(context.x_mm, context.y_mm, pnl_bounds):
            hits = step_hits(renderer, root_step, (hypothesis.x_in, hypothesis.y_in))
            pnl_inside = any(hit.step == root_step.upper() for hit in hits)
            strip_hits_count = sum(hit.step == "STRIP" for hit in hits)
            unit_hits_count = sum(hit.step == "UNIT" for hit in hits)
            deepest = "UNIT" if unit_hits_count else "STRIP" if strip_hits_count else root_step.upper() if pnl_inside else "NONE"
            result = ValidationResult(
                image=str(context.image_path), item_revision=context.item_revision, layer=context.layer,
                lot=context.lot, panel=context.panel, ert=str(resources.ert_path), odb=str(resources.odb_path),
                resolution_um_per_px=ert.resolution_um_per_px, aoi_x_mm=context.x_mm, aoi_y_mm=context.y_mm,
                roi_width_mm=roi_w, roi_height_mm=roi_h, hypothesis=hypothesis.name,
                hypothesis_description=hypothesis.description,
                odb_x_mm=hypothesis.x_in * 25.4, odb_y_mm=hypothesis.y_in * 25.4,
                pnl_inside=pnl_inside, strip_hits=strip_hits_count, unit_hits=unit_hits_count, deepest_step=deepest,
            )
            results.append(result)
            detail["hypotheses"].append({
                **asdict(result),
                "step_hits": [asdict(hit) for hit in hits],
            })
        return results, detail
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


def discover_images(root: str | Path, limit: Optional[int] = None) -> list[Path]:
    gids = Path(root) / "GIDS"
    images = sorted(p for p in gids.rglob("*.png") if p.is_file())
    return images if limit is None else images[: max(0, limit)]


def run_validation(root: str | Path, output_dir: str | Path, limit: Optional[int] = None) -> int:
    root = Path(root).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    images = discover_images(root, limit)
    if not images:
        raise FileNotFoundError(f"No PNG images found under {root / 'GIDS'}")

    rows: list[dict] = []
    details: list[dict] = []
    failures: list[dict] = []
    for image in images:
        try:
            results, detail = validate_image(root, image)
            rows.extend(asdict(result) for result in results)
            details.append(detail)
        except Exception as exc:
            failures.append({"image": str(image), "error": f"{type(exc).__name__}: {exc}"})

    csv_path = output / "coordinate_validation.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader(); writer.writerows(rows)
    (output / "coordinate_validation.json").write_text(
        json.dumps({"results": details, "failures": failures}, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    print(f"Images checked : {len(images)}")
    print(f"Rows written   : {len(rows)}")
    print(f"Failures       : {len(failures)}")
    print(f"CSV            : {csv_path}")
    print(f"JSON           : {output / 'coordinate_validation.json'}")
    return 0 if not failures else 2


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate AOI filename coordinates against ODB step geometry")
    parser.add_argument("root", type=Path, help="Test root containing GIDS/, ODB/, and ERT data")
    parser.add_argument("--output", type=Path, default=Path("validation_output"))
    parser.add_argument("--limit", type=int, default=None, help="Only inspect the first N PNG files")
    args = parser.parse_args()
    return run_validation(args.root, args.output, args.limit)


if __name__ == "__main__":
    raise SystemExit(main())
