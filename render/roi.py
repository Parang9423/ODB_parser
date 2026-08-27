#!/usr/bin/env python3
"""Small ROI CAM rendering for AOI coordinate validation.

The renderer draws only the requested physical window instead of rasterizing the
whole panel. SIGNAL features are rendered at GV 255 and drill-like features at
GV 125 by default. PNL/STRIP/UNIT hierarchy transforms are still applied.

Layer selection is based on physical layer identity, not product-specific names:
- AOI L1 / L1-TU / L1_TD all mean physical signal layer 1.
- Drill names such as DLD_1-2 or TH_1-4 are selected only when their layer span
  contains the current signal layer.

For debugging, ``return_components=True`` returns SIGNAL-only and DRILL-only
rasters plus per-layer primitive/symbol-size diagnostics. This is useful for
separating coordinate errors from ODB symbol/raster interpretation errors.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image, ImageChops, ImageDraw

from alignment.guide_drill import is_drill_layer
from app_core import inspect_job
from hierarchy_renderer import FastODBRenderer
from odb_cam_renderer import RasterCanvas, Transform, parse_standard_symbol


@dataclass(frozen=True)
class ROILayerSelection:
    physical_signal_layer: int
    signal_layer: str
    drill_layers: tuple[str, ...]
    excluded_drill_layers: tuple[str, ...] = ()


class FixedRasterCanvas(RasterCanvas):
    """RasterCanvas with an exact requested output size."""

    def __init__(self, bounds, dpi_x: float, dpi_y: float, width_px: int, height_px: int, background: int = 0):
        self.xmin, self.ymin, self.xmax, self.ymax = bounds
        self.dpi_x = float(dpi_x)
        self.dpi_y = float(dpi_y)
        self.dpi = self.dpi_x
        self.margin = 0
        self.image = Image.new("L", (int(width_px), int(height_px)), color=int(background))
        self.draw = ImageDraw.Draw(self.image)


def roi_bounds_in(center_x_mm: float, center_y_mm: float, resolution_um_per_px: float,
                  width_px: int = 100, height_px: int = 100):
    if resolution_um_per_px <= 0:
        raise ValueError("resolution_um_per_px must be positive")
    width_in = (width_px * resolution_um_per_px / 1000.0) / 25.4
    height_in = (height_px * resolution_um_per_px / 1000.0) / 25.4
    cx, cy = center_x_mm / 25.4, center_y_mm / 25.4
    return (
        cx - width_in / 2.0,
        cy - height_in / 2.0,
        cx + width_in / 2.0,
        cy + height_in / 2.0,
    )


def _recipe_tokens(recipe_layer: str) -> tuple[str | None, str | None]:
    parts = [part for part in re.split(r"[-_]+", recipe_layer.strip().upper()) if part]
    layer_no = next((part for part in parts if re.fullmatch(r"L\d+", part)), None)
    orientation = next((part for part in parts if part in {"TU", "TD", "BU", "BD"}), None)
    return layer_no, orientation


def _physical_layer_number(value: str) -> int | None:
    match = re.search(r"(?:^|[-_])L(\d+)(?=$|[-_])", value.strip().upper())
    if match:
        return int(match.group(1))
    match = re.match(r"^L(\d+)(?:\D|$)", value.strip().upper())
    return int(match.group(1)) if match else None


def _normalized_layer_parts(name: str) -> set[str]:
    return {part for part in re.split(r"[-_]+", name.strip().upper()) if part}


def _drill_layer_span(name: str) -> tuple[int, int] | None:
    """Parse connected physical-layer span from DLD_1-2 / TH_1-4 style names."""
    text = name.strip().upper()
    matches = list(re.finditer(r"(?<!\d)(\d+)\s*[-_]\s*(\d+)(?!\d)", text))
    if not matches:
        return None
    a, b = int(matches[-1].group(1)), int(matches[-1].group(2))
    return min(a, b), max(a, b)


def _span_contains(span: tuple[int, int] | None, physical_layer: int) -> bool:
    return span is not None and span[0] <= physical_layer <= span[1]


def select_roi_layers(job: Path, recipe_layer: str) -> ROILayerSelection:
    """Select SIGNAL and DRILL layers using physical layer number as authority."""
    info = inspect_job(job)
    signal_layers = [layer for layer in info.layers if layer.layer_type.upper() == "SIGNAL"]
    if not signal_layers:
        raise ValueError("ODB Matrix contains no SIGNAL layer")

    layer_token, orientation = _recipe_tokens(recipe_layer)
    if not layer_token:
        raise ValueError(f"Could not extract physical layer number from AOI layer {recipe_layer!r}")
    physical_layer = int(layer_token[1:])

    by_layer = [layer.name for layer in signal_layers if _physical_layer_number(layer.name) == physical_layer]
    if not by_layer:
        names = ", ".join(layer.name for layer in signal_layers)
        raise ValueError(
            f"No ODB SIGNAL layer matches AOI physical layer L{physical_layer} from {recipe_layer!r}. "
            f"Refusing to fall back to another physical layer. SIGNAL layers=[{names}]"
        )

    if len(by_layer) == 1:
        selected = by_layer[0]
    elif orientation:
        oriented = [name for name in by_layer if orientation in _normalized_layer_parts(name)]
        if len(oriented) == 1:
            selected = oriented[0]
        else:
            raise ValueError(
                f"Multiple ODB SIGNAL layers match physical layer L{physical_layer}, but orientation "
                f"{orientation!r} is not unique. Candidates={by_layer}"
            )
    else:
        bare = [name for name in by_layer if name.strip().upper() == f"L{physical_layer}"]
        if len(bare) == 1:
            selected = bare[0]
        else:
            raise ValueError(
                f"Multiple ODB SIGNAL layers match physical layer L{physical_layer} and AOI recipe has no "
                f"orientation discriminator. Candidates={by_layer}"
            )

    included_drills: list[str] = []
    excluded_drills: list[str] = []
    for layer in info.layers:
        if not is_drill_layer(layer):
            continue
        if _span_contains(_drill_layer_span(layer.name), physical_layer):
            included_drills.append(layer.name)
        else:
            excluded_drills.append(layer.name)

    return ROILayerSelection(
        physical_signal_layer=physical_layer,
        signal_layer=selected,
        drill_layers=tuple(included_drills),
        excluded_drill_layers=tuple(excluded_drills),
    )


def _render_layer_mask(renderer: FastODBRenderer, root_step: str, layer: str,
                       visible_steps: Iterable[str], bounds, width_px: int, height_px: int) -> Image.Image:
    canvas = FixedRasterCanvas(bounds, renderer.dpi_x, renderer.dpi_y, width_px, height_px, background=0)
    renderer._render_step_filtered(
        canvas,
        root_step.lower(),
        layer.lower(),
        Transform(),
        {step.lower() for step in visible_steps},
    )
    return canvas.image


def _nonzero_pixels(image: Image.Image) -> int:
    hist = image.histogram()
    return int(sum(hist[1:])) if hist else 0


def _bbox_intersects(a, b) -> bool:
    return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3])


def _feature_diagnostics(renderer: FastODBRenderer, root_step: str, layer: str,
                         visible_steps: Iterable[str], bounds,
                         resolution_um_per_px: float, max_samples: int = 20) -> dict:
    """Inspect primitives whose transformed geometry intersects the ROI.

    This does not replace the rasterizer. It records raw ODB symbol strings and
    the renderer's interpreted physical/pixel dimensions so scale errors can be
    diagnosed independently of coordinate matching.
    """
    visible = {str(step).lower() for step in visible_steps}
    roi = tuple(map(float, bounds))
    counts = {"pads": 0, "lines": 0, "surfaces": 0}
    samples: list[dict] = []
    symbol_counts: dict[str, int] = {}
    unsupported_symbols: dict[str, int] = {}
    dpi = 25400.0 / float(resolution_um_per_px)

    for instance in renderer.collect_instances(root_step):
        if instance.step not in visible:
            continue
        feature_file = renderer._step_dir(instance.step) / "layers" / layer.lower() / "features"
        symbols, records = renderer._feature_data(feature_file)
        if not records:
            continue
        index = 0
        while index < len(records):
            raw = records[index].strip(); index += 1
            if not raw or raw.startswith(("#", "$")):
                continue
            tokens = raw.split()
            cmd = tokens[0]
            try:
                if cmd == "P" and len(tokens) >= 5:
                    x, y = float(tokens[1]), float(tokens[2])
                    symbol_id = int(tokens[3])
                    polarity = tokens[4].upper()
                    symbol = symbols.get(symbol_id, "")
                    parsed = parse_standard_symbol(symbol)
                    center = instance.transform.apply((x, y))
                    if parsed is not None:
                        kind, width_in, height_in = parsed
                        half_w, half_h = width_in / 2.0, height_in / 2.0
                    else:
                        kind, width_in, height_in = "unsupported", 0.0, 0.0
                    bbox = (center[0] - half_w, center[1] - half_h, center[0] + half_w, center[1] + half_h)
                    if _bbox_intersects(bbox, roi):
                        counts["pads"] += 1
                        symbol_counts[symbol or f"#{symbol_id}"] = symbol_counts.get(symbol or f"#{symbol_id}", 0) + 1
                        if parsed is None:
                            unsupported_symbols[symbol or f"#{symbol_id}"] = unsupported_symbols.get(symbol or f"#{symbol_id}", 0) + 1
                        if len(samples) < max_samples:
                            samples.append({
                                "step": instance.step.upper(), "type": "PAD", "polarity": polarity,
                                "symbol_id": symbol_id, "symbol": symbol, "symbol_kind": kind,
                                "center_mm": [center[0] * 25.4, center[1] * 25.4],
                                "width_mm": width_in * 25.4, "height_mm": height_in * 25.4,
                                "expected_width_px": width_in * dpi, "expected_height_px": height_in * dpi,
                            })
                elif cmd == "L" and len(tokens) >= 7:
                    p1 = instance.transform.apply((float(tokens[1]), float(tokens[2])))
                    p2 = instance.transform.apply((float(tokens[3]), float(tokens[4])))
                    symbol_id = int(tokens[5]); polarity = tokens[6].upper()
                    symbol = symbols.get(symbol_id, "")
                    parsed = parse_standard_symbol(symbol)
                    diameter = parsed[1] if parsed is not None else 0.0
                    bbox = (min(p1[0], p2[0]) - diameter / 2, min(p1[1], p2[1]) - diameter / 2,
                            max(p1[0], p2[0]) + diameter / 2, max(p1[1], p2[1]) + diameter / 2)
                    if _bbox_intersects(bbox, roi):
                        counts["lines"] += 1
                        symbol_counts[symbol or f"#{symbol_id}"] = symbol_counts.get(symbol or f"#{symbol_id}", 0) + 1
                        if parsed is None:
                            unsupported_symbols[symbol or f"#{symbol_id}"] = unsupported_symbols.get(symbol or f"#{symbol_id}", 0) + 1
                        if len(samples) < max_samples:
                            samples.append({
                                "step": instance.step.upper(), "type": "LINE", "polarity": polarity,
                                "symbol_id": symbol_id, "symbol": symbol,
                                "start_mm": [p1[0] * 25.4, p1[1] * 25.4],
                                "end_mm": [p2[0] * 25.4, p2[1] * 25.4],
                                "width_mm": diameter * 25.4, "expected_width_px": diameter * dpi,
                            })
                elif cmd == "S" and len(tokens) >= 2:
                    polarity = tokens[1].upper()
                    points = []
                    while index < len(records):
                        sr = records[index].strip(); index += 1
                        if not sr or sr.startswith("#"):
                            continue
                        vals = sr.split(); scmd = vals[0]
                        if scmd in {"OB", "OS"} and len(vals) >= 3:
                            points.append(instance.transform.apply((float(vals[1]), float(vals[2]))))
                        elif scmd == "OC" and len(vals) >= 3:
                            points.append(instance.transform.apply((float(vals[1]), float(vals[2]))))
                        elif scmd == "SE":
                            break
                    if points:
                        xs = [p[0] for p in points]; ys = [p[1] for p in points]
                        bbox = (min(xs), min(ys), max(xs), max(ys))
                        if _bbox_intersects(bbox, roi):
                            counts["surfaces"] += 1
                            if len(samples) < max_samples:
                                samples.append({
                                    "step": instance.step.upper(), "type": "SURFACE", "polarity": polarity,
                                    "bbox_mm": [v * 25.4 for v in bbox], "vertex_records": len(points),
                                })
            except (ValueError, IndexError):
                continue

    return {
        "layer": layer,
        "roi_primitive_counts": counts,
        "symbol_counts": symbol_counts,
        "unsupported_symbols": unsupported_symbols,
        "samples": samples,
        "assumed_symbol_unit": "ODB standard symbol numeric dimension interpreted as mil (0.001 inch)",
        "resolution_um_per_px": resolution_um_per_px,
        "raster_dpi": dpi,
    }


def render_roi_cam(job: Path, center_x_mm: float, center_y_mm: float, resolution_um_per_px: float,
                   recipe_layer: str, width_px: int = 100, height_px: int = 100,
                   signal_gv: int = 255, drill_gv: int = 125,
                   visible_steps: Sequence[str] = ("pnl", "strip", "unit"),
                   return_components: bool = False):
    if width_px <= 0 or height_px <= 0:
        raise ValueError("ROI pixel dimensions must be positive")
    if not 0 <= signal_gv <= 255 or not 0 <= drill_gv <= 255:
        raise ValueError("GV must be between 0 and 255")

    selection = select_roi_layers(job, recipe_layer)
    renderer = FastODBRenderer.from_um_per_pixel(job, resolution_um_per_px, resolution_um_per_px)
    root_step = "pnl" if (job / "steps" / "pnl").is_dir() else next(
        p.name for p in (job / "steps").iterdir() if p.is_dir()
    )
    available_steps = {p.name.lower() for p in (job / "steps").iterdir() if p.is_dir()}
    visible = tuple(step for step in visible_steps if step.lower() in available_steps)
    bounds = roi_bounds_in(center_x_mm, center_y_mm, resolution_um_per_px, width_px, height_px)

    signal_mask = _render_layer_mask(renderer, root_step, selection.signal_layer, visible, bounds, width_px, height_px)
    signal_nonzero = _nonzero_pixels(signal_mask)
    signal_image = Image.new("L", (width_px, height_px), color=0)
    signal_image.paste(int(signal_gv), mask=signal_mask.point(lambda v: 255 if v else 0, mode="L"))
    result = signal_image.copy()

    drill_union = Image.new("L", result.size, color=0)
    used_drills: list[str] = []
    drill_layer_nonzero: dict[str, int] = {}
    drill_masks: dict[str, Image.Image] = {}
    for drill_layer in selection.drill_layers:
        mask = _render_layer_mask(renderer, root_step, drill_layer, visible, bounds, width_px, height_px)
        count = _nonzero_pixels(mask)
        drill_layer_nonzero[drill_layer] = count
        if return_components:
            drill_masks[drill_layer] = mask.copy()
        if count > 0:
            drill_union = ImageChops.lighter(drill_union, mask)
            used_drills.append(drill_layer)
    drill_nonzero = _nonzero_pixels(drill_union)
    drill_image = Image.new("L", result.size, color=0)
    if drill_nonzero > 0:
        drill_binary = drill_union.point(lambda v: 255 if v else 0, mode="L")
        drill_image.paste(int(drill_gv), mask=drill_binary)
        result.paste(int(drill_gv), mask=drill_binary)

    final_nonzero = _nonzero_pixels(result)
    xmin, ymin, xmax, ymax = bounds
    metadata = {
        "center_x_mm": center_x_mm,
        "center_y_mm": center_y_mm,
        "resolution_um_per_px": resolution_um_per_px,
        "size_px": [width_px, height_px],
        "physical_size_mm": [width_px * resolution_um_per_px / 1000.0, height_px * resolution_um_per_px / 1000.0],
        "roi_bounds_mm": [xmin * 25.4, ymin * 25.4, xmax * 25.4, ymax * 25.4],
        "physical_signal_layer": selection.physical_signal_layer,
        "signal_layer": selection.signal_layer,
        "signal_gv": signal_gv,
        "signal_nonzero_pixels": signal_nonzero,
        "drill_layers_considered": list(selection.drill_layers),
        "drill_layers_excluded": list(selection.excluded_drill_layers),
        "drill_layers_rendered": used_drills,
        "drill_layer_nonzero_pixels": drill_layer_nonzero,
        "drill_nonzero_pixels": drill_nonzero,
        "drill_gv": drill_gv,
        "final_nonzero_pixels": final_nonzero,
        "visible_steps": [step.upper() for step in visible],
        "renderer_processed_primitives": {
            "pads": renderer.stats.pads,
            "lines": renderer.stats.lines,
            "surfaces": renderer.stats.surfaces,
            "repeats": renderer.stats.repeats,
            "unsupported": renderer.stats.unsupported,
        },
    }

    if not return_components:
        return result, metadata

    signal_diag = _feature_diagnostics(
        renderer, root_step, selection.signal_layer, visible, bounds, resolution_um_per_px
    )
    drill_diag = {
        layer: _feature_diagnostics(renderer, root_step, layer, visible, bounds, resolution_um_per_px)
        for layer in selection.drill_layers
    }
    metadata["feature_diagnostics"] = {"signal": signal_diag, "drill": drill_diag}
    components = {
        "signal": signal_image,
        "drill": drill_image,
        "signal_mask": signal_mask,
        "drill_mask": drill_union,
        "drill_layer_masks": drill_masks,
    }
    return result, metadata, components
