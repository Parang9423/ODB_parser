#!/usr/bin/env python3
"""Small ROI CAM rendering for AOI coordinate validation.

The renderer draws only the requested physical window instead of rasterizing the
whole panel.  SIGNAL features are rendered at GV 255 and drill-like features at
GV 125 by default.  PNL/STRIP/UNIT hierarchy transforms are still applied.
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
from odb_cam_renderer import RasterCanvas, Transform


@dataclass(frozen=True)
class ROILayerSelection:
    signal_layer: str
    drill_layers: tuple[str, ...]


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


def _layer_token(recipe_layer: str) -> str:
    match = re.match(r"^(L\d+)", recipe_layer.strip(), flags=re.I)
    return match.group(1) if match else recipe_layer.strip().split("-", 1)[0]


def select_roi_layers(job: Path, recipe_layer: str) -> ROILayerSelection:
    info = inspect_job(job)
    signal_layers = [layer for layer in info.layers if layer.layer_type.upper() == "SIGNAL"]
    if not signal_layers:
        raise ValueError("ODB Matrix contains no SIGNAL layer")

    token = _layer_token(recipe_layer).casefold()
    exact = [layer.name for layer in signal_layers if layer.name.casefold() == token]
    if len(exact) == 1:
        signal = exact[0]
    else:
        prefix = [layer.name for layer in signal_layers if layer.name.casefold().startswith(token)]
        if len(prefix) == 1:
            signal = prefix[0]
        else:
            names = ", ".join(layer.name for layer in signal_layers)
            raise ValueError(
                f"Could not uniquely map AOI layer {recipe_layer!r} to ODB SIGNAL layer. "
                f"Derived token={token!r}; SIGNAL layers=[{names}]"
            )

    drills = tuple(layer.name for layer in info.layers if is_drill_layer(layer))
    return ROILayerSelection(signal_layer=signal, drill_layers=drills)


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


def render_roi_cam(job: Path, center_x_mm: float, center_y_mm: float, resolution_um_per_px: float,
                   recipe_layer: str, width_px: int = 100, height_px: int = 100,
                   signal_gv: int = 255, drill_gv: int = 125,
                   visible_steps: Sequence[str] = ("pnl", "strip", "unit")):
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
    result = Image.new("L", (width_px, height_px), color=0)
    result.paste(int(signal_gv), mask=signal_mask.point(lambda v: 255 if v else 0, mode="L"))

    drill_union = Image.new("L", result.size, color=0)
    used_drills: list[str] = []
    for drill_layer in selection.drill_layers:
        mask = _render_layer_mask(renderer, root_step, drill_layer, visible, bounds, width_px, height_px)
        if mask.getbbox() is not None:
            drill_union = ImageChops.lighter(drill_union, mask)
            used_drills.append(drill_layer)
    if drill_union.getbbox() is not None:
        result.paste(int(drill_gv), mask=drill_union.point(lambda v: 255 if v else 0, mode="L"))

    metadata = {
        "center_x_mm": center_x_mm,
        "center_y_mm": center_y_mm,
        "resolution_um_per_px": resolution_um_per_px,
        "size_px": [width_px, height_px],
        "physical_size_mm": [width_px * resolution_um_per_px / 1000.0, height_px * resolution_um_per_px / 1000.0],
        "signal_layer": selection.signal_layer,
        "signal_gv": signal_gv,
        "drill_layers_considered": list(selection.drill_layers),
        "drill_layers_rendered": used_drills,
        "drill_gv": drill_gv,
        "visible_steps": [step.upper() for step in visible],
    }
    return result, metadata
