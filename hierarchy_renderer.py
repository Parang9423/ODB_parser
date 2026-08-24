#!/usr/bin/env python3
"""Fast hierarchy-aware helpers for ODB++ CAM preview rendering.

Nested STEP-REPEAT placement is datum-aware: the X/Y in a parent STEP-REPEAT
locates the child step datum, not necessarily the child profile's local (0, 0).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image, ImageChops

from odb_cam_renderer import (
    CompositeLayer,
    ODBError,
    ODBRenderer,
    RasterCanvas,
    Transform,
    arc_points,
    contours_bounds,
    parse_profile_contours,
    parse_repeats,
    parse_standard_symbol,
    repeat_transform,
    round_symbol_diameter_in,
)

Point = Tuple[float, float]
ProfileContours = List[Tuple[str, List[Point]]]


@dataclass(frozen=True)
class StepInstance:
    step: str
    depth: int
    transform: Transform


@dataclass(frozen=True)
class StepFrame:
    """Coordinate metadata declared in a step stephdr.

    X_DATUM/Y_DATUM are the local coordinates of the step datum used by
    STEP-REPEAT placement. X_ORIGIN/Y_ORIGIN are retained for diagnostics;
    they are not blindly applied as placement offsets.
    """
    x_datum: float = 0.0
    y_datum: float = 0.0
    x_origin: float = 0.0
    y_origin: float = 0.0


@dataclass(frozen=True)
class CacheSnapshot:
    feature_hits: int
    feature_misses: int
    repeat_hits: int
    repeat_misses: int
    profile_hits: int
    profile_misses: int


def _file_key(path: Path) -> Tuple[str, int, int]:
    stat = path.stat()
    return str(path), stat.st_mtime_ns, stat.st_size


@lru_cache(maxsize=256)
def _cached_feature_data(path_str: str, _mtime_ns: int, _size: int) -> Tuple[Dict[int, str], Tuple[str, ...]]:
    path = Path(path_str)
    records = tuple(path.read_text(errors="replace").splitlines())
    symbols: Dict[int, str] = {}
    for raw in records:
        value = raw.strip()
        if value.startswith("$"):
            match = re.match(r"\$(\d+)\s+(\S+)", value)
            if match:
                symbols[int(match.group(1))] = match.group(2)
    return symbols, records


@lru_cache(maxsize=128)
def _cached_repeats(path_str: str, _mtime_ns: int, _size: int):
    return tuple(parse_repeats(Path(path_str)))


@lru_cache(maxsize=128)
def _cached_profile(path_str: str, _mtime_ns: int, _size: int):
    contours = parse_profile_contours(Path(path_str))
    return tuple((kind, tuple(points)) for kind, points in contours)


@lru_cache(maxsize=128)
def _cached_step_frame(path_str: str, _mtime_ns: int, _size: int) -> StepFrame:
    """Read top-level datum/origin fields without consuming STEP-REPEAT blocks."""
    text = Path(path_str).read_text(errors="replace")
    # Remove blocks so NAME/X/Y inside STEP-REPEAT cannot be mistaken for step metadata.
    top = re.sub(r"STEP-REPEAT\s*\{.*?\}", "", text, flags=re.S | re.I)
    values: Dict[str, float] = {}
    for raw in top.splitlines():
        raw = raw.strip()
        if not raw or raw.startswith("#") or "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        key = key.strip().upper()
        if key not in {"X_DATUM", "Y_DATUM", "X_ORIGIN", "Y_ORIGIN"}:
            continue
        try:
            values[key] = float(value.strip())
        except ValueError:
            pass
    return StepFrame(
        values.get("X_DATUM", 0.0),
        values.get("Y_DATUM", 0.0),
        values.get("X_ORIGIN", 0.0),
        values.get("Y_ORIGIN", 0.0),
    )


def cached_repeats(path: Path):
    if not path.exists():
        return ()
    return _cached_repeats(*_file_key(path))


def cached_profile(path: Path) -> ProfileContours:
    if not path.exists():
        raise ODBError(f"Missing profile: {path}")
    frozen = _cached_profile(*_file_key(path))
    return [(kind, list(points)) for kind, points in frozen]


def cached_step_frame(path: Path) -> StepFrame:
    if not path.exists():
        return StepFrame()
    return _cached_step_frame(*_file_key(path))


def cache_snapshot() -> CacheSnapshot:
    feature = _cached_feature_data.cache_info()
    repeat = _cached_repeats.cache_info()
    profile = _cached_profile.cache_info()
    return CacheSnapshot(feature.hits, feature.misses, repeat.hits, repeat.misses, profile.hits, profile.misses)


def clear_render_caches() -> None:
    _cached_feature_data.cache_clear()
    _cached_repeats.cache_clear()
    _cached_profile.cache_clear()
    _cached_step_frame.cache_clear()


def adaptive_preview_dpi(bounds: Tuple[float, float, float, float], requested_dpi: float,
                         max_pixels: int = 12_000_000, min_dpi: float = 72.0) -> float:
    xmin, ymin, xmax, ymax = bounds
    width_in = max(1e-12, xmax - xmin)
    height_in = max(1e-12, ymax - ymin)
    requested = max(float(min_dpi), float(requested_dpi))
    estimated_pixels = width_in * height_in * requested * requested
    if estimated_pixels <= max_pixels:
        return requested
    capped = math.sqrt(max_pixels / (width_in * height_in))
    return max(float(min_dpi), min(requested, capped))


class FastODBRenderer(ODBRenderer):
    """ODBRenderer with shared caches, hierarchy filtering and datum-aware placement."""

    def _repeats(self, step: str):
        return cached_repeats(self._step_dir(step) / "stephdr")

    def step_frame(self, step: str) -> StepFrame:
        return cached_step_frame(self._step_dir(step) / "stephdr")

    def profile(self, step: str) -> ProfileContours:
        return cached_profile(self._step_dir(step) / "profile")

    def profile_bounds(self, step: str) -> Tuple[float, float, float, float]:
        return contours_bounds(self.profile(step))

    def _child_transform(self, child_step: str, x: float, y: float,
                         angle: float, mirror: bool) -> Transform:
        """Map child local coordinates into its parent STEP coordinate system.

        ODB STEP-REPEAT X/Y specifies where the child DATUM is placed. Therefore
        local coordinates must first be shifted by -child_datum, then mirrored /
        rotated, and finally translated to X/Y.
        """
        frame = self.step_frame(child_step)
        placement = repeat_transform(x, y, angle, mirror)
        local_to_datum = Transform(tx=-frame.x_datum, ty=-frame.y_datum)
        return placement.compose(local_to_datum)

    def coordinate_diagnostics(self, root_step: str) -> List[str]:
        rows: List[str] = []
        for step in (root_step.lower(), "strip", "unit"):
            try:
                frame = self.step_frame(step)
                bounds = self.profile_bounds(step)
            except ODBError:
                continue
            rows.append(
                f"{step.upper()}: datum=({frame.x_datum:.6f},{frame.y_datum:.6f}) "
                f"origin=({frame.x_origin:.6f},{frame.y_origin:.6f}) "
                f"profile=({bounds[0]:.6f},{bounds[1]:.6f})-({bounds[2]:.6f},{bounds[3]:.6f})"
            )
        return rows

    def _feature_data(self, feature_file: Path) -> Tuple[Dict[int, str], Tuple[str, ...]]:
        if not feature_file.exists():
            return {}, ()
        return _cached_feature_data(*_file_key(feature_file))

    def _render_feature_file(self, canvas: RasterCanvas, feature_file: Path, transform: Transform) -> None:
        symbols, records = self._feature_data(feature_file)
        if not records:
            return
        index = 0
        while index < len(records):
            record = records[index].strip(); index += 1
            if not record or record.startswith("#") or record.startswith("$") or record == "SE":
                continue
            tokens = record.split(); command = tokens[0]
            try:
                if command == "P" and len(tokens) >= 6:
                    x, y = float(tokens[1]), float(tokens[2])
                    symbol_id, polarity = int(tokens[3]), tokens[4].upper()
                    parsed = parse_standard_symbol(symbols.get(symbol_id, ""))
                    if parsed is None:
                        self._warn(f"Unsupported P symbol {symbols.get(symbol_id)!r}"); continue
                    kind, width, height = parsed
                    value = 255 if polarity == "P" else 0
                    rotation = float(tokens[5]) if len(tokens) > 5 else 0.0
                    if kind == "round":
                        canvas.draw_round_pad(transform.apply((x, y)), width, value)
                    else:
                        canvas.draw_rect_pad((x, y), width, height, value, rotation, transform)
                    self.stats.pads += 1
                elif command == "L" and len(tokens) >= 8:
                    x1, y1, x2, y2 = map(float, tokens[1:5])
                    symbol_id, polarity = int(tokens[5]), tokens[6].upper()
                    diameter = round_symbol_diameter_in(symbols.get(symbol_id, ""))
                    if diameter is None:
                        self._warn(f"Unsupported L symbol {symbols.get(symbol_id)!r}"); continue
                    canvas.draw_round_line(transform.apply((x1, y1)), transform.apply((x2, y2)), diameter,
                                           255 if polarity == "P" else 0)
                    self.stats.lines += 1
                elif command == "S" and len(tokens) >= 2:
                    polarity = tokens[1].upper()
                    contours: List[Tuple[str, List[Point]]] = []
                    current: Optional[List[Point]] = None; current_kind = "I"
                    while index < len(records):
                        surface_record = records[index].strip(); index += 1
                        if not surface_record or surface_record.startswith("#"): continue
                        values = surface_record.split(); cmd = values[0]
                        if cmd == "OB" and len(values) >= 4:
                            current = [(float(values[1]), float(values[2]))]; current_kind = values[3].upper()
                        elif cmd == "OS" and current is not None:
                            current.append((float(values[1]), float(values[2])))
                        elif cmd == "OC" and current is not None and len(values) >= 6:
                            end = float(values[1]), float(values[2]); center = float(values[3]), float(values[4])
                            current.extend(arc_points(current[-1], end, center, values[5].upper().startswith("Y")))
                        elif cmd == "OE":
                            if current: contours.append((current_kind, [transform.apply(point) for point in current]))
                            current = None
                        elif cmd == "SE": break
                        else: self._warn(f"Unsupported surface record {surface_record[:80]}")
                    canvas.draw_surface(contours, polarity); self.stats.surfaces += 1
                else:
                    self._warn(f"Unsupported feature record {record[:100]}")
            except (ValueError, IndexError) as exc:
                self._warn(f"Parse error: {record[:100]} ({exc})")

    def _render_step_recursive(self, canvas: RasterCanvas, step: str, layer: str,
                               parent_transform: Transform, depth: int = 0) -> None:
        if depth > 8: raise ODBError("STEP-REPEAT recursion too deep")
        step_name = step.lower(); step_dir = self._step_dir(step_name)
        for repeat in self._repeats(step_name):
            for iy in range(repeat.ny):
                for ix in range(repeat.nx):
                    tx = repeat.x + ix * repeat.dx; ty = repeat.y + iy * repeat.dy
                    child_transform = parent_transform.compose(
                        self._child_transform(repeat.name, tx, ty, repeat.angle, repeat.mirror)
                    )
                    self._render_step_recursive(canvas, repeat.name, layer, child_transform, depth + 1)
                    self.stats.repeats += 1
        self._render_feature_file(canvas, step_dir / "layers" / layer.lower() / "features", parent_transform)

    def _render_step_filtered(self, canvas: RasterCanvas, step: str, layer: str,
                              parent_transform: Transform, visible_steps: set[str], depth: int = 0) -> None:
        if depth > 8: raise ODBError("STEP-REPEAT recursion too deep")
        step_name = step.lower(); step_dir = self._step_dir(step_name)
        for repeat in self._repeats(step_name):
            for iy in range(repeat.ny):
                for ix in range(repeat.nx):
                    tx = repeat.x + ix * repeat.dx; ty = repeat.y + iy * repeat.dy
                    child_transform = parent_transform.compose(
                        self._child_transform(repeat.name, tx, ty, repeat.angle, repeat.mirror)
                    )
                    self._render_step_filtered(canvas, repeat.name, layer, child_transform, visible_steps, depth + 1)
                    self.stats.repeats += 1
        if step_name in visible_steps:
            self._render_feature_file(canvas, step_dir / "layers" / layer.lower() / "features", parent_transform)

    def render(self, step: str, layer: str, margin_px: int = 0) -> Image.Image:
        canvas = RasterCanvas(self.profile_bounds(step), self.dpi_x, self.dpi_y, margin_px=margin_px, background=0)
        self._render_step_recursive(canvas, step.lower(), layer.lower(), Transform())
        return canvas.image

    def render_hierarchy(self, root_step: str, layer: str, visible_steps: Iterable[str], margin_px: int = 0) -> Image.Image:
        visible = {name.lower() for name in visible_steps}
        if not visible: raise ValueError("At least one visible Step is required")
        canvas = RasterCanvas(self.profile_bounds(root_step), self.dpi_x, self.dpi_y, margin_px=margin_px, background=0)
        self._render_step_filtered(canvas, root_step.lower(), layer.lower(), Transform(), visible)
        return canvas.image

    def render_composite_hierarchy(self, root_step: str, layers: Sequence[CompositeLayer],
                                   visible_steps: Iterable[str], margin_px: int = 0,
                                   background: int = 0) -> Image.Image:
        specs = [spec.normalized() for spec in layers]
        if not specs: raise ValueError("At least one composite layer is required")
        result: Optional[Image.Image] = None
        for spec in specs:
            mask_image = self.render_hierarchy(root_step, spec.layer, visible_steps, margin_px)
            mask = mask_image.point(lambda value: 255 if value > 0 else 0, mode="L")
            if result is None: result = Image.new("L", mask_image.size, color=int(background))
            if spec.operation == "REPLACE": result.paste(spec.gv, mask=mask)
            elif spec.operation == "SUBTRACT": result.paste(0, mask=mask)
            else:
                contribution = Image.new("L", result.size, color=0); contribution.paste(spec.gv, mask=mask)
                result = ImageChops.lighter(result, contribution)
        assert result is not None
        return result

    def collect_instances(self, root_step: str) -> List[StepInstance]:
        instances: List[StepInstance] = []
        def walk(step: str, transform: Transform, depth: int) -> None:
            if depth > 8: raise ODBError("STEP-REPEAT recursion too deep")
            step_name = step.lower(); instances.append(StepInstance(step_name, depth, transform))
            for repeat in self._repeats(step_name):
                for iy in range(repeat.ny):
                    for ix in range(repeat.nx):
                        tx = repeat.x + ix * repeat.dx; ty = repeat.y + iy * repeat.dy
                        child = transform.compose(self._child_transform(repeat.name, tx, ty, repeat.angle, repeat.mirror))
                        walk(repeat.name, child, depth + 1)
        walk(root_step.lower(), Transform(), 0)
        return instances

    def transformed_profile(self, instance: StepInstance) -> ProfileContours:
        return [(kind, [instance.transform.apply(point) for point in points])
                for kind, points in self.profile(instance.step)]
