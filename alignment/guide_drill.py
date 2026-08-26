#!/usr/bin/env python3
"""Guide-drill candidate extraction for the CAM Master-style alignment UI."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

from odb_cam_renderer import Transform, parse_standard_symbol


@dataclass(frozen=True)
class GuideDrillCandidate:
    step: str
    layer: str
    x_in: float
    y_in: float
    diameter_in: float
    polarity: str = "P"

    @property
    def x_mm(self) -> float:
        return self.x_in * 25.4

    @property
    def y_mm(self) -> float:
        return self.y_in * 25.4

    @property
    def diameter_mm(self) -> float:
        return self.diameter_in * 25.4


def is_drill_layer(layer) -> bool:
    name = str(getattr(layer, "name", "")).upper()
    kind = str(getattr(layer, "layer_type", "")).upper()
    return (
        "DRILL" in kind
        or "DRILL" in name
        or name.startswith(("UV_", "TH_", "GDRILL_", "DLD_"))
    )


def drill_layer_names(layers: Iterable) -> List[str]:
    return [str(layer.name) for layer in layers if is_drill_layer(layer)]


def collect_guide_drill_candidates(renderer, root_step: str, layer_names: Sequence[str],
                                   visible_steps: Iterable[str] | None = None) -> List[GuideDrillCandidate]:
    """Collect transformed positive round-pad centers from drill-like layers.

    Coordinates are returned in the root PNL coordinate system. Feature files
    are loaded through FastODBRenderer's existing cache, so this only incurs
    the drill-layer parse cost when alignment mode is actually used.
    """
    layers = tuple(dict.fromkeys(str(name).lower() for name in layer_names if name))
    visible = None if visible_steps is None else {str(step).lower() for step in visible_steps}
    result: List[GuideDrillCandidate] = []

    def collect_step(step: str, transform: Transform, depth: int = 0) -> None:
        if depth > 8:
            return
        step_name = step.lower()
        step_dir = renderer._step_dir(step_name)

        if visible is None or step_name in visible:
            for layer in layers:
                feature_file = step_dir / "layers" / layer / "features"
                symbols, records = renderer._feature_data(feature_file)
                if not records:
                    continue
                for raw in records:
                    tokens = raw.strip().split()
                    if not tokens or tokens[0] != "P" or len(tokens) < 5:
                        continue
                    try:
                        x, y = float(tokens[1]), float(tokens[2])
                        symbol_id = int(tokens[3])
                        polarity = tokens[4].upper()
                    except (ValueError, IndexError):
                        continue
                    if polarity != "P":
                        continue
                    parsed = parse_standard_symbol(symbols.get(symbol_id, ""))
                    if parsed is None or parsed[0] != "round":
                        continue
                    center = transform.apply((x, y))
                    result.append(GuideDrillCandidate(
                        step=step_name,
                        layer=layer,
                        x_in=center[0],
                        y_in=center[1],
                        diameter_in=float(parsed[1]),
                        polarity=polarity,
                    ))

        for repeat in renderer._repeats(step_name):
            for iy in range(repeat.ny):
                for ix in range(repeat.nx):
                    tx = repeat.x + ix * repeat.dx
                    ty = repeat.y + iy * repeat.dy
                    child = transform.compose(
                        renderer._child_transform(repeat.name, tx, ty, repeat.angle, repeat.mirror)
                    )
                    collect_step(repeat.name, child, depth + 1)

    collect_step(root_step.lower(), Transform())
    return result


def nearest_candidate(candidates: Sequence[GuideDrillCandidate], x_in: float, y_in: float,
                      max_distance_in: float | None = None):
    best = None
    best_d2 = None
    for candidate in candidates:
        dx = candidate.x_in - x_in
        dy = candidate.y_in - y_in
        d2 = dx * dx + dy * dy
        if best_d2 is None or d2 < best_d2:
            best, best_d2 = candidate, d2
    if best is None:
        return None
    if max_distance_in is not None and best_d2 is not None and best_d2 > max_distance_in * max_distance_in:
        return None
    return best
