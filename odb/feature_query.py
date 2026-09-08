from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

from hierarchy_renderer import FastODBRenderer, StepInstance
from odb_cam_renderer import arc_points, parse_standard_symbol, round_symbol_diameter_in

Bounds = Tuple[float, float, float, float]


@dataclass(frozen=True)
class ODBVectorFeature:
    """One positive ODB primitive flattened into the root-step coordinate frame."""

    feature_id: str
    layer: str
    step: str
    depth: int
    primitive_type: str
    symbol: str | None
    polarity: str
    geometry: object

    @property
    def bounds(self) -> Bounds:
        return tuple(float(v) for v in self.geometry.bounds)  # type: ignore[return-value]


@dataclass(frozen=True)
class FeatureIntersection:
    feature: ODBVectorFeature
    intersection_geometry: object
    intersection_area: float
    defect_overlap_pct: float


def _shapely():
    try:
        from shapely.geometry import LineString, Point, Polygon
        from shapely.ops import unary_union
    except ImportError as exc:
        raise RuntimeError("Shapely is required for ODB vector feature queries") from exc
    return Point, LineString, Polygon, unary_union


def _transform_polygon(points, transform):
    _, _, Polygon, _ = _shapely()
    return Polygon([transform.apply(point) for point in points])


def _pad_geometry(x: float, y: float, symbol: str, rotation_deg: float, transform):
    Point, _, Polygon, _ = _shapely()
    parsed = parse_standard_symbol(symbol)
    if parsed is None:
        return None
    kind, width, height = parsed
    if kind == "round":
        center = Point(transform.apply((x, y)))
        # Shapely's buffer approximates the round aperture. quad_segs is kept
        # reasonably high because these geometries feed area-overlap decisions.
        return center.buffer(width / 2.0, quad_segs=16)

    from shapely.affinity import rotate
    local = Polygon([
        (x - width / 2.0, y - height / 2.0),
        (x + width / 2.0, y - height / 2.0),
        (x + width / 2.0, y + height / 2.0),
        (x - width / 2.0, y + height / 2.0),
    ])
    if rotation_deg:
        local = rotate(local, rotation_deg, origin=(x, y), use_radians=False)
    return Polygon([transform.apply(point) for point in local.exterior.coords])


def _line_geometry(x1: float, y1: float, x2: float, y2: float, diameter: float, transform):
    _, LineString, _, _ = _shapely()
    line = LineString([transform.apply((x1, y1)), transform.apply((x2, y2))])
    return line.buffer(diameter / 2.0, cap_style=1, join_style=1, quad_segs=16)


def _surface_geometry(contours):
    """Build an ODB surface using I contours as solids and H contours as holes."""
    _, _, Polygon, unary_union = _shapely()
    islands = []
    holes = []
    for kind, points in contours:
        if len(points) < 3:
            continue
        poly = Polygon(points)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty:
            continue
        if str(kind).upper().startswith("H"):
            holes.append(poly)
        else:
            islands.append(poly)
    if not islands:
        return None
    geom = unary_union(islands)
    if holes:
        geom = geom.difference(unary_union(holes))
    return geom if not geom.is_empty else None


def extract_vector_features(
    renderer: FastODBRenderer,
    root_step: str,
    layers: Sequence[str],
    visible_steps: Iterable[str] | None = None,
    positive_only: bool = True,
) -> List[ODBVectorFeature]:
    """Flatten supported ODB P/L/S primitives into root-step coordinates.

    This is deliberately read-only and reuses FastODBRenderer's hierarchy and
    feature caches. Unsupported symbols/records are skipped rather than guessed.
    """
    wanted_layers = tuple(dict.fromkeys(str(layer).lower() for layer in layers if layer))
    visible = None if visible_steps is None else {str(step).lower() for step in visible_steps}
    features: List[ODBVectorFeature] = []

    for instance_index, instance in enumerate(renderer.collect_instances(root_step)):
        if visible is not None and instance.step not in visible:
            continue
        step_dir = renderer._step_dir(instance.step)
        for layer in wanted_layers:
            symbols, records = renderer._feature_data(step_dir / "layers" / layer / "features")
            index = 0
            record_index = 0
            while index < len(records):
                raw = records[index].strip()
                index += 1
                if not raw or raw.startswith("#") or raw.startswith("$") or raw == "SE":
                    continue
                tokens = raw.split()
                command = tokens[0].upper()
                record_index += 1
                polarity = "P"
                symbol = None
                geometry = None
                try:
                    if command == "P" and len(tokens) >= 5:
                        x, y = float(tokens[1]), float(tokens[2])
                        symbol_id = int(tokens[3])
                        polarity = tokens[4].upper()
                        symbol = symbols.get(symbol_id, "")
                        rotation = float(tokens[5]) if len(tokens) > 5 else 0.0
                        geometry = _pad_geometry(x, y, symbol, rotation, instance.transform)
                    elif command == "L" and len(tokens) >= 7:
                        x1, y1, x2, y2 = map(float, tokens[1:5])
                        symbol_id = int(tokens[5])
                        polarity = tokens[6].upper()
                        symbol = symbols.get(symbol_id, "")
                        diameter = round_symbol_diameter_in(symbol)
                        if diameter is not None:
                            geometry = _line_geometry(x1, y1, x2, y2, diameter, instance.transform)
                    elif command == "S" and len(tokens) >= 2:
                        polarity = tokens[1].upper()
                        contours = []
                        current = None
                        current_kind = "I"
                        while index < len(records):
                            surface_raw = records[index].strip()
                            index += 1
                            if not surface_raw or surface_raw.startswith("#"):
                                continue
                            values = surface_raw.split()
                            cmd = values[0].upper()
                            if cmd == "OB" and len(values) >= 4:
                                current = [(float(values[1]), float(values[2]))]
                                current_kind = values[3].upper()
                            elif cmd == "OS" and current is not None:
                                current.append((float(values[1]), float(values[2])))
                            elif cmd == "OC" and current is not None and len(values) >= 6:
                                end = (float(values[1]), float(values[2]))
                                center = (float(values[3]), float(values[4]))
                                current.extend(arc_points(current[-1], end, center, values[5].upper().startswith("Y")))
                            elif cmd == "OE":
                                if current:
                                    contours.append((current_kind, [instance.transform.apply(p) for p in current]))
                                current = None
                            elif cmd == "SE":
                                break
                        geometry = _surface_geometry(contours)
                    else:
                        continue
                except (ValueError, IndexError):
                    continue

                if geometry is None or geometry.is_empty:
                    continue
                if positive_only and polarity != "P":
                    continue
                features.append(ODBVectorFeature(
                    feature_id=f"{instance.step}:{instance_index}:{layer}:{record_index}",
                    layer=layer,
                    step=instance.step,
                    depth=instance.depth,
                    primitive_type=command,
                    symbol=symbol or None,
                    polarity=polarity,
                    geometry=geometry,
                ))
    return features


def _bounds_overlap(a: Bounds, b: Bounds) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


class DefectContourQuery:
    """Exact contour-to-feature intersection query.

    Bounds checks are only a candidate prefilter. A feature is returned only
    when its geometry has a non-empty exact intersection with the defect.
    """

    def __init__(self, features: Sequence[ODBVectorFeature]):
        self.features = tuple(features)

    def query(self, defect_polygon) -> List[FeatureIntersection]:
        if defect_polygon is None or defect_polygon.is_empty or defect_polygon.area <= 0:
            raise ValueError("defect polygon must have positive area")
        defect_area = float(defect_polygon.area)
        defect_bounds = tuple(float(v) for v in defect_polygon.bounds)
        hits: List[FeatureIntersection] = []
        for feature in self.features:
            if not _bounds_overlap(defect_bounds, feature.bounds):
                continue
            intersection = defect_polygon.intersection(feature.geometry)
            if intersection.is_empty:
                continue
            area = float(intersection.area)
            # Boundary-only contact is not an affected area for the current
            # business-area overlap policy. Keep exact positive-area hits only.
            if area <= 0:
                continue
            hits.append(FeatureIntersection(
                feature=feature,
                intersection_geometry=intersection,
                intersection_area=area,
                defect_overlap_pct=min(100.0, area / defect_area * 100.0),
            ))
        return hits
