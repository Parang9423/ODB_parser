from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

from odb.feature_query import ODBVectorFeature

Bounds = Tuple[float, float, float, float]


@dataclass(frozen=True)
class EffectiveIntersection:
    geometry: object
    area_mm2: float
    overlap_pct: float
    residual_area_mm2: float
    residual_pct: float
    positive_hit_count: int
    negative_hit_count: int
    applied_feature_ids: tuple[str, ...]


def _bounds_overlap(a: Bounds, b: Bounds) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


class EffectiveLayerGeometry:
    """Compose ordered ODB primitives inside a defect polygon.

    Composition follows renderer semantics in feature order:
    positive geometry adds material, negative geometry removes material.
    Geometry is clipped to the defect polygon first so a full-panel union is not
    required for each query.
    """

    def __init__(self, features: Sequence[ODBVectorFeature]):
        self.features = tuple(features)

    def intersect(self, defect_polygon) -> EffectiveIntersection:
        if defect_polygon is None or defect_polygon.is_empty or defect_polygon.area <= 0:
            raise ValueError("defect polygon must have positive area")

        try:
            from shapely.geometry import GeometryCollection
        except ImportError as exc:
            raise RuntimeError("Shapely is required for effective ODB geometry") from exc

        defect_area = float(defect_polygon.area)
        defect_bounds = tuple(float(v) for v in defect_polygon.bounds)
        effective = GeometryCollection()
        positive_hits = 0
        negative_hits = 0
        applied: list[str] = []

        for feature in self.features:
            if not _bounds_overlap(defect_bounds, feature.bounds):
                continue
            clipped = defect_polygon.intersection(feature.geometry)
            if clipped.is_empty or clipped.area <= 0:
                continue

            polarity = feature.polarity.upper()
            if polarity == "P":
                effective = effective.union(clipped)
                positive_hits += 1
            elif polarity == "N":
                effective = effective.difference(clipped)
                negative_hits += 1
            else:
                continue
            applied.append(feature.feature_id)

        if not effective.is_empty and not effective.is_valid:
            effective = effective.buffer(0)

        area = max(0.0, min(defect_area, float(effective.area)))
        overlap_pct = min(100.0, area / defect_area * 100.0)
        residual_area = max(0.0, defect_area - area)
        residual_pct = max(0.0, 100.0 - overlap_pct)

        return EffectiveIntersection(
            geometry=effective,
            area_mm2=area,
            overlap_pct=overlap_pct,
            residual_area_mm2=residual_area,
            residual_pct=residual_pct,
            positive_hit_count=positive_hits,
            negative_hit_count=negative_hits,
            applied_feature_ids=tuple(applied),
        )
