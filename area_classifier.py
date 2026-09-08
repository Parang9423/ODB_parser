from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Mapping, Sequence

AREA_UNK = "AREA_UNK"


@dataclass(frozen=True)
class AreaRule:
    area: str
    threshold_pct: float
    priority: int

    def __post_init__(self) -> None:
        if not self.area:
            raise ValueError("area must not be empty")
        if not 0.0 <= self.threshold_pct <= 100.0:
            raise ValueError("threshold_pct must be between 0 and 100")
        if self.priority < 0:
            raise ValueError("priority must be >= 0")


@dataclass(frozen=True)
class AreaCandidate:
    area: str
    overlap_area: float
    overlap_pct: float
    threshold_pct: float
    priority: int
    eligible: bool


@dataclass(frozen=True)
class AreaClassification:
    final_area: str
    defect_area: float
    candidates: Sequence[AreaCandidate]
    reason: str

    def to_dict(self) -> dict:
        return {
            "final_area": self.final_area,
            "defect_area": self.defect_area,
            "reason": self.reason,
            "candidates": [asdict(c) for c in self.candidates],
        }


def classify_area_from_intersections(
    defect_area: float,
    intersection_areas: Mapping[str, float],
    rules: Iterable[AreaRule],
) -> AreaClassification:
    """Classify one defect instance using overlap thresholds, then area priority.

    overlap_pct = intersection area / defect contour area * 100.
    Percentage only determines whether an area is eligible. Among eligible areas,
    the lowest numeric priority wins. If priorities tie, larger overlap wins and
    area name is the deterministic final tie-breaker.

    `intersection_areas` should contain the unioned intersection area per business
    area (PAD/CIRCUIT/SPACE/...). Do not sum overlapping raw ODB features directly,
    because that can double-count pixels/geometry within the same business area.
    """
    if defect_area <= 0:
        return AreaClassification(AREA_UNK, defect_area, (), "invalid_defect_area")

    rule_map: Dict[str, AreaRule] = {}
    for rule in rules:
        if rule.area in rule_map:
            raise ValueError(f"duplicate area rule: {rule.area}")
        rule_map[rule.area] = rule

    candidates: List[AreaCandidate] = []
    for area, rule in rule_map.items():
        overlap_area = max(0.0, float(intersection_areas.get(area, 0.0)))
        # Geometry tolerances can make an intersection microscopically larger.
        overlap_pct = min(100.0, overlap_area / defect_area * 100.0)
        candidates.append(
            AreaCandidate(
                area=area,
                overlap_area=overlap_area,
                overlap_pct=overlap_pct,
                threshold_pct=rule.threshold_pct,
                priority=rule.priority,
                eligible=overlap_pct >= rule.threshold_pct,
            )
        )

    candidates.sort(key=lambda c: (c.priority, -c.overlap_pct, c.area))
    eligible = [c for c in candidates if c.eligible]
    if not eligible:
        return AreaClassification(AREA_UNK, defect_area, tuple(candidates), "no_area_passed_threshold")

    winner = eligible[0]
    return AreaClassification(winner.area, defect_area, tuple(candidates), "threshold_pass_then_priority")


def classify_area_from_polygons(defect_polygon, area_polygons: Mapping[str, object], rules: Iterable[AreaRule]) -> AreaClassification:
    """Shapely-compatible polygon adapter.

    Each value in `area_polygons` is expected to expose `.intersection()` and
    `.area`. Callers should union all ODB features belonging to the same business
    area before passing them here.
    """
    defect_area = float(defect_polygon.area)
    intersections: Dict[str, float] = {}
    for area, polygon in area_polygons.items():
        intersections[area] = float(defect_polygon.intersection(polygon).area)
    return classify_area_from_intersections(defect_area, intersections, rules)
