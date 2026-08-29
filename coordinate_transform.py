from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True)
class CalibrationPoint:
    aoi_x: float
    aoi_y: float
    odb_x: float
    odb_y: float


@dataclass(frozen=True)
class AffineTransform:
    """AOI coordinate -> ODB rendered-image pixel affine transform.

    odb_x = a * aoi_x + b * aoi_y + c
    odb_y = d * aoi_x + e * aoi_y + f
    """

    a: float
    b: float
    c: float
    d: float
    e: float
    f: float
    rmse: float = 0.0
    point_count: int = 0

    def apply(self, x: float, y: float) -> tuple[float, float]:
        return (
            self.a * x + self.b * y + self.c,
            self.d * x + self.e * y + self.f,
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "AffineTransform":
        return cls(**data)


def _solve_3x3(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> tuple[float, float, float]:
    aug = [list(map(float, row)) + [float(rhs)] for row, rhs in zip(matrix, vector)]
    n = 3

    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(aug[row][col]))
        if abs(aug[pivot][col]) < 1e-12:
            raise ValueError("Calibration points are degenerate. Use points spread across the ODB image.")
        if pivot != col:
            aug[col], aug[pivot] = aug[pivot], aug[col]

        pivot_value = aug[col][col]
        aug[col] = [v / pivot_value for v in aug[col]]

        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col]
            if abs(factor) < 1e-15:
                continue
            aug[row] = [a - factor * b for a, b in zip(aug[row], aug[col])]

    return aug[0][3], aug[1][3], aug[2][3]


def fit_affine(points: Iterable[CalibrationPoint]) -> AffineTransform:
    pts = list(points)
    if len(pts) < 3:
        raise ValueError("At least 3 calibration points are required.")

    sx = sy = sxx = syy = sxy = 0.0
    su = sv = sxu = syu = sxv = syv = 0.0

    for p in pts:
        x, y, u, v = p.aoi_x, p.aoi_y, p.odb_x, p.odb_y
        sx += x
        sy += y
        sxx += x * x
        syy += y * y
        sxy += x * y
        su += u
        sv += v
        sxu += x * u
        syu += y * u
        sxv += x * v
        syv += y * v

    normal = (
        (sxx, sxy, sx),
        (sxy, syy, sy),
        (sx, sy, float(len(pts))),
    )
    a, b, c = _solve_3x3(normal, (sxu, syu, su))
    d, e, f = _solve_3x3(normal, (sxv, syv, sv))

    squared_error = 0.0
    for p in pts:
        pu = a * p.aoi_x + b * p.aoi_y + c
        pv = d * p.aoi_x + e * p.aoi_y + f
        squared_error += (pu - p.odb_x) ** 2 + (pv - p.odb_y) ** 2
    rmse = math.sqrt(squared_error / len(pts))

    return AffineTransform(a, b, c, d, e, f, rmse=rmse, point_count=len(pts))


_COORD_PATTERNS = (
    re.compile(
        r"(?i)(?:^|[^a-z0-9])x\s*[_:=]?\s*(-?\d+(?:\.\d+)?)"
        r".*?(?:^|[^a-z0-9])y\s*[_:=]?\s*(-?\d+(?:\.\d+)?)"
    ),
    re.compile(
        r"(?i)(?:^|[^a-z0-9])y\s*[_:=]?\s*(-?\d+(?:\.\d+)?)"
        r".*?(?:^|[^a-z0-9])x\s*[_:=]?\s*(-?\d+(?:\.\d+)?)"
    ),
)


def extract_xy_from_filename(path_or_name: str | Path) -> tuple[float, float] | None:
    name = Path(path_or_name).stem
    first = _COORD_PATTERNS[0].search(name)
    if first:
        return float(first.group(1)), float(first.group(2))

    second = _COORD_PATTERNS[1].search(name)
    if second:
        return float(second.group(2)), float(second.group(1))
    return None


def save_calibration(
    path: str | Path,
    transform: AffineTransform,
    points: Sequence[CalibrationPoint],
    metadata: dict | None = None,
) -> None:
    payload = {
        "version": 1,
        "transform": transform.to_dict(),
        "points": [asdict(p) for p in points],
        "metadata": metadata or {},
    }
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_calibration(path: str | Path) -> tuple[AffineTransform, list[CalibrationPoint], dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    transform = AffineTransform.from_dict(payload["transform"])
    points = [CalibrationPoint(**item) for item in payload.get("points", [])]
    return transform, points, payload.get("metadata", {})
