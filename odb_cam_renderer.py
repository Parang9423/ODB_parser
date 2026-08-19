#!/usr/bin/env python3
"""ODB++ CAM raster renderer for AOI reference-image generation."""
from __future__ import annotations

import argparse
import math
import re
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageChops, ImageDraw

Point = Tuple[float, float]


@dataclass(frozen=True)
class Transform:
    a: float = 1.0
    b: float = 0.0
    c: float = 0.0
    d: float = 1.0
    tx: float = 0.0
    ty: float = 0.0

    def apply(self, p: Point) -> Point:
        x, y = p
        return self.a * x + self.b * y + self.tx, self.c * x + self.d * y + self.ty

    def compose(self, child: "Transform") -> "Transform":
        return Transform(
            a=self.a * child.a + self.b * child.c,
            b=self.a * child.b + self.b * child.d,
            c=self.c * child.a + self.d * child.c,
            d=self.c * child.b + self.d * child.d,
            tx=self.a * child.tx + self.b * child.ty + self.tx,
            ty=self.c * child.tx + self.d * child.ty + self.ty,
        )


@dataclass
class Repeat:
    name: str
    x: float
    y: float
    dx: float
    dy: float
    nx: int
    ny: int
    angle: float
    mirror: bool


@dataclass(frozen=True)
class CompositeLayer:
    layer: str
    operation: str = "ADD"
    gv: int = 255

    def normalized(self) -> "CompositeLayer":
        op = self.operation.upper()
        if op not in {"ADD", "REPLACE", "SUBTRACT"}:
            raise ValueError(f"Unsupported composite operation: {self.operation}")
        gv = int(self.gv)
        if not 0 <= gv <= 255:
            raise ValueError("Composite GV must be between 0 and 255")
        return CompositeLayer(self.layer, op, gv)


@dataclass
class RenderStats:
    pads: int = 0
    lines: int = 0
    surfaces: int = 0
    repeats: int = 0
    unsupported: int = 0


class ODBError(RuntimeError):
    pass


def parse_kv_blocks(path: Path, block_name: str) -> List[Dict[str, str]]:
    text = path.read_text(errors="replace")
    blocks: List[Dict[str, str]] = []
    pattern = re.compile(rf"{re.escape(block_name)}\s*\{{(.*?)\}}", re.S | re.I)
    for match in pattern.finditer(text):
        block: Dict[str, str] = {}
        for raw in match.group(1).splitlines():
            raw = raw.strip()
            if not raw or raw.startswith("#") or "=" not in raw:
                continue
            key, value = raw.split("=", 1)
            block[key.strip().upper()] = value.strip()
        blocks.append(block)
    return blocks


def parse_repeats(stephdr: Path) -> List[Repeat]:
    if not stephdr.exists():
        return []
    return [
        Repeat(
            name=b.get("NAME", "").lower(),
            x=float(b.get("X", 0)), y=float(b.get("Y", 0)),
            dx=float(b.get("DX", 0)), dy=float(b.get("DY", 0)),
            nx=int(b.get("NX", 1)), ny=int(b.get("NY", 1)),
            angle=float(b.get("ANGLE", 0)),
            mirror=b.get("MIRROR", "NO").upper() == "YES",
        )
        for b in parse_kv_blocks(stephdr, "STEP-REPEAT")
    ]


def repeat_transform(x: float, y: float, angle_deg: float, mirror: bool) -> Transform:
    mirror_x = -1.0 if mirror else 1.0
    theta = math.radians(angle_deg % 360)
    cs, sn = math.cos(theta), math.sin(theta)
    return Transform(a=cs * mirror_x, b=-sn, c=sn * mirror_x, d=cs, tx=x, ty=y)


def arc_points(start: Point, end: Point, center: Point, clockwise: bool, max_step_deg: float = 4.0) -> List[Point]:
    sx, sy = start
    ex, ey = end
    cx, cy = center
    r0 = math.hypot(sx - cx, sy - cy)
    r1 = math.hypot(ex - cx, ey - cy)
    if r0 < 1e-12 or abs(r0 - r1) > max(1e-5, r0 * 0.02):
        return [end]
    a0 = math.atan2(sy - cy, sx - cx)
    a1 = math.atan2(ey - cy, ex - cx)
    twopi = 2 * math.pi
    if clockwise:
        delta = -((a0 - a1) % twopi)
        if abs(delta) < 1e-12 and start != end:
            delta = -twopi
    else:
        delta = (a1 - a0) % twopi
        if abs(delta) < 1e-12 and start != end:
            delta = twopi
    count = max(1, int(math.ceil(abs(math.degrees(delta)) / max_step_deg)))
    return [
        (cx + r0 * math.cos(a0 + delta * i / count), cy + r0 * math.sin(a0 + delta * i / count))
        for i in range(1, count + 1)
    ]


def parse_profile_contours(profile: Path) -> List[Tuple[str, List[Point]]]:
    if not profile.exists():
        raise ODBError(f"Missing profile: {profile}")
    contours: List[Tuple[str, List[Point]]] = []
    current: Optional[List[Point]] = None
    kind = "I"
    for raw in profile.read_text(errors="replace").splitlines():
        value = raw.strip()
        if not value or value.startswith("#"):
            continue
        tokens = value.split()
        if tokens[0] == "OB" and len(tokens) >= 4:
            current = [(float(tokens[1]), float(tokens[2]))]
            kind = tokens[3].upper()
        elif tokens[0] == "OS" and current is not None:
            current.append((float(tokens[1]), float(tokens[2])))
        elif tokens[0] == "OC" and current is not None:
            end = float(tokens[1]), float(tokens[2])
            center = float(tokens[3]), float(tokens[4])
            current.extend(arc_points(current[-1], end, center, tokens[5].upper().startswith("Y")))
        elif tokens[0] == "OE" and current:
            contours.append((kind, current))
            current = None
    return contours


def contours_bounds(contours: Sequence[Tuple[str, Sequence[Point]]]) -> Tuple[float, float, float, float]:
    points = [point for _, contour in contours for point in contour]
    if not points:
        raise ODBError("Profile contains no contour points")
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def parse_symbol_table(feature_file: Path) -> Dict[int, str]:
    symbols: Dict[int, str] = {}
    for raw in feature_file.read_text(errors="replace").splitlines():
        value = raw.strip()
        if value.startswith("$"):
            match = re.match(r"\$(\d+)\s+(\S+)", value)
            if match:
                symbols[int(match.group(1))] = match.group(2)
    return symbols


def parse_standard_symbol(symbol: str):
    match = re.fullmatch(r"r([0-9]+(?:\.[0-9]+)?)", symbol, re.I)
    if match:
        diameter = float(match.group(1)) / 1000.0
        return "round", diameter, diameter
    match = re.fullmatch(r"s([0-9]+(?:\.[0-9]+)?)", symbol, re.I)
    if match:
        size = float(match.group(1)) / 1000.0
        return "rect", size, size
    match = re.fullmatch(r"rect([0-9]+(?:\.[0-9]+)?)x([0-9]+(?:\.[0-9]+)?)", symbol, re.I)
    if match:
        return "rect", float(match.group(1)) / 1000.0, float(match.group(2)) / 1000.0
    return None


def round_symbol_diameter_in(symbol: str) -> Optional[float]:
    parsed = parse_standard_symbol(symbol)
    return parsed[1] if parsed and parsed[0] == "round" else None


class RasterCanvas:
    def __init__(self, bounds: Tuple[float, float, float, float], dpi_x: float,
                 dpi_y: Optional[float] = None, margin_px: int = 0, background: int = 0):
        self.xmin, self.ymin, self.xmax, self.ymax = bounds
        self.dpi_x = float(dpi_x)
        self.dpi_y = float(dpi_x if dpi_y is None else dpi_y)
        self.dpi = self.dpi_x
        self.margin = int(margin_px)
        width = max(1, int(math.ceil((self.xmax - self.xmin) * self.dpi_x)) + 2 * self.margin)
        height = max(1, int(math.ceil((self.ymax - self.ymin) * self.dpi_y)) + 2 * self.margin)
        self.image = Image.new("L", (width, height), color=background)
        self.draw = ImageDraw.Draw(self.image)

    def px(self, p: Point) -> Tuple[float, float]:
        x, y = p
        return ((x - self.xmin) * self.dpi_x + self.margin,
                (self.ymax - y) * self.dpi_y + self.margin)

    def draw_round_pad(self, p: Point, diameter_in: float, value: int) -> None:
        x, y = self.px(p)
        rx, ry = diameter_in * self.dpi_x / 2.0, diameter_in * self.dpi_y / 2.0
        self.draw.ellipse((x - rx, y - ry, x + rx, y + ry), fill=value)

    def draw_rect_pad(self, center: Point, width_in: float, height_in: float, value: int,
                      angle_deg: float = 0.0, transform: Transform = Transform()) -> None:
        cx, cy = center
        half_w, half_h = width_in / 2.0, height_in / 2.0
        theta = math.radians(angle_deg)
        cs, sn = math.cos(theta), math.sin(theta)
        points = []
        for x, y in [(-half_w, -half_h), (half_w, -half_h), (half_w, half_h), (-half_w, half_h)]:
            xr, yr = x * cs - y * sn + cx, x * sn + y * cs + cy
            points.append(self.px(transform.apply((xr, yr))))
        self.draw.polygon(points, fill=value)

    def draw_round_line(self, p1: Point, p2: Point, width_in: float, value: int) -> None:
        start, end = self.px(p1), self.px(p2)
        width = max(1, int(round(width_in * math.sqrt(self.dpi_x * self.dpi_y))))
        self.draw.line((start, end), fill=value, width=width)
        rx, ry = width_in * self.dpi_x / 2.0, width_in * self.dpi_y / 2.0
        for x, y in (start, end):
            self.draw.ellipse((x - rx, y - ry, x + rx, y + ry), fill=value)

    def draw_surface(self, contours: Sequence[Tuple[str, Sequence[Point]]], polarity: str) -> None:
        positive = 255 if polarity == "P" else 0
        negative = 0 if polarity == "P" else 255
        for kind, points in contours:
            if len(points) >= 3:
                self.draw.polygon([self.px(p) for p in points], fill=positive if kind.upper().startswith("I") else negative)


class ODBRenderer:
    def __init__(self, job_dir: Path, dpi: float, dpi_y: Optional[float] = None):
        self.job = job_dir
        self.dpi_x = float(dpi)
        self.dpi_y = float(dpi if dpi_y is None else dpi_y)
        self.dpi = self.dpi_x
        self.stats = RenderStats()
        self.warnings: List[str] = []

    @classmethod
    def from_um_per_pixel(cls, job_dir: Path, um_per_pixel_x: float,
                          um_per_pixel_y: Optional[float] = None) -> "ODBRenderer":
        if um_per_pixel_x <= 0:
            raise ValueError("um_per_pixel_x must be > 0")
        uy = um_per_pixel_x if um_per_pixel_y is None else um_per_pixel_y
        if uy <= 0:
            raise ValueError("um_per_pixel_y must be > 0")
        return cls(job_dir, 25400.0 / um_per_pixel_x, 25400.0 / uy)

    def _step_dir(self, step: str) -> Path:
        path = self.job / "steps" / step.lower()
        if not path.is_dir():
            raise ODBError(f"Step not found: {step}")
        return path

    def _warn(self, text: str) -> None:
        self.stats.unsupported += 1
        if len(self.warnings) < 50:
            self.warnings.append(text)

    def _render_feature_file(self, canvas: RasterCanvas, feature_file: Path, transform: Transform) -> None:
        if not feature_file.exists():
            return
        symbols = parse_symbol_table(feature_file)
        records = feature_file.read_text(errors="replace").splitlines()
        index = 0
        while index < len(records):
            record = records[index].strip()
            index += 1
            if not record or record.startswith("#") or record.startswith("$") or record == "SE":
                continue
            tokens = record.split()
            command = tokens[0]
            try:
                if command == "P" and len(tokens) >= 6:
                    x, y = float(tokens[1]), float(tokens[2])
                    symbol_id, polarity = int(tokens[3]), tokens[4].upper()
                    parsed = parse_standard_symbol(symbols.get(symbol_id, ""))
                    if parsed is None:
                        self._warn(f"Unsupported P symbol {symbols.get(symbol_id)!r}")
                        continue
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
                        self._warn(f"Unsupported L symbol {symbols.get(symbol_id)!r}")
                        continue
                    canvas.draw_round_line(transform.apply((x1, y1)), transform.apply((x2, y2)), diameter,
                                           255 if polarity == "P" else 0)
                    self.stats.lines += 1
                elif command == "S" and len(tokens) >= 2:
                    polarity = tokens[1].upper()
                    contours: List[Tuple[str, List[Point]]] = []
                    current: Optional[List[Point]] = None
                    current_kind = "I"
                    while index < len(records):
                        surface_record = records[index].strip()
                        index += 1
                        if not surface_record or surface_record.startswith("#"):
                            continue
                        values = surface_record.split()
                        cmd = values[0]
                        if cmd == "OB" and len(values) >= 4:
                            current = [(float(values[1]), float(values[2]))]
                            current_kind = values[3].upper()
                        elif cmd == "OS" and current is not None:
                            current.append((float(values[1]), float(values[2])))
                        elif cmd == "OC" and current is not None and len(values) >= 6:
                            end = float(values[1]), float(values[2])
                            center = float(values[3]), float(values[4])
                            current.extend(arc_points(current[-1], end, center, values[5].upper().startswith("Y")))
                        elif cmd == "OE":
                            if current:
                                contours.append((current_kind, [transform.apply(p) for p in current]))
                            current = None
                        elif cmd == "SE":
                            break
                        else:
                            self._warn(f"Unsupported surface record {surface_record[:80]}")
                    canvas.draw_surface(contours, polarity)
                    self.stats.surfaces += 1
                else:
                    self._warn(f"Unsupported feature record {record[:100]}")
            except (ValueError, IndexError) as exc:
                self._warn(f"Parse error: {record[:100]} ({exc})")

    def _render_step_recursive(self, canvas: RasterCanvas, step: str, layer: str,
                               parent_transform: Transform, depth: int = 0) -> None:
        if depth > 8:
            raise ODBError("STEP-REPEAT recursion too deep")
        step_dir = self._step_dir(step)
        for repeat in parse_repeats(step_dir / "stephdr"):
            for iy in range(repeat.ny):
                for ix in range(repeat.nx):
                    tx = repeat.x + ix * repeat.dx
                    ty = repeat.y + iy * repeat.dy
                    child_transform = parent_transform.compose(repeat_transform(tx, ty, repeat.angle, repeat.mirror))
                    self._render_step_recursive(canvas, repeat.name, layer, child_transform, depth + 1)
                    self.stats.repeats += 1
        self._render_feature_file(canvas, step_dir / "layers" / layer.lower() / "features", parent_transform)

    def render(self, step: str, layer: str, margin_px: int = 0) -> Image.Image:
        profile = self._step_dir(step) / "profile"
        canvas = RasterCanvas(contours_bounds(parse_profile_contours(profile)), self.dpi_x, self.dpi_y,
                              margin_px=margin_px, background=0)
        self._render_step_recursive(canvas, step.lower(), layer.lower(), Transform())
        return canvas.image

    def render_composite(self, step: str, layers: Sequence[CompositeLayer], margin_px: int = 0,
                         background: int = 0) -> Image.Image:
        """Render ordered ODB++ layers into an 8-bit grayscale composite image."""
        specs = [spec.normalized() for spec in layers]
        if not specs:
            raise ValueError("At least one composite layer is required")
        if not 0 <= int(background) <= 255:
            raise ValueError("Composite background must be between 0 and 255")

        result: Optional[Image.Image] = None
        for spec in specs:
            mask_image = self.render(step, spec.layer, margin_px=margin_px)
            mask = mask_image.point(lambda value: 255 if value > 0 else 0, mode="L")
            if result is None:
                result = Image.new("L", mask_image.size, color=int(background))
            elif result.size != mask_image.size:
                raise ODBError("Composite layer dimensions do not match")

            if spec.operation == "REPLACE":
                result.paste(spec.gv, mask=mask)
            elif spec.operation == "SUBTRACT":
                result.paste(0, mask=mask)
            else:
                contribution = Image.new("L", result.size, color=0)
                contribution.paste(spec.gv, mask=mask)
                result = ImageChops.lighter(result, contribution)

        assert result is not None
        return result


def locate_job_dir(root: Path) -> Path:
    if (root / "steps").is_dir() and (root / "matrix").exists():
        return root
    candidates = [p for p in root.iterdir() if p.is_dir() and (p / "steps").is_dir()]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise ODBError(f"Could not find ODB++ job under {root}")
    raise ODBError(f"Multiple ODB++ jobs found under {root}: {[p.name for p in candidates]}")


def extract_input(input_path: Path) -> Tuple[Path, Optional[tempfile.TemporaryDirectory]]:
    if input_path.is_dir():
        return locate_job_dir(input_path), None
    if not input_path.is_file():
        raise ODBError(f"Input not found: {input_path}")
    temp_dir = tempfile.TemporaryDirectory(prefix="odb_cam_")
    root = Path(temp_dir.name)
    try:
        with tarfile.open(input_path, "r:*") as archive:
            try:
                archive.extractall(root, filter="data")
            except TypeError:
                archive.extractall(root)
    except Exception:
        temp_dir.cleanup()
        raise
    return locate_job_dir(root), temp_dir


def list_job(job: Path) -> str:
    steps = sorted(p.name for p in (job / "steps").iterdir() if p.is_dir())
    layers: List[str] = []
    matrix = job / "matrix" / "matrix"
    if matrix.exists():
        for block in parse_kv_blocks(matrix, "LAYER"):
            if block.get("NAME"):
                layers.append(f"{block['NAME']} [{block.get('TYPE', '?')}]")
    return "Steps: " + ", ".join(steps) + "\nLayers:\n  " + "\n  ".join(layers)


def main() -> int:
    parser = argparse.ArgumentParser(description="Render ODB++ layer data to a monochrome CAM PNG")
    parser.add_argument("input", type=Path, help="ODB++ .tgz/.tar.gz or extracted job directory")
    parser.add_argument("--step", default="unit")
    parser.add_argument("--layer", default="l1")
    parser.add_argument("--dpi", type=float, default=1200.0)
    parser.add_argument("--um-per-pixel-x", type=float)
    parser.add_argument("--um-per-pixel-y", type=float)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--invert", action="store_true")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.dpi <= 0 or args.dpi > 10000:
        parser.error("--dpi must be > 0 and <= 10000")
    if args.um_per_pixel_x is not None and args.um_per_pixel_x <= 0:
        parser.error("--um-per-pixel-x must be > 0")
    if args.um_per_pixel_y is not None and args.um_per_pixel_y <= 0:
        parser.error("--um-per-pixel-y must be > 0")
    if args.um_per_pixel_y is not None and args.um_per_pixel_x is None:
        parser.error("--um-per-pixel-y requires --um-per-pixel-x")

    job, temp_dir = extract_input(args.input)
    try:
        if args.list:
            print(f"Job: {job.name}")
            print(list_job(job))
            return 0
        if args.um_per_pixel_x is not None:
            uy = args.um_per_pixel_x if args.um_per_pixel_y is None else args.um_per_pixel_y
            renderer = ODBRenderer.from_um_per_pixel(job, args.um_per_pixel_x, uy)
            suffix = f"{args.um_per_pixel_x:g}x{uy:g}umpp"
            description = f"AOI X={args.um_per_pixel_x:g}, Y={uy:g} um/pixel"
        else:
            renderer = ODBRenderer(job, args.dpi)
            suffix = f"{int(args.dpi)}dpi"
            description = f"{args.dpi:g} DPI"
        output = args.output or Path(f"{job.name}_{args.step}_{args.layer}_{suffix}.png")
        image = renderer.render(args.step, args.layer)
        if args.invert:
            image = image.point(lambda value: 255 - value)
        output.parent.mkdir(parents=True, exist_ok=True)
        image.save(output, optimize=True, dpi=(renderer.dpi_x, renderer.dpi_y))
        print(f"Saved: {output}")
        print(f"Image: {image.width} x {image.height} px | {description}")
        stats = renderer.stats
        print(f"Rendered: pads={stats.pads}, lines={stats.lines}, surfaces={stats.surfaces}, repeats={stats.repeats}")
        print(f"Unsupported/parse warnings: {stats.unsupported}")
        for warning in renderer.warnings[:10]:
            print(f"  WARNING: {warning}")
        return 0 if stats.unsupported == 0 else 2
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
