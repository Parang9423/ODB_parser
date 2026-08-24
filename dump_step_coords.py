#!/usr/bin/env python3
"""Print ODB++ step coordinate metadata for PNL/STRIP/UNIT alignment debugging."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

from hierarchy_renderer import FastODBRenderer
from odb_cam_renderer import Transform, extract_input

IN_TO_MM = 25.4


def fmt_point(point) -> str:
    x, y = point
    return f"({x:.9f}, {y:.9f}) in / ({x*IN_TO_MM:.6f}, {y*IN_TO_MM:.6f}) mm"


def fmt_scalar(value: float) -> str:
    return f"{value:.9f} in ({value*IN_TO_MM:.6f} mm)"


def fmt_bounds(bounds) -> str:
    xmin, ymin, xmax, ymax = bounds
    return (
        f"({xmin:.9f}, {ymin:.9f})-({xmax:.9f}, {ymax:.9f}) in / "
        f"({xmin*IN_TO_MM:.6f}, {ymin*IN_TO_MM:.6f})-({xmax*IN_TO_MM:.6f}, {ymax*IN_TO_MM:.6f}) mm"
    )


def profile_corners(bounds):
    xmin, ymin, xmax, ymax = bounds
    return [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]


def transformed_aabb(transform: Transform, bounds):
    points = [transform.apply(point) for point in profile_corners(bounds)]
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def print_step(renderer: FastODBRenderer, step: str) -> None:
    frame = renderer.step_frame(step)
    bounds = renderer.profile_bounds(step)
    stephdr = renderer._step_dir(step) / "stephdr"

    print("=" * 100)
    print(f"STEP: {step.upper()}")
    print(f"X_DATUM  : {fmt_scalar(frame.x_datum)}")
    print(f"Y_DATUM  : {fmt_scalar(frame.y_datum)}")
    print(f"X_ORIGIN : {fmt_scalar(frame.x_origin)}")
    print(f"Y_ORIGIN : {fmt_scalar(frame.y_origin)}")
    print(f"PROFILE  : {fmt_bounds(bounds)}")
    print(f"PROFILE MIN relative to datum: {fmt_point((bounds[0]-frame.x_datum, bounds[1]-frame.y_datum))}")
    print(f"PROFILE MAX relative to datum: {fmt_point((bounds[2]-frame.x_datum, bounds[3]-frame.y_datum))}")
    print("\n[RAW stephdr]")
    print(stephdr.read_text(errors="replace").strip() if stephdr.exists() else "<missing>")
    print()


def walk(renderer: FastODBRenderer, step: str, parent_transform: Transform, depth: int = 0) -> None:
    indent = "  " * depth
    for index, repeat in enumerate(renderer._repeats(step), start=1):
        child = repeat.name.lower()
        child_frame = renderer.step_frame(child)
        child_bounds = renderer.profile_bounds(child)
        print(f"{indent}{step.upper()} -> {child.upper()} STEP-REPEAT #{index}")
        print(
            f"{indent}  X={repeat.x:.9f} Y={repeat.y:.9f} DX={repeat.dx:.9f} DY={repeat.dy:.9f} "
            f"NX={repeat.nx} NY={repeat.ny} ANGLE={repeat.angle:g} MIRROR={repeat.mirror}"
        )
        print(
            f"{indent}  child datum=({child_frame.x_datum:.9f},{child_frame.y_datum:.9f}) "
            f"origin=({child_frame.x_origin:.9f},{child_frame.y_origin:.9f})"
        )

        indices = [(0, 0)]
        last = (max(0, repeat.nx - 1), max(0, repeat.ny - 1))
        if last != (0, 0):
            indices.append(last)

        for ix, iy in indices:
            target = (repeat.x + ix * repeat.dx, repeat.y + iy * repeat.dy)
            local_transform = renderer._child_transform(child, target[0], target[1], repeat.angle, repeat.mirror)
            total_transform = parent_transform.compose(local_transform)
            print(f"{indent}  instance ({ix},{iy}) target datum : {fmt_point(target)}")
            print(f"{indent}    mapped child datum            : {fmt_point(total_transform.apply((child_frame.x_datum, child_frame.y_datum)))}")
            print(f"{indent}    mapped child local (0,0)       : {fmt_point(total_transform.apply((0.0, 0.0)))}")
            print(f"{indent}    transformed profile AABB       : {fmt_bounds(transformed_aabb(total_transform, child_bounds))}")

        first = renderer._child_transform(child, repeat.x, repeat.y, repeat.angle, repeat.mirror)
        walk(renderer, child, parent_transform.compose(first), depth + 1)


def build_report(source: Path) -> None:
    job, temp_dir = extract_input(source)
    try:
        renderer = FastODBRenderer(job, 100.0)
        available = [path.name.lower() for path in (job / "steps").iterdir() if path.is_dir()]
        ordered = [step for step in ("pnl", "strip", "unit") if step in available]
        ordered += [step for step in sorted(available) if step not in ordered]

        print(f"ODB JOB : {job.name}")
        print(f"STEPS   : {', '.join(ordered)}")
        print()
        for step in ordered:
            print_step(renderer, step)

        root = "pnl" if "pnl" in available else ordered[0]
        print("#" * 100)
        print(f"NESTED PLACEMENT CHECK / ROOT={root.upper()}")
        print("#" * 100)
        walk(renderer, root, Transform())
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Dump ODB++ stephdr/profile/STEP-REPEAT coordinate diagnostics")
    parser.add_argument("source", type=Path, help="ODB++ .tgz/.tar.gz or extracted job directory")
    args = parser.parse_args(argv)
    build_report(args.source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
