#!/usr/bin/env python3
"""Memory-efficient checked-Step composite helpers.

The selected PNL/STRIP/UNIT feature data is rendered in one hierarchy pass so
large PNL exports do not allocate one full-size raster per Step. The selected
Step profiles are also burned into the returned grayscale image by default.

For alignment/render debugging a lightweight text report is written to
``render_diagnostics_latest.txt`` in the current working directory. It records
Step instance counts, per-Step Layer feature-file presence/counts, and a low-DPI
non-zero pixel contribution check.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image, ImageChops, ImageDraw

from odb_cam_renderer import CompositeLayer

DIAGNOSTIC_DPI = 72.0
DIAGNOSTIC_FILE = "render_diagnostics_latest.txt"


def _draw_selected_profiles(renderer, image: Image.Image, root_step: str,
                            visible_steps: set[str], value: int = 255,
                            width_px: int = 1) -> None:
    """Rasterize selected Step profile outlines into the root image in-place."""
    if not all(hasattr(renderer, name) for name in ("profile_bounds", "collect_instances", "transformed_profile")):
        return
    xmin, _ymin, _xmax, ymax = renderer.profile_bounds(root_step)
    draw = ImageDraw.Draw(image)
    width = max(1, int(width_px))

    def px(point):
        x, y = point
        return (
            (x - xmin) * renderer.dpi_x,
            (ymax - y) * renderer.dpi_y,
        )

    for instance in renderer.collect_instances(root_step):
        if instance.step not in visible_steps:
            continue
        for _kind, points in renderer.transformed_profile(instance):
            if len(points) < 2:
                continue
            coords = [px(point) for point in points]
            if points[0] != points[-1]:
                coords.append(coords[0])
            draw.line(coords, fill=int(value), width=width)


def _legacy_test_fallback(renderer, root_step: str, specs: Sequence[CompositeLayer],
                          visible: set[str], margin_px: int, background: int) -> Image.Image:
    """Compatibility fallback for minimal/fake renderers used by unit tests."""
    merged = None
    ordered = [step for step in ("pnl", "strip", "unit") if step in visible]
    ordered.extend(sorted(visible.difference(ordered)))
    for step in ordered:
        step_result = None
        for raw_spec in specs:
            spec = raw_spec.normalized()
            mask_image = renderer.render_hierarchy(root_step, spec.layer, {step}, margin_px)
            mask = mask_image.point(lambda value: 255 if value > 0 else 0, mode="L")
            if step_result is None:
                step_result = Image.new("L", mask_image.size, color=int(background))
            if spec.operation == "REPLACE":
                step_result.paste(spec.gv, mask=mask)
            elif spec.operation == "SUBTRACT":
                step_result.paste(0, mask=mask)
            else:
                contribution = Image.new("L", step_result.size, 0)
                contribution.paste(spec.gv, mask=mask)
                step_result = ImageChops.lighter(step_result, contribution)
        merged = step_result if merged is None else ImageChops.lighter(merged, step_result)
    if merged is None:
        raise ValueError("At least one composite layer is required")
    return merged


def _feature_counts(path: Path) -> tuple[int, int, int, int]:
    """Return (records, pads, lines, surfaces) for top-level feature commands."""
    if not path.is_file():
        return 0, 0, 0, 0
    records = pads = lines = surfaces = 0
    for raw in path.read_text(errors="replace").splitlines():
        text = raw.strip()
        if not text or text.startswith(("#", "$")):
            continue
        cmd = text.split(None, 1)[0]
        if cmd == "P":
            pads += 1; records += 1
        elif cmd == "L":
            lines += 1; records += 1
        elif cmd == "S":
            surfaces += 1; records += 1
    return records, pads, lines, surfaces


def _nonzero_pixels(image: Image.Image) -> int:
    hist = image.histogram()
    return int(sum(hist[1:])) if len(hist) > 1 else 0


def build_render_diagnostics(renderer, root_step: str,
                             specs: Sequence[CompositeLayer],
                             visible_steps: Iterable[str]) -> str:
    """Build a Step x Layer diagnostic report without using export resolution."""
    visible = {str(step).lower() for step in visible_steps}
    ordered = [step for step in ("pnl", "strip", "unit") if step in visible]
    ordered.extend(sorted(visible.difference(ordered)))
    rows = [
        "ODB++ RENDER DIAGNOSTICS",
        "=" * 88,
        f"JOB              : {getattr(renderer, 'job', '-')}",
        f"ROOT STEP        : {root_step.upper()}",
        f"VISIBLE STEPS    : {', '.join(step.upper() for step in ordered)}",
        f"EXPORT SCALE     : X={getattr(renderer, 'dpi_x', 0):g} DPI / Y={getattr(renderer, 'dpi_y', 0):g} DPI",
        f"DIAGNOSTIC SCALE : {DIAGNOSTIC_DPI:g} DPI",
        "",
    ]

    instance_counts = {step: 0 for step in ordered}
    if hasattr(renderer, "collect_instances"):
        try:
            for instance in renderer.collect_instances(root_step):
                if instance.step in instance_counts:
                    instance_counts[instance.step] += 1
        except Exception as exc:
            rows.append(f"INSTANCE SCAN ERROR: {type(exc).__name__}: {exc}")
    rows.append("[STEP INSTANCES]")
    for step in ordered:
        rows.append(f"  {step.upper():6s}: {instance_counts.get(step, 0)}")
    rows.append("")

    diag_renderer = None
    try:
        if hasattr(renderer, "job"):
            diag_renderer = type(renderer)(renderer.job, DIAGNOSTIC_DPI)
    except Exception as exc:
        rows.append(f"LOW-DPI RENDERER ERROR: {type(exc).__name__}: {exc}")

    rows.append("[STEP x LAYER]")
    for step in ordered:
        rows.append(f"\nSTEP={step.upper()} instances={instance_counts.get(step, 0)}")
        for raw_spec in specs:
            spec = raw_spec.normalized()
            feature_file = Path(getattr(renderer, "job", ".")) / "steps" / step / "layers" / spec.layer.lower() / "features"
            exists = feature_file.is_file()
            size = feature_file.stat().st_size if exists else 0
            records, pads, lines, surfaces = _feature_counts(feature_file)
            nz_text = "-"
            if diag_renderer is not None:
                try:
                    mask = diag_renderer.render_hierarchy(root_step, spec.layer, {step})
                    nz_text = str(_nonzero_pixels(mask))
                except Exception as exc:
                    nz_text = f"ERROR({type(exc).__name__}: {exc})"
            rows.append(
                f"  Layer={spec.layer:<22} op={spec.operation:<8} gv={spec.gv:3d} "
                f"file={'YES' if exists else 'NO ':3s} bytes={size:<9d} "
                f"features={records:<7d} P={pads:<7d} L={lines:<7d} S={surfaces:<6d} "
                f"nonzero@72dpi={nz_text}"
            )

    rows.extend([
        "",
        "[INTERPRETATION]",
        "- file=NO: selected Matrix Layer has no feature file in that Step.",
        "- features>0 but nonzero@72dpi=0: parser/raster/coordinate rendering path is suspect.",
        "- features=0 and file=YES: file exists but contains no supported P/L/S top-level feature records.",
        "- UNIT/STRIP nonzero>0 here but missing in export: final composite/export operation is suspect.",
    ])
    return "\n".join(rows)


def _write_latest_diagnostics(renderer, root_step: str,
                              specs: Sequence[CompositeLayer],
                              visible_steps: Iterable[str]) -> None:
    """Best-effort diagnostics; never fail the actual CAM render."""
    if not hasattr(renderer, "job") or not hasattr(renderer, "render_hierarchy"):
        return
    try:
        report = build_render_diagnostics(renderer, root_step, specs, visible_steps)
        Path.cwd().joinpath(DIAGNOSTIC_FILE).write_text(report, encoding="utf-8")
    except Exception:
        pass


def render_selected_steps_composite(renderer, root_step: str,
                                    specs: Sequence[CompositeLayer],
                                    visible_steps: Iterable[str],
                                    margin_px: int = 0,
                                    background: int = 0,
                                    include_profiles: bool = True,
                                    profile_value: int = 255,
                                    profile_width_px: int = 1) -> Image.Image:
    """Render checked Step features and profiles in the root PNL coordinate system.

    Production ``FastODBRenderer`` uses one hierarchy pass for substantially
    lower peak memory. PNL/STRIP/UNIT profile outlines are included by default,
    so the saved image reflects the same Step selections as the viewer.
    """
    visible = {str(step).lower() for step in visible_steps}
    if not visible:
        raise ValueError("At least one visible Step is required")
    if not specs:
        raise ValueError("At least one composite layer is required")

    if hasattr(renderer, "render_composite_hierarchy"):
        image = renderer.render_composite_hierarchy(
            root_step, specs, visible, margin_px=margin_px, background=background
        )
    else:
        image = _legacy_test_fallback(renderer, root_step, specs, visible, margin_px, background)

    if include_profiles:
        _draw_selected_profiles(
            renderer, image, root_step, visible,
            value=profile_value, width_px=profile_width_px,
        )

    _write_latest_diagnostics(renderer, root_step, specs, visible)
    return image
