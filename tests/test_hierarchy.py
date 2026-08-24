from pathlib import Path

from hierarchy_renderer import FastODBRenderer, adaptive_preview_dpi, clear_render_caches, cache_snapshot


def _profile(path: Path, width: float, height: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"OB 0 0 I\nOS {width} 0\nOS {width} {height}\nOS 0 {height}\nOS 0 0\nOE\n",
        encoding="utf-8",
    )


def _repeat(path: Path, name: str, x: float, y: float, nx: int = 1, dx: float = 0.0) -> None:
    path.write_text(
        "STEP-REPEAT {\n"
        f"NAME={name}\nX={x}\nY={y}\nDX={dx}\nDY=0\nNX={nx}\nNY=1\n"
        "ANGLE=0\nMIRROR=NO\n}\n",
        encoding="utf-8",
    )


def _fake_job(tmp_path: Path) -> Path:
    job = tmp_path / "job"
    (job / "matrix").mkdir(parents=True)
    (job / "matrix" / "matrix").write_text("", encoding="utf-8")
    for step, size in (("pnl", (10.0, 8.0)), ("strip", (4.0, 2.0)), ("unit", (1.0, 1.0))):
        step_dir = job / "steps" / step
        step_dir.mkdir(parents=True)
        _profile(step_dir / "profile", *size)
    _repeat(job / "steps" / "pnl" / "stephdr", "strip", 1.0, 2.0, nx=2, dx=4.0)
    _repeat(job / "steps" / "strip" / "stephdr", "unit", 0.5, 0.25, nx=2, dx=1.5)
    return job


def test_adaptive_preview_dpi_caps_large_panel():
    # 16 x 20 inch at 600 DPI would exceed 100M pixels; preview should be capped.
    dpi = adaptive_preview_dpi((0.0, 0.0, 16.0, 20.0), 600.0, max_pixels=12_000_000)
    assert 190.0 < dpi < 195.0


def test_adaptive_preview_dpi_keeps_small_unit_resolution():
    assert adaptive_preview_dpi((0.0, 0.0, 0.7, 0.7), 600.0) == 600.0


def test_collect_instances_uses_pnl_coordinate_transform(tmp_path: Path):
    job = _fake_job(tmp_path)
    renderer = FastODBRenderer(job, 100)
    instances = renderer.collect_instances("pnl")
    assert len(instances) == 7  # 1 PNL + 2 STRIP + 4 UNIT
    strips = [item for item in instances if item.step == "strip"]
    units = [item for item in instances if item.step == "unit"]
    assert [item.transform.apply((0, 0)) for item in strips] == [(1.0, 2.0), (5.0, 2.0)]
    assert units[0].transform.apply((0, 0)) == (1.5, 2.25)
    assert units[-1].transform.apply((0, 0)) == (7.0, 2.25)


def test_profile_and_repeat_parsing_are_cached(tmp_path: Path):
    clear_render_caches()
    job = _fake_job(tmp_path)
    first = FastODBRenderer(job, 100)
    first.collect_instances("pnl")
    first.profile_bounds("pnl")
    before = cache_snapshot()

    second = FastODBRenderer(job, 100)
    second.collect_instances("pnl")
    second.profile_bounds("pnl")
    after = cache_snapshot()

    assert after.repeat_hits > before.repeat_hits
    assert after.profile_hits > before.profile_hits
