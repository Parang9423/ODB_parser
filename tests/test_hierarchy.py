from pathlib import Path

from hierarchy_renderer import FastODBRenderer, adaptive_preview_dpi, clear_render_caches, cache_snapshot


def _profile(path: Path, width: float, height: float, x0: float = 0.0, y0: float = 0.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"OB {x0} {y0} I\nOS {x0+width} {y0}\nOS {x0+width} {y0+height}\n"
        f"OS {x0} {y0+height}\nOS {x0} {y0}\nOE\n",
        encoding="utf-8",
    )


def _repeat(path: Path, name: str, x: float, y: float, nx: int = 1, dx: float = 0.0,
            x_datum: float = 0.0, y_datum: float = 0.0) -> None:
    path.write_text(
        f"X_DATUM={x_datum}\nY_DATUM={y_datum}\nX_ORIGIN=0\nY_ORIGIN=0\n"
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
    dpi = adaptive_preview_dpi((0.0, 0.0, 16.0, 20.0), 600.0, max_pixels=12_000_000)
    assert 190.0 < dpi < 195.0


def test_adaptive_preview_dpi_keeps_small_unit_resolution():
    assert adaptive_preview_dpi((0.0, 0.0, 0.7, 0.7), 600.0) == 600.0


def test_collect_instances_uses_pnl_coordinate_transform(tmp_path: Path):
    job = _fake_job(tmp_path)
    renderer = FastODBRenderer(job, 100)
    instances = renderer.collect_instances("pnl")
    assert len(instances) == 7
    strips = [item for item in instances if item.step == "strip"]
    units = [item for item in instances if item.step == "unit"]
    assert [item.transform.apply((0, 0)) for item in strips] == [(1.0, 2.0), (5.0, 2.0)]
    assert units[0].transform.apply((0, 0)) == (1.5, 2.25)
    assert units[-1].transform.apply((0, 0)) == (7.0, 2.25)


def test_child_datum_is_aligned_to_repeat_xy(tmp_path: Path):
    job = tmp_path / "job"
    (job / "matrix").mkdir(parents=True)
    (job / "matrix" / "matrix").write_text("", encoding="utf-8")
    pnl = job / "steps" / "pnl"; strip = job / "steps" / "strip"
    pnl.mkdir(parents=True); strip.mkdir(parents=True)
    _profile(pnl / "profile", 10.0, 8.0)
    # Child geometry uses local coordinates around datum (2, 1), not around (0, 0).
    _profile(strip / "profile", 4.0, 2.0, x0=2.0, y0=1.0)
    _repeat(pnl / "stephdr", "strip", 5.0, 4.0)
    (strip / "stephdr").write_text("X_DATUM=2\nY_DATUM=1\nX_ORIGIN=0\nY_ORIGIN=0\n", encoding="utf-8")

    renderer = FastODBRenderer(job, 100)
    strip_instance = [x for x in renderer.collect_instances("pnl") if x.step == "strip"][0]
    # The child datum itself must land exactly at parent repeat X/Y.
    assert strip_instance.transform.apply((2.0, 1.0)) == (5.0, 4.0)
    transformed = renderer.transformed_profile(strip_instance)
    xs = [p[0] for _, pts in transformed for p in pts]
    ys = [p[1] for _, pts in transformed for p in pts]
    assert (min(xs), min(ys), max(xs), max(ys)) == (5.0, 4.0, 9.0, 6.0)


def test_step_frame_does_not_confuse_repeat_xy_with_datum(tmp_path: Path):
    job = _fake_job(tmp_path)
    renderer = FastODBRenderer(job, 100)
    frame = renderer.step_frame("pnl")
    assert frame.x_datum == 0.0
    assert frame.y_datum == 0.0


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
