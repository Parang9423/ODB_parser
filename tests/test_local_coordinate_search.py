from search_local_coordinate_match import _axis_offsets, _grid, _inside_panel, _parse_levels


def test_axis_offsets_include_center_and_boundaries():
    values = _axis_offsets(1.0, 0.5)
    assert values == [-1.0, -0.5, 0.0, 0.5, 1.0]


def test_grid_is_centered_on_panel_mm_coordinate_without_origin_conversion():
    points = list(_grid(32.0, 101.5, 0.5, 0.5))
    assert (32.0, 101.5) in points
    assert (31.5, 101.0) in points
    assert (32.5, 102.0) in points


def test_inside_panel_uses_physical_bounds():
    bounds = [0.0, 0.0, 400.0, 500.0]
    assert _inside_panel(32.0, 101.5, bounds)
    assert not _inside_panel(-0.01, 101.5, bounds)


def test_parse_levels():
    assert _parse_levels(["10:2", "0.5:0.1"]) == [(10.0, 2.0), (0.5, 0.1)]
