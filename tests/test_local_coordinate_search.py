from search_local_coordinate_match import (
    _axis_offsets,
    _axis_swap_required,
    _grid,
    _infer_region_bounds,
    _inside_panel,
    _map_aoi_to_pnl,
    _parse_levels,
    _spatial_sample,
)


def test_axis_offsets_include_center_and_boundaries():
    assert _axis_offsets(1.0, 0.5) == [-1.0, -0.5, 0.0, 0.5, 1.0]


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


def _detail(x, y, name, region=(0.0, 0.0, 100.0, 200.0)):
    return {
        "image_context": {"x_mm": x, "y_mm": y, "image_path": name},
        "ert": {"region_values": list(region)},
    }


def test_spatial_sample_prefers_corners_and_centre():
    details = [
        _detail(0, 0, "ll"), _detail(100, 0, "lr"),
        _detail(0, 100, "ul"), _detail(100, 100, "ur"),
        _detail(50, 50, "centre"), _detail(51, 50, "near-centre"),
    ]
    picked = _spatial_sample(details, 5)
    names = {row["image_context"]["image_path"] for row in picked}
    assert names == {"ll", "lr", "ul", "ur", "centre"}


def test_spatial_sample_returns_all_when_count_exceeds_population():
    details = [_detail(1, 2, "a"), _detail(3, 4, "b")]
    assert _spatial_sample(details, 5) == details


def test_region_order_is_inferred_from_observed_filename_coordinates():
    # Region values are [Ymin, Xmin, Ymax, Xmax]; X=180 only fits YX interpretation.
    details = [_detail(180, 50, "a", (0, 0, 100, 200)), _detail(190, 60, "b", (0, 0, 100, 200))]
    name, bounds, hits = _infer_region_bounds(details)
    assert name == "YX"
    assert bounds == (0.0, 0.0, 200.0, 100.0)
    assert hits == 2


def test_axis_swap_uses_physical_extent_similarity():
    # AOI X=200/Y=100 maps naturally to PNL X=110/Y=210 after swapping.
    assert _axis_swap_required((0, 0, 200, 100), (-55, -105, 55, 105))


def test_swapped_mapping_centres_inspection_region_inside_pnl():
    aoi = (0.0, 0.0, 200.0, 100.0)
    pnl = (-55.0, -105.0, 55.0, 105.0)
    # Margins are 5 mm on each PNL axis after swapping.
    x, y = _map_aoi_to_pnl(20.0, 30.0, aoi, pnl, "SWAP_X+_Y+")
    assert (x, y) == (-20.0, -80.0)
    xf, yf = _map_aoi_to_pnl(20.0, 30.0, aoi, pnl, "SWAP_X-_Y-")
    assert (xf, yf) == (20.0, 80.0)
