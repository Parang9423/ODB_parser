from PIL import Image

from search_coordinate_mapping import _best_reference_orientation, _candidate_points
from render.surface_validation import _point_in_polygon, _polygon_rect_intersects


def test_coordinate_candidates_keep_expected_panel_corner_conventions():
    rows = {name: (x, y) for name, x, y, _ in _candidate_points(10.0, 20.0, [-100.0, -200.0, 100.0, 200.0])}
    assert rows["DIRECT_XY"] == (10.0, 20.0)
    assert rows["LEFT_TOP_XY"] == (-90.0, 180.0)
    assert rows["RIGHT_BOTTOM_XY"] == (90.0, -180.0)


def test_reference_comparison_requires_native_equal_size():
    cam = Image.new("L", (200, 200), 0)
    reference = Image.new("L", (200, 200), 0)
    score, mode = _best_reference_orientation(cam, reference)
    assert score == 0.0
    assert mode in {"normal", "inverted"}


def test_surface_geometry_helpers_distinguish_polygon_from_bbox():
    triangle = [(0.0, 0.0), (10.0, 0.0), (0.0, 10.0)]
    assert _point_in_polygon((1.0, 1.0), triangle)
    assert not _point_in_polygon((9.0, 9.0), triangle)
    assert not _polygon_rect_intersects(triangle, (8.0, 8.0, 9.0, 9.0))
    assert _polygon_rect_intersects(triangle, (0.5, 0.5, 1.5, 1.5))
