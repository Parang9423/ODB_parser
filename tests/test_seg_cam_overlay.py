from pathlib import Path

from PIL import Image

from aoi.seg_cam_overlay import (
    discover_gid_pairs,
    image_center,
    map_contour_by_shared_center,
    parse_coordinate_key,
)


def test_parse_coordinate_key_ignores_prefix_and_suffix():
    assert parse_coordinate_key("G_383.205_63.183_19.jpg") == (383.205, 63.183)
    assert parse_coordinate_key("C_383.205_63.183_anything.png") == (383.205, 63.183)


def test_image_center_uses_pixel_center_convention():
    assert image_center((100, 100)) == (49.5, 49.5)
    assert image_center((200, 200)) == (99.5, 99.5)


def test_100_to_200_shared_center_is_translation_not_scaling():
    contour = [(49.5, 49.5), (40.0, 60.0), (55.0, 55.0)]
    mapped = map_contour_by_shared_center(contour, (100, 100), (200, 200))
    assert mapped == [(99.5, 99.5), (90.0, 110.0), (105.0, 105.0)]


def test_arbitrary_sizes_keep_relative_offset_from_center():
    contour = [(10.0, 20.0)]
    mapped = map_contour_by_shared_center(contour, (80, 60), (180, 120))
    source_center = image_center((80, 60))
    cam_center = image_center((180, 120))
    assert mapped[0][0] - cam_center[0] == contour[0][0] - source_center[0]
    assert mapped[0][1] - cam_center[1] == contour[0][1] - source_center[1]


def test_discover_gid_pairs_matches_first_two_coordinates(tmp_path: Path):
    g = tmp_path / "G_383.205_63.183_19.jpg"
    c = tmp_path / "C_383.205_63.183_CAM.png"
    other = tmp_path / "G_1.000_2.000_3.jpg"
    Image.new("L", (100, 100)).save(g)
    Image.new("L", (200, 200)).save(c)
    Image.new("L", (100, 100)).save(other)

    pairs = discover_gid_pairs(tmp_path)
    assert pairs == [(g, c)]
