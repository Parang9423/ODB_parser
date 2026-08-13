from pathlib import Path

from odb_cam_renderer import (
    contours_bounds,
    parse_profile_contours,
    parse_standard_symbol,
    repeat_transform,
)


def test_standard_symbols():
    assert parse_standard_symbol("r100") == ("round", 0.1, 0.1)
    assert parse_standard_symbol("s50") == ("rect", 0.05, 0.05)
    assert parse_standard_symbol("rect100x200") == ("rect", 0.1, 0.2)
    assert parse_standard_symbol("unsupported") is None


def test_profile_bounds(tmp_path: Path):
    profile = tmp_path / "profile"
    profile.write_text("OB 0 0 I\nOS 1 0\nOS 1 2\nOS 0 2\nOS 0 0\nOE\n")
    contours = parse_profile_contours(profile)
    assert contours_bounds(contours) == (0.0, 0.0, 1.0, 2.0)


def test_repeat_translation():
    transform = repeat_transform(2.0, 3.0, 0.0, False)
    assert transform.apply((1.0, 1.0)) == (3.0, 4.0)
