import pytest

from aoi.contour_mapping import AoiOdbTransform, CropGeometry, contour_pixel_to_odb_points
from aoi.synthetic_contours import diamond_contour, rectangle_contour


def test_rectangle_is_centered_on_arbitrary_crop_geometry():
    geometry = CropGeometry.isotropic(320, 240, 2.5)
    contour = rectangle_contour(geometry, width_px=40, height_px=20)
    cx, cy = geometry.center_px
    assert contour == pytest.approx([
        (cx - 20, cy - 10),
        (cx + 20, cy - 10),
        (cx + 20, cy + 10),
        (cx - 20, cy + 10),
    ])


def test_rectangle_offset_can_simulate_defect_away_from_crop_center():
    geometry = CropGeometry.isotropic(200, 100, 5.5)
    contour = rectangle_contour(geometry, 20, 10, offset_px=(15, -8))
    xs = [p[0] for p in contour]
    ys = [p[1] for p in contour]
    assert (min(xs) + max(xs)) / 2 == pytest.approx(geometry.center_px[0] + 15)
    assert (min(ys) + max(ys)) / 2 == pytest.approx(geometry.center_px[1] - 8)


def test_synthetic_contour_uses_existing_aoi_to_odb_pipeline():
    geometry = CropGeometry.isotropic(100, 100, 2.5)
    contour = rectangle_contour(geometry, width_px=20, height_px=20)
    transform = AoiOdbTransform(tx_mm=-196.3925, ty_mm=242.6690)
    odb = contour_pixel_to_odb_points(contour, (45.138, 101.663), geometry, transform)

    xs = [p[0] for p in odb]
    ys = [p[1] for p in odb]
    assert (min(xs) + max(xs)) / 2 == pytest.approx(-94.7295)
    assert (min(ys) + max(ys)) / 2 == pytest.approx(197.5310)
    assert max(xs) - min(xs) == pytest.approx(0.05)
    assert max(ys) - min(ys) == pytest.approx(0.05)


def test_diamond_provides_non_rectangular_contour():
    geometry = CropGeometry.isotropic(100, 80, 2.5)
    contour = diamond_contour(geometry, 30, 20)
    assert len(contour) == 4
    assert contour[0][0] == pytest.approx(geometry.center_px[0])
    assert contour[1][1] == pytest.approx(geometry.center_px[1])


def test_invalid_synthetic_size_rejected():
    geometry = CropGeometry.isotropic(100, 100, 2.5)
    with pytest.raises(ValueError):
        rectangle_contour(geometry, 0, 20)
    with pytest.raises(ValueError):
        diamond_contour(geometry, 20, -1)
