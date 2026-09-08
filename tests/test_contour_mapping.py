import pytest

from aoi.contour_mapping import (
    AoiOdbTransform,
    CropGeometry,
    contour_pixel_to_aoi_points,
    contour_pixel_to_odb_points,
)


def test_100x100_uses_geometric_center_49_5():
    geometry = CropGeometry.isotropic(100, 100, 2.5)
    assert geometry.center_px == (49.5, 49.5)


def test_arbitrary_rectangular_image_size():
    geometry = CropGeometry.isotropic(320, 240, 5.5)
    assert geometry.center_px == (159.5, 119.5)
    points = contour_pixel_to_aoi_points([(159.5, 119.5)], (10.0, 20.0), geometry)
    assert points[0] == pytest.approx((10.0, 20.0))


def test_resolution_changes_physical_offset():
    contour = [(59.5, 49.5)]  # +10 px in X from 100x100 geometric center
    p25 = contour_pixel_to_aoi_points(contour, (10.0, 20.0), CropGeometry.isotropic(100, 100, 2.5))[0]
    p55 = contour_pixel_to_aoi_points(contour, (10.0, 20.0), CropGeometry.isotropic(100, 100, 5.5))[0]
    assert p25 == pytest.approx((10.025, 20.0))
    assert p55 == pytest.approx((10.055, 20.0))


def test_anisotropic_xy_resolution_supported():
    geometry = CropGeometry(200, 100, 2.0, 4.0)
    center_x, center_y = geometry.center_px
    point = contour_pixel_to_aoi_points([(center_x + 10, center_y + 10)], (1.0, 2.0), geometry)[0]
    assert point == pytest.approx((1.020, 2.040))


def test_existing_swap_transform_applies_to_every_contour_vertex():
    geometry = CropGeometry.isotropic(100, 100, 2.5)
    transform = AoiOdbTransform(tx_mm=-196.3925, ty_mm=242.6690)
    center = geometry.center_px
    points = contour_pixel_to_odb_points([center, (center[0] + 20, center[1] - 20)], (45.138, 101.663), geometry, transform)
    assert points[0] == pytest.approx((-94.7295, 197.5310))
    assert points[1] == pytest.approx((-94.7795, 197.4810))


def test_invalid_geometry_rejected():
    with pytest.raises(ValueError):
        CropGeometry.isotropic(0, 100, 2.5)
    with pytest.raises(ValueError):
        CropGeometry.isotropic(100, 100, 0)
