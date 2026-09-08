import pytest

from coordinate_transform import CalibrationPoint, extract_xy_from_filename, fit_affine


def test_fit_affine_exact():
    src = [(0, 0), (10, 0), (0, 20), (8, 5), (3, 11)]
    points = [
        CalibrationPoint(x, y, 2 * x + 3 * y + 10, -x + 4 * y + 5)
        for x, y in src
    ]
    t = fit_affine(points)
    assert t.a == pytest.approx(2.0)
    assert t.b == pytest.approx(3.0)
    assert t.c == pytest.approx(10.0)
    assert t.d == pytest.approx(-1.0)
    assert t.e == pytest.approx(4.0)
    assert t.f == pytest.approx(5.0)
    assert t.rmse == pytest.approx(0.0, abs=1e-10)

    u, v = t.apply(7, 9)
    assert u == pytest.approx(51)
    assert v == pytest.approx(34)


def test_requires_three_points():
    with pytest.raises(ValueError):
        fit_affine([
            CalibrationPoint(0, 0, 1, 2),
            CalibrationPoint(1, 0, 3, 4),
        ])


def test_rejects_collinear_points():
    with pytest.raises(ValueError):
        fit_affine([
            CalibrationPoint(0, 0, 0, 0),
            CalibrationPoint(1, 1, 2, 2),
            CalibrationPoint(2, 2, 4, 4),
        ])


@pytest.mark.parametrize(
    "name, expected",
    [
        ("ABC_X15320_Y8240.png", (15320.0, 8240.0)),
        ("ABC-x=15320-y=8240.jpg", (15320.0, 8240.0)),
        ("part_Y_8240_X_15320.bmp", (15320.0, 8240.0)),
        ("crop_x-12.5_y-8.25.png", (-12.5, -8.25)),
    ],
)
def test_extract_xy(name, expected):
    assert extract_xy_from_filename(name) == expected


def test_extract_xy_none():
    assert extract_xy_from_filename("no_coordinate_here.png") is None
