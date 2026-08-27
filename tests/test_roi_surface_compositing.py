from render.roi import FixedRasterCanvas


def test_positive_surface_hole_does_not_erase_previous_positive_surface():
    canvas = FixedRasterCanvas((0.0, 0.0, 1.0, 1.0), 100.0, 100.0, 100, 100, background=0)

    # First positive surface deposits copper on the left half.
    canvas.draw_surface([
        ("I", [(0.05, 0.05), (0.55, 0.05), (0.55, 0.95), (0.05, 0.95)]),
    ], "P")

    # Second positive surface has a large island and an H contour that overlaps
    # part of the first surface. The H contour is local to this second surface;
    # it must not erase copper already deposited by the first surface.
    canvas.draw_surface([
        ("I", [(0.40, 0.05), (0.95, 0.05), (0.95, 0.95), (0.40, 0.95)]),
        ("H", [(0.45, 0.20), (0.70, 0.20), (0.70, 0.80), (0.45, 0.80)]),
    ], "P")

    # x=50,y=50 lies inside the second surface's H contour but also inside the
    # first positive surface, so it must remain set after union compositing.
    assert canvas.image.getpixel((50, 50)) == 255


def test_negative_surface_subtracts_its_filled_geometry():
    canvas = FixedRasterCanvas((0.0, 0.0, 1.0, 1.0), 100.0, 100.0, 100, 100, background=0)
    canvas.draw_surface([
        ("I", [(0.05, 0.05), (0.95, 0.05), (0.95, 0.95), (0.05, 0.95)]),
    ], "P")
    canvas.draw_surface([
        ("I", [(0.40, 0.40), (0.60, 0.40), (0.60, 0.60), (0.40, 0.60)]),
    ], "N")

    assert canvas.image.getpixel((50, 50)) == 0
    assert canvas.image.getpixel((10, 10)) == 255
