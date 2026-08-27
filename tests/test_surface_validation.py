from render.surface_validation import _rasterize_surface_decomposition


def test_surface_decomposition_separates_island_holes_and_final():
    contours = [
        ("I", [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]),
        ("H", [(0.4, 0.4), (0.6, 0.4), (0.6, 0.6), (0.4, 0.6)]),
    ]
    island, holes, final = _rasterize_surface_decomposition(
        contours,
        "P",
        (0.0, 0.0, 1.0, 1.0),
        100.0,
        100.0,
        100,
        100,
    )

    assert island.size == (100, 100)
    assert holes.size == (100, 100)
    assert final.size == (100, 100)
    assert island.getbbox() is not None
    assert holes.getbbox() is not None
    assert final.getpixel((50, 50)) == 0
    assert final.getpixel((10, 10)) == 255
    assert sum(final.histogram()[1:]) < sum(island.histogram()[1:])
