from pathlib import Path

from odb_cam_renderer import (
    ODBRenderer,
    RasterCanvas,
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


def test_anisotropic_raster_canvas_size():
    canvas = RasterCanvas((0.0, 0.0, 1.0, 1.0), 100.0, 200.0)
    assert canvas.image.size == (100, 200)


def test_um_per_pixel_conversion():
    renderer = ODBRenderer.from_um_per_pixel(Path("."), 10.0, 5.0)
    assert renderer.dpi_x == 2540.0
    assert renderer.dpi_y == 5080.0


def test_composite_gv_operations(monkeypatch):
    from PIL import Image
    from odb_cam_renderer import CompositeLayer

    masks = {
        "base": Image.new("L", (4, 1), 0),
        "via": Image.new("L", (4, 1), 0),
        "hole": Image.new("L", (4, 1), 0),
    }
    masks["base"].putdata([255, 255, 255, 0])
    masks["via"].putdata([0, 255, 0, 255])
    masks["hole"].putdata([0, 0, 255, 0])

    renderer = ODBRenderer(Path("."), 100.0)
    monkeypatch.setattr(renderer, "render", lambda _step, layer, margin_px=0: masks[layer].copy())
    image = renderer.render_composite(
        "unit",
        [
            CompositeLayer("base", "REPLACE", 255),
            CompositeLayer("via", "REPLACE", 160),
            CompositeLayer("hole", "SUBTRACT", 80),
        ],
    )
    assert list(image.getdata()) == [255, 160, 0, 160]


def test_composite_add_uses_maximum(monkeypatch):
    from PIL import Image
    from odb_cam_renderer import CompositeLayer

    mask = Image.new("L", (2, 1), 255)
    renderer = ODBRenderer(Path("."), 100.0)
    monkeypatch.setattr(renderer, "render", lambda _step, _layer, margin_px=0: mask.copy())
    image = renderer.render_composite(
        "unit",
        [CompositeLayer("a", "REPLACE", 200), CompositeLayer("b", "ADD", 120)],
    )
    assert list(image.getdata()) == [200, 200]
