from pathlib import Path

from aoi.coordinate_validation import (
    _match_ert_for_panel,
    coordinate_hypotheses,
    discover_images,
    parse_image_context,
)


def test_parse_image_context_y_x_order_png(tmp_path: Path):
    image = tmp_path / "GIDS" / "MCP19013C00-006" / "L1-13-T-050" / "LOT001" / "PANEL001" / "G_23.465_471.321_6.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"")
    ctx = parse_image_context(image, tmp_path / "GIDS")
    assert ctx.item_revision == "MCP19013C00-006"
    assert ctx.layer == "L1-13-T-050"
    assert ctx.lot == "LOT001"
    assert ctx.panel == "PANEL001"
    assert ctx.y_mm == 23.465
    assert ctx.x_mm == 471.321
    assert ctx.image_index == "6"
    assert ctx.image_kind == "G"


def test_parse_image_context_supports_jpg_and_cam_prefix(tmp_path: Path):
    panel = tmp_path / "GIDS" / "ITEM-001" / "L1" / "LOT1" / "PANEL1"
    panel.mkdir(parents=True)
    cam = panel / "C_10.125_20.250_3.jpg"
    cam.write_bytes(b"")
    ctx = parse_image_context(cam, tmp_path / "GIDS")
    assert ctx.image_kind == "C"
    assert ctx.y_mm == 10.125
    assert ctx.x_mm == 20.250
    assert ctx.image_index == "3"


def test_discover_images_only_returns_g_inputs_and_supports_jpg(tmp_path: Path):
    panel = tmp_path / "GIDS" / "ITEM-001" / "L1" / "LOT1" / "PANEL1"
    panel.mkdir(parents=True)
    g_jpg = panel / "G_1.000_2.000_1.jpg"
    g_png = panel / "G_3.000_4.000_2.PNG"
    cam = panel / "C_1.000_2.000_1.jpg"
    unrelated = panel / "other.jpg"
    for path in (g_jpg, g_png, cam, unrelated):
        path.write_bytes(b"")
    assert discover_images(tmp_path) == sorted([g_jpg, g_png])


def test_zero_padded_ert_panel_resolution(tmp_path: Path):
    files = []
    for name in ("001.ERT", "002.ERT", "010.ERT"):
        path = tmp_path / name
        path.write_text("", encoding="utf-8")
        files.append(path)
    assert _match_ert_for_panel(files, "1").name == "001.ERT"
    assert _match_ert_for_panel(files, "01").name == "001.ERT"
    assert _match_ert_for_panel(files, "2").name == "002.ERT"
    assert _match_ert_for_panel(files, "10").name == "010.ERT"


def test_coordinate_hypotheses_keep_direct_local_explicit():
    bounds = (-10.0, -20.0, 10.0, 20.0)
    rows = coordinate_hypotheses(25.4, 50.8, bounds)
    direct = rows[0]
    assert direct.name == "DIRECT_LOCAL"
    assert direct.x_in == 1.0
    assert direct.y_in == 2.0
    assert rows[1].x_in == -9.0
    assert rows[1].y_in == -18.0
    assert rows[2].x_in == -9.0
    assert rows[2].y_in == 18.0
