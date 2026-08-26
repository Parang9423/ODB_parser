from pathlib import Path

from aoi.coordinate_validation import coordinate_hypotheses, parse_image_context


def test_parse_image_context_y_x_order(tmp_path: Path):
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
