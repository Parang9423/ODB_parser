from diagnose_full_panel_mapping import _point_to_pixel


def test_point_to_pixel_uses_odb_y_up_image_y_down():
    bounds=[-207.5,-257.5,207.5,257.5]
    assert _point_to_pixel(-207.5,257.5,bounds,(416,516)) == (0.0,0.0)
    assert _point_to_pixel(207.5,-257.5,bounds,(416,516)) == (415.0,515.0)


def test_point_to_pixel_center():
    x,y=_point_to_pixel(0.0,0.0,[-10.0,-20.0,10.0,20.0],(201,401))
    assert x == 100.0
    assert y == 200.0
