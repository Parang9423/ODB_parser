from validate_coordinate_geometry import _inside, _map_top_left, _sample_details


def test_top_left_mm_maps_to_cartesian_bounds():
    bounds=[-100.0,-50.0,100.0,50.0]
    assert _map_top_left(bounds,0,0)==(-100.0,50.0)
    assert _map_top_left(bounds,200,100)==(100.0,-50.0)
    assert _map_top_left(bounds,25,10)==(-75.0,40.0)


def test_inside_bounds():
    assert _inside((-100,50),[-100,-50,100,50])
    assert not _inside((100.1,0),[-100,-50,100,50])


def _d(x,y): return {"image_context":{"x_mm":x,"y_mm":y}}


def test_sample_details_spreads_across_aoi_x():
    rows=[_d(x,0) for x in range(10)]
    picked=_sample_details(rows,3)
    assert [r["image_context"]["x_mm"] for r in picked]==[0,4,9]
