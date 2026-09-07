from validate_coordinate_geometry import _apply,_bounds,_center,_inside,_union,ORIENTATIONS

def test_swap_negative_orientation():
    assert _apply(ORIENTATIONS["SWAP_X-_Y-"],(45.0,100.0))==(-100.0,-45.0)

def test_bounds_and_center():
    b=_bounds([(-2,3),(4,-5),(1,9)])
    assert b==[-2,-5,4,9]
    assert _center(b)==(1,2)

def test_inside_and_union():
    assert _inside((0,0),[-1,-1,1,1])
    assert not _inside((2,0),[-1,-1,1,1])
    assert _union([[-2,-3,0,1],[1,-1,5,4]])==[-2,-3,5,4]
