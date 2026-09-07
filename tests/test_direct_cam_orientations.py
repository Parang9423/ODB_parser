from validate_direct_cam_orientations import ORIENTATIONS,_translation_for_frame,_map,_joint


def test_strip_array_center_alignment_swap_negative_negative():
    frame=[0.0,0.0,485.338,392.785]
    target=[-196.35,-242.625,196.35,242.625]
    m=ORIENTATIONS["SWAP_X-_Y-"]
    t=_translation_for_frame(m,frame,target)
    assert round(t[0],4)==196.3925
    assert round(t[1],4)==242.6690
    x,y=_map(m,t,(45.138,101.663))
    assert round(x,4)==94.7295
    assert round(y,4)==197.5310


def test_joint_score_penalizes_low_member():
    assert _joint([0.8,0.8,0.8]) > _joint([0.8,0.8,0.1])
