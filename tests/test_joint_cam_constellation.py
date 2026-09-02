from search_joint_cam_constellation import ORIENTATIONS, _offset, _nearest_group, _joint_score_values, _candidate_centers


def _d(x,y,name): return {"image_context":{"x_mm":x,"y_mm":y,"image_path":name}}

def test_swap_negative_offsets_preserve_relative_geometry():
    m=ORIENTATIONS["SWAP_X-_Y-"]
    assert _offset(m,(10,20),(13,25)) == (-5,-3)

def test_nearest_group_selects_central_anchor_and_neighbors():
    ds=[_d(0,0,"a"),_d(10,10,"b"),_d(11,10,"c"),_d(10,11,"d"),_d(100,100,"far")]
    g=_nearest_group(ds,4)
    assert g[0]["image_context"]["image_path"]=="b"
    assert "far" not in {x["image_context"]["image_path"] for x in g}

def test_joint_geometric_mean_penalizes_bad_member():
    assert _joint_score_values([.8,.8,.8]) > _joint_score_values([.8,.8,.02])

def test_candidate_centers_deduplicate_overlapping_beams():
    pts=_candidate_centers([(0,0),(1,0)],1,1)
    assert len(pts)==12
    assert (0.0,0.0) in pts
    assert (2.0,1.0) in pts
