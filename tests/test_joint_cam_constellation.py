from search_joint_cam_constellation import ORIENTATIONS, _offset, _nearest_group, _joint_score
from PIL import Image


def _d(x,y,name): return {"image_context":{"x_mm":x,"y_mm":y,"image_path":name}}

def test_swap_negative_offsets_preserve_relative_geometry():
    m=ORIENTATIONS["SWAP_X-_Y-"]
    assert _offset(m,(10,20),(13,25)) == (-5,-3)

def test_nearest_group_selects_central_anchor_and_neighbors():
    ds=[_d(0,0,"a"),_d(10,10,"b"),_d(11,10,"c"),_d(10,11,"d"),_d(100,100,"far")]
    g=_nearest_group(ds,4)
    assert g[0]["image_context"]["image_path"]=="b"
    assert "far" not in {x["image_context"]["image_path"] for x in g}

def test_joint_geometric_mean_penalizes_one_bad_member():
    # Identical binary structures score high; a solid mismatch is strongly penalized.
    ref=Image.new("L",(20,20),0)
    for x in range(5,15):
        for y in range(5,15): ref.putpixel((x,y),255)
    good=ref.copy(); bad=Image.new("L",(20,20),255)
    all_good,_,_=_joint_score([good,good],[ref,ref])
    one_bad,_,_=_joint_score([good,bad],[ref,ref])
    assert all_good > one_bad
