from search_coordinate_constellation import solve


def _row(aoi,odb,score=.8): return {"aoi_panel_mm":list(aoi),"best_odb_mm":list(odb),"final_score":score,"g_image":"x"}


def test_constellation_recovers_swap_reflection_and_rejects_outlier():
    # M = [[0,-1],[-1,0]], t=(100,200)
    rows=[]
    for x,y in [(0,0),(100,0),(0,50),(100,50)]: rows.append(_row((x,y),(100-y,200-x),.85))
    rows.append(_row((40,20),(999,-999),.95))
    r=solve({"images":rows},threshold=.2)
    assert r["best_orientation"]=="SWAP_X-_Y-"
    assert r["inlier_count"]==4
    assert abs(r["translation_mm"][0]-100)<1e-9
    assert abs(r["translation_mm"][1]-200)<1e-9


def test_constellation_uses_geometry_over_higher_outlier_image_score():
    rows=[_row((0,0),(10,20),.6),_row((50,0),(60,20),.6),_row((0,50),(10,70),.6),_row((25,25),(500,500),.99)]
    r=solve({"images":rows},threshold=.1)
    assert r["inlier_count"]==3
    assert r["best_orientation"]=="DIRECT_X+_Y+"
