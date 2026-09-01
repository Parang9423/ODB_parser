from calibrate_global_coordinates import _fit_affine, _predict, _robust_affine, calibrate


def _p(x,y,u,v,score=.9): return {"aoi":[x,y],"odb":[u,v],"score":score}


def test_affine_fit_recovers_known_transform():
    # u=2x+3y+5, v=-x+4y-2
    pts=[_p(0,0,5,-2),_p(1,0,7,-3),_p(0,1,8,2),_p(2,3,18,8)]
    m=_fit_affine(pts)
    u,v=_predict(m,(4,5))
    assert abs(u-28)<1e-9
    assert abs(v-14)<1e-9


def test_robust_affine_rejects_single_bad_match():
    good=[_p(0,0,10,20),_p(100,0,110,20),_p(0,100,10,120),_p(100,100,110,120)]
    pts=good+[_p(50,50,999,-999)]
    m,inliers=_robust_affine(pts,threshold_mm=.1)
    assert len(inliers)==4
    assert abs(_predict(m,(25,75))[0]-35)<1e-6
    assert abs(_predict(m,(25,75))[1]-95)<1e-6


def test_calibrate_consumes_local_search_report():
    report={"images":[
        {"g_image":"a","aoi_panel_mm":[0,0],"best_odb_mm":[10,20],"final_score":.9},
        {"g_image":"b","aoi_panel_mm":[100,0],"best_odb_mm":[110,20],"final_score":.9},
        {"g_image":"c","aoi_panel_mm":[0,100],"best_odb_mm":[10,120],"final_score":.9},
        {"g_image":"bad","aoi_panel_mm":[50,50],"best_odb_mm":[500,500],"final_score":.8},
    ]}
    result=calibrate(report,threshold_mm=.2)
    assert result["inlier_count"]==3
    assert result["point_count"]==4
