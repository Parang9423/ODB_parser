#!/usr/bin/env python3
"""Geometry-aware global calibration from local AOI/ODB candidate matches.

The local matcher may lock onto repeated PCB patterns. This tool scores global
transform hypotheses jointly: image similarity is retained, but candidates must
also preserve AOI pairwise displacement geometry. It evaluates physically
plausible orthogonal transforms (axis swap/flip, unit scale) plus translation,
selects a consensus constellation, then refits translation from its inliers.

Input: local_coordinate_search.json
Output: coordinate_constellation.json
"""
from __future__ import annotations
import argparse, itertools, json, math
from pathlib import Path


def _orientations():
    # Orthogonal transforms allowed for panel coordinate systems.
    return {
        "DIRECT_X+_Y+": ((1,0),(0,1)), "DIRECT_X-_Y+": ((-1,0),(0,1)),
        "DIRECT_X+_Y-": ((1,0),(0,-1)), "DIRECT_X-_Y-": ((-1,0),(0,-1)),
        "SWAP_X+_Y+": ((0,1),(1,0)), "SWAP_X-_Y+": ((0,-1),(1,0)),
        "SWAP_X+_Y-": ((0,1),(-1,0)), "SWAP_X-_Y-": ((0,-1),(-1,0)),
    }


def _apply(m,p):
    x,y=p; return (m[0][0]*x+m[0][1]*y,m[1][0]*x+m[1][1]*y)


def _load(report,min_score):
    out=[]
    for i,r in enumerate(report.get("images",[])):
        s=float(r.get("final_score",0))
        if s<min_score: continue
        out.append({"index":i,"g_image":r.get("g_image"),"aoi":tuple(map(float,r["aoi_panel_mm"])),"odb":tuple(map(float,r["best_odb_mm"])),"image_score":s})
    return out


def _translation(m,p):
    q=_apply(m,p["aoi"]); return (p["odb"][0]-q[0],p["odb"][1]-q[1])


def _residual(m,t,p):
    q=_apply(m,p["aoi"]); return math.hypot(q[0]+t[0]-p["odb"][0],q[1]+t[1]-p["odb"][1])


def _pair_geometry_error(m,a,b):
    aa=_apply(m,a["aoi"]); bb=_apply(m,b["aoi"])
    pd=(bb[0]-aa[0],bb[1]-aa[1]); od=(b["odb"][0]-a["odb"][0],b["odb"][1]-a["odb"][1])
    return math.hypot(pd[0]-od[0],pd[1]-od[1])


def _evaluate_orientation(name,m,points,threshold):
    best=None
    for anchor in points:
        t=_translation(m,anchor); residuals=[_residual(m,t,p) for p in points]; ids=[i for i,r in enumerate(residuals) if r<=threshold]
        if not ids: continue
        mean=sum(residuals[i] for i in ids)/len(ids); img=sum(points[i]["image_score"] for i in ids)/len(ids)
        # Consensus dominates; image score breaks ties between equally geometric hypotheses.
        key=(len(ids),-mean,img)
        if best is None or key>best[0]: best=(key,t,ids,residuals)
    if best is None: return None
    ids=best[2]
    # Translation refit is robust mean over geometric inliers.
    ts=[_translation(m,points[i]) for i in ids]; t=(sum(x for x,_ in ts)/len(ts),sum(y for _,y in ts)/len(ts))
    residuals=[_residual(m,t,p) for p in points]; ids=[i for i,r in enumerate(residuals) if r<=threshold]
    pair_errors=[]
    for i,j in itertools.combinations(ids,2): pair_errors.append(_pair_geometry_error(m,points[i],points[j]))
    rmse=math.sqrt(sum(residuals[i]**2 for i in ids)/len(ids)) if ids else None
    return {"orientation":name,"matrix_2x2":[list(m[0]),list(m[1])],"translation_mm":list(t),"inlier_ids":ids,"inlier_count":len(ids),"rmse_mm":rmse,"mean_pair_geometry_error_mm":sum(pair_errors)/len(pair_errors) if pair_errors else 0.0,"mean_inlier_image_score":sum(points[i]["image_score"] for i in ids)/len(ids) if ids else 0.0,"residuals_mm":residuals}


def solve(report,threshold=2.0,min_score=0.0):
    pts=_load(report,min_score)
    if len(pts)<2: raise ValueError("At least two local matches are required")
    trials=[]
    for name,m in _orientations().items():
        r=_evaluate_orientation(name,m,pts,threshold)
        if r: trials.append(r)
    trials.sort(key=lambda r:(-r["inlier_count"],r["rmse_mm"],r["mean_pair_geometry_error_mm"],-r["mean_inlier_image_score"]))
    best=trials[0]; ins=set(best["inlier_ids"]); m=tuple(tuple(x) for x in best["matrix_2x2"]); t=best["translation_mm"]
    rows=[]
    for i,p in enumerate(pts):
        q=_apply(m,p["aoi"]); pred=[q[0]+t[0],q[1]+t[1]]
        rows.append({**p,"aoi":list(p["aoi"]),"odb":list(p["odb"]),"predicted_odb_mm":pred,"residual_mm":math.hypot(pred[0]-p["odb"][0],pred[1]-p["odb"][1]),"inlier":i in ins})
    return {"model":"orthogonal panel transform + translation","equation":"ODB = M * AOI + t","threshold_mm":threshold,"min_image_score":min_score,"best_orientation":best["orientation"],"matrix_2x2":best["matrix_2x2"],"translation_mm":best["translation_mm"],"inlier_count":best["inlier_count"],"point_count":len(pts),"rmse_mm":best["rmse_mm"],"mean_pair_geometry_error_mm":best["mean_pair_geometry_error_mm"],"points":rows,"orientation_trials":trials}


def main():
    p=argparse.ArgumentParser(description="Joint geometry-aware AOI/ODB constellation calibration")
    p.add_argument("search_json",type=Path); p.add_argument("--output",type=Path,default=Path("coordinate_constellation.json")); p.add_argument("--threshold-mm",type=float,default=2.0); p.add_argument("--min-score",type=float,default=0.0)
    a=p.parse_args(); report=json.loads(a.search_json.resolve().read_text(encoding="utf-8")); result=solve(report,a.threshold_mm,a.min_score); a.output.resolve().write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"Orientation: {result['best_orientation']}"); print(f"M={result['matrix_2x2']}  t={result['translation_mm']}"); print(f"Inliers: {result['inlier_count']}/{result['point_count']} RMSE={result['rmse_mm']:.6f} mm pair_error={result['mean_pair_geometry_error_mm']:.6f} mm");
    for r in result["points"]: print(f"  {'IN ' if r['inlier'] else 'OUT'} AOI={r['aoi']} ODB={r['odb']} pred={[round(v,4) for v in r['predicted_odb_mm']]} residual={r['residual_mm']:.4f} score={r['image_score']:.4f}")
    print(f"Output: {a.output.resolve()}"); return 0

if __name__=="__main__": raise SystemExit(main())
