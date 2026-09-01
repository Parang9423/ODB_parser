#!/usr/bin/env python3
"""Fit a robust global AOI(panel-mm) -> ODB(PNL-mm) calibration.

Consumes local_coordinate_search.json. It uses the local matcher only to obtain
candidate correspondences, then fits a global affine transform with exhaustive
3-point RANSAC-style consensus. This rejects repeated-pattern/outlier matches and
reports residuals, inliers, matrix coefficients, scale/shear diagnostics, and
predicted ODB coordinates for every sampled AOI point.

No third-party numeric dependency is required.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path


def _solve_linear(a, b):
    n=len(b); m=[list(map(float,a[i]))+[float(b[i])] for i in range(n)]
    for col in range(n):
        pivot=max(range(col,n),key=lambda r:abs(m[r][col]))
        if abs(m[pivot][col])<1e-12: raise ValueError("Singular calibration system")
        m[col],m[pivot]=m[pivot],m[col]
        div=m[col][col]; m[col]=[v/div for v in m[col]]
        for r in range(n):
            if r==col: continue
            f=m[r][col]
            if f: m[r]=[m[r][c]-f*m[col][c] for c in range(n+1)]
    return [m[i][n] for i in range(n)]


def _fit_affine(points):
    """Least-squares affine fit using normal equations: [u v] = A[x y] + t."""
    if len(points)<3: raise ValueError("At least 3 correspondences are required")
    rows=[]; bx=[]; by=[]
    for p in points:
        x,y=p["aoi"]; u,v=p["odb"]; rows.append([x,y,1.0]); bx.append(u); by.append(v)
    ata=[[sum(r[i]*r[j] for r in rows) for j in range(3)] for i in range(3)]
    atx=[sum(r[i]*z for r,z in zip(rows,bx)) for i in range(3)]
    aty=[sum(r[i]*z for r,z in zip(rows,by)) for i in range(3)]
    cx=_solve_linear(ata,atx); cy=_solve_linear(ata,aty)
    return [cx,cy]


def _predict(model,xy):
    x,y=xy; return (model[0][0]*x+model[0][1]*y+model[0][2],model[1][0]*x+model[1][1]*y+model[1][2])


def _residual(model,p):
    u,v=_predict(model,p["aoi"]); return math.hypot(u-p["odb"][0],v-p["odb"][1])


def _robust_affine(points,threshold_mm=1.0):
    if len(points)<3: raise ValueError("At least 3 successful matches are required")
    best=None
    for ids in itertools.combinations(range(len(points)),3):
        try: model=_fit_affine([points[i] for i in ids])
        except ValueError: continue
        residuals=[_residual(model,p) for p in points]; inliers=[i for i,r in enumerate(residuals) if r<=threshold_mm]
        if len(inliers)<3: continue
        mean=sum(residuals[i] for i in inliers)/len(inliers)
        key=(len(inliers),-mean)
        if best is None or key>best[0]: best=(key,inliers,model)
    if best is None:
        model=_fit_affine(points); return model,list(range(len(points)))
    inliers=best[1]
    try: model=_fit_affine([points[i] for i in inliers])
    except ValueError: model=best[2]
    return model,inliers


def _diagnostics(model):
    a,b,_=model[0]; c,d,_=model[1]
    sx=math.hypot(a,c); sy=math.hypot(b,d); det=a*d-b*c
    dot=a*b+c*d; denom=max(1e-12,sx*sy); shear_cos=dot/denom
    rotation=math.degrees(math.atan2(c,a))
    return {"scale_x":sx,"scale_y":sy,"determinant":det,"reflection":det<0,"rotation_deg_from_aoi_x":rotation,"axis_orthogonality_cos":shear_cos}


def _load_points(report,min_score):
    pts=[]
    for i,row in enumerate(report.get("images",[])):
        score=float(row.get("final_score",0.0))
        if score<min_score: continue
        pts.append({"source_index":i,"g_image":row.get("g_image"),"aoi":list(map(float,row["aoi_panel_mm"])),"odb":list(map(float,row["best_odb_mm"])),"score":score})
    return pts


def calibrate(report,threshold_mm=1.0,min_score=0.0):
    points=_load_points(report,min_score); model,inlier_ids=_robust_affine(points,threshold_mm); inlier_set=set(inlier_ids)
    rows=[]
    for i,p in enumerate(points):
        pred=_predict(model,p["aoi"]); residual=math.hypot(pred[0]-p["odb"][0],pred[1]-p["odb"][1])
        rows.append({**p,"predicted_odb_mm":[pred[0],pred[1]],"residual_mm":residual,"inlier":i in inlier_set})
    inlier_res=[r["residual_mm"] for r in rows if r["inlier"]]
    return {"model":"affine","equations":{"odb_x":"a*x + b*y + tx","odb_y":"c*x + d*y + ty"},"matrix_2x3":model,"diagnostics":_diagnostics(model),"ransac_threshold_mm":threshold_mm,"min_match_score":min_score,"point_count":len(points),"inlier_count":len(inlier_ids),"inlier_rmse_mm":math.sqrt(sum(r*r for r in inlier_res)/len(inlier_res)) if inlier_res else None,"points":rows}


def main():
    p=argparse.ArgumentParser(description="Fit robust global AOI-to-ODB affine calibration from local search matches")
    p.add_argument("search_json",type=Path); p.add_argument("--output",type=Path,default=Path("global_coordinate_calibration.json")); p.add_argument("--threshold-mm",type=float,default=1.0); p.add_argument("--min-score",type=float,default=0.0)
    args=p.parse_args(); report=json.loads(args.search_json.resolve().read_text(encoding="utf-8")); result=calibrate(report,args.threshold_mm,args.min_score)
    args.output.resolve().write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"Affine matrix: {result['matrix_2x3']}"); print(f"Inliers: {result['inlier_count']}/{result['point_count']}  RMSE={result['inlier_rmse_mm']}"); print(f"Diagnostics: {result['diagnostics']}"); print(f"Output: {args.output.resolve()}"); return 0


if __name__=="__main__": raise SystemExit(main())
