#!/usr/bin/env python3
"""Direct CAM validation for the four remaining swapped-axis orientations.

The AOI frame is taken from ERT region_values interpreted as YX, then center-
aligned to the transformed ODB STRIP-array bounds. For each of the four swap
flip orientations we compute the exact ODB coordinate, render one native CAM ROI,
and compare it with the matching C_ reference. No coordinate search is done.
"""
from __future__ import annotations
import argparse,json,math,time
from pathlib import Path
from PIL import Image,ImageOps
from odb_cam_renderer import contours_bounds,extract_input
from hierarchy_renderer import FastODBRenderer
from render.roi import render_roi_cam
from search_local_coordinate_match import _find_reference,_score_crop,_spatial_sample,_fmt_seconds

ORIENTATIONS={
 "SWAP_X+_Y+":((0,1),(1,0)),
 "SWAP_X-_Y+":((0,-1),(1,0)),
 "SWAP_X+_Y-":((0,1),(-1,0)),
 "SWAP_X-_Y-":((0,-1),(-1,0)),
}

def _apply(m,p):x,y=p;return m[0][0]*x+m[0][1]*y,m[1][0]*x+m[1][1]*y
def _bounds(ps):xs=[p[0] for p in ps];ys=[p[1] for p in ps];return [min(xs),min(ys),max(xs),max(ys)]
def _center(b):return ((b[0]+b[2])/2,(b[1]+b[3])/2)
def _corners(b):return [(b[0],b[1]),(b[0],b[3]),(b[2],b[1]),(b[2],b[3])]
def _union(bs):return [min(b[0] for b in bs),min(b[1] for b in bs),max(b[2] for b in bs),max(b[3] for b in bs)]
def _instance_bounds_mm(r,i):return [v*25.4 for v in contours_bounds(r.transformed_profile(i))]
def _translation_for_frame(m,frame,target):
    tb=_bounds([_apply(m,p) for p in _corners(frame)]);a=_center(tb);b=_center(target);return (b[0]-a[0],b[1]-a[1])
def _map(m,t,p):q=_apply(m,p);return q[0]+t[0],q[1]+t[1]
def _joint(scores):
    eps=1e-6;return math.exp(sum(math.log(max(eps,s)) for s in scores)/len(scores)) if scores else 0.0

def main():
    p=argparse.ArgumentParser(description="Direct 4-orientation CAM validation")
    p.add_argument("validation_json",type=Path);p.add_argument("--output",type=Path,default=Path("direct_cam_orientation_validation"));p.add_argument("--samples",type=int,default=5)
    a=p.parse_args();payload=json.loads(a.validation_json.resolve().read_text(encoding="utf-8"));rows=list(payload.get("results",[]))
    if not rows:raise ValueError("coordinate_validation.json has no results")
    rows=_spatial_sample(rows,max(1,a.samples));out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    odb=Path(rows[0]["resources"]["odb_path"]);layer=str(rows[0]["image_context"]["layer"]);rv=list(map(float,rows[0]["ert"]["region_values"]));frame=[min(rv[1],rv[3]),min(rv[0],rv[2]),max(rv[1],rv[3]),max(rv[0],rv[2])]
    job,tmp=extract_input(odb);started=time.perf_counter()
    try:
        r=FastODBRenderer(job,72.0);steps={p.name.lower() for p in (job/"steps").iterdir() if p.is_dir()};root="pnl" if "pnl" in steps else sorted(steps)[0];sbs=[_instance_bounds_mm(r,i) for i in r.collect_instances(root) if i.step=="strip"]
        if not sbs:raise RuntimeError("No STRIP instances found")
        target=_union(sbs);translations={name:_translation_for_frame(m,frame,target) for name,m in ORIENTATIONS.items()};results={name:[] for name in ORIENTATIONS}
        print(f"ERT_YX frame={frame} STRIP_ARRAY={target} samples={len(rows)}",flush=True)
        for si,d in enumerate(rows,1):
            ic=d["image_context"];aoi=(float(ic["x_mm"]),float(ic["y_mm"]));g=Path(ic["image_path"]);c=_find_reference(g);native=float(d["ert"]["resolution_um_per_px"])
            with Image.open(c) as im:ref=ImageOps.grayscale(im);ref.load();ref=ref.copy()
            print(f"[{si}/{len(rows)}] {g.name} AOI={aoi}",flush=True)
            for name,m in ORIENTATIONS.items():
                t=translations[name];x,y=_map(m,t,aoi);st=time.perf_counter();cam,meta=render_roi_cam(job,x,y,native,layer,width_px=ref.width,height_px=ref.height,signal_gv=255,drill_gv=125,return_components=False);score,sd=_score_crop(cam,ref);elapsed=time.perf_counter()-st
                op=out/f"S{si:02d}_{name}_CAM.png";cam.save(op)
                results[name].append({"sample":si,"g_image":str(g),"reference":str(c),"aoi_mm":list(aoi),"odb_mm":[x,y],"score":score,"score_detail":sd,"cam":str(op),"elapsed_seconds":elapsed})
                print(f"  {name}: ODB=({x:.3f},{y:.3f}) score={score:.4f} elapsed={_fmt_seconds(elapsed)}",flush=True)
            ref.save(out/f"S{si:02d}_REFERENCE_C.png")
        summary=[]
        for name,items in results.items():
            scores=[x["score"] for x in items];summary.append({"orientation":name,"matrix_2x2":[list(ORIENTATIONS[name][0]),list(ORIENTATIONS[name][1])],"translation_mm":list(translations[name]),"mean_score":sum(scores)/len(scores),"joint_geometric_score":_joint(scores),"min_score":min(scores),"scores":scores})
        summary.sort(key=lambda x:(x["joint_geometric_score"],x["mean_score"],x["min_score"]),reverse=True)
        result={"algorithm":"direct four-orientation CAM validation","frame_interpretation":"ERT_YX","aoi_frame_bounds_mm":frame,"strip_array_bounds_mm":target,"sample_count":len(rows),"ranking":summary,"results":results,"elapsed_seconds":time.perf_counter()-started}
        op=out/"direct_cam_orientation_validation.json";op.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
        print("\nRANKING")
        for i,s in enumerate(summary,1):print(f" {i}. {s['orientation']} joint={s['joint_geometric_score']:.4f} mean={s['mean_score']:.4f} min={s['min_score']:.4f} t={[round(v,3) for v in s['translation_mm']]}")
        print(f"Output: {op}")
    finally:
        if tmp is not None:tmp.cleanup()
    return 0
if __name__=="__main__":raise SystemExit(main())
