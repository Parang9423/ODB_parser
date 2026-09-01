#!/usr/bin/env python3
"""Fast coarse-to-fine AOI(panel-mm) -> ODB coordinate search.

AOI filename coordinates are physical panel millimetres, but AOI and ODB PNL do
not necessarily share origin/axis direction. The search therefore infers the AOI
inspection rectangle from ERT region values, determines whether axes must be
swapped from physical extents, tests four flip orientations on the first sample,
and locks the best orientation for the remaining spatial samples.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from PIL import Image, ImageChops, ImageFilter, ImageOps

from odb_cam_renderer import extract_input
from render.roi import render_roi_cam


def _find_reference(g_path: Path) -> Path:
    suffix = g_path.name[2:] if g_path.name[:2].upper() == "G_" else g_path.name
    wanted = "C_" + Path(suffix).stem
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
        candidate = g_path.with_name(wanted + ext)
        if candidate.is_file(): return candidate
    raise FileNotFoundError(f"Matching C_ reference not found for {g_path.name}")


def _axis_offsets(radius_mm: float, step_mm: float) -> list[float]:
    if radius_mm < 0 or step_mm <= 0: raise ValueError("radius must be >= 0 and step must be > 0")
    count = int(math.floor(radius_mm / step_mm + 1e-9)); values = [i * step_mm for i in range(-count, count + 1)]
    if radius_mm > 0 and (not values or abs(values[0] + radius_mm) > 1e-9): values.insert(0, -radius_mm)
    if radius_mm > 0 and abs(values[-1] - radius_mm) > 1e-9: values.append(radius_mm)
    return sorted({round(v, 9) for v in values})


def _grid(cx, cy, radius, step):
    o = _axis_offsets(radius, step); return [(cx+dx, cy+dy) for dy in o for dx in o]


def _inside_panel(x, y, bounds):
    xmin,ymin,xmax,ymax=map(float,bounds); return xmin <= x <= xmax and ymin <= y <= ymax


def _parse_levels(values):
    out=[]
    for raw in values:
        try: r,s=map(float,raw.split(":",1))
        except Exception as exc: raise ValueError(f"Invalid search level {raw!r}; expected RADIUS:STEP") from exc
        if r<0 or s<=0: raise ValueError(f"Invalid search level {raw!r}")
        out.append((r,s))
    if not out: raise ValueError("At least one search level is required")
    return out


def _fmt_seconds(v):
    v=max(0.0,float(v)); return f"{v:.1f}s" if v<60 else (f"{v/60:.1f}m" if v<3600 else f"{v/3600:.2f}h")


def _stage_resolution(native_um, step_mm): return max(float(native_um), min(20.0, float(step_mm)*20.0))


def _resize_reference(reference, native_um, stage_um):
    scale=float(native_um)/float(stage_um)
    return ImageOps.grayscale(reference).resize((max(8,round(reference.width*scale)),max(8,round(reference.height*scale))),Image.Resampling.BILINEAR)


def _binary(image,invert=False):
    gray=ImageOps.autocontrast(ImageOps.grayscale(image)); gray=ImageOps.invert(gray) if invert else gray
    return gray.point(lambda v:255 if v>=128 else 0,mode="L")


def _edge_mask(binary): return binary.filter(ImageFilter.FIND_EDGES).point(lambda v:255 if v>=32 else 0,mode="L")
def _count_on(binary): return int(binary.histogram()[255])
def _occupancy(binary): return _count_on(binary)/float(max(1,binary.width*binary.height))


def _tolerant_edge_dice(a,b):
    na,nb=_count_on(a),_count_on(b)
    if na+nb==0: return 0.0
    da,db=a.filter(ImageFilter.MaxFilter(3)),b.filter(ImageFilter.MaxFilter(3))
    oa=_count_on(ImageChops.multiply(a,db)); ob=_count_on(ImageChops.multiply(b,da))
    return min(1.0,(oa+ob)/float(na+nb))


def _score_crop(cam_crop,reference_stage):
    cam_bin=_binary(cam_crop); cam_edge=_edge_mask(cam_bin); cam_occ=_occupancy(cam_bin); best=(-1.0,{})
    for mode,invert in (("normal",False),("inverted",True)):
        rb=_binary(reference_stage,invert); re=_edge_mask(rb); ro=_occupancy(rb)
        os=max(0.0,1.0-abs(cam_occ-ro)); es=_tolerant_edge_dice(cam_edge,re); score=.35*os+.65*es
        cs=cam_occ<=.005 or cam_occ>=.995; rs=ro<=.005 or ro>=.995
        if cs and not rs: score*=.02
        if score>best[0]: best=(score,{"reference_mode":mode,"occupancy_score":round(os,6),"edge_score":round(es,6),"cam_occupancy":round(cam_occ,6),"reference_occupancy":round(ro,6),"cam_solid_rejected":bool(cs and not rs)})
    return best


def _patch_geometry(radius,ref,um):
    wmm=2*radius+ref.width*um/1000+2*um/1000; hmm=2*radius+ref.height*um/1000+2*um/1000
    return max(ref.width,math.ceil(wmm*1000/um)),max(ref.height,math.ceil(hmm*1000/um))


def _candidate_crop(patch,pcx,pcy,x,y,um,crop_size):
    ppm=1000/float(um); cx=patch.width/2+(x-pcx)*ppm; cy=patch.height/2-(y-pcy)*ppm; w,h=crop_size
    left=round(cx-w/2); top=round(cy-h/2); right=left+w; bottom=top+h
    if left<0 or top<0 or right>patch.width or bottom>patch.height: return None
    return patch.crop((left,top,right,bottom))


def _xy(detail):
    i=detail["image_context"]; return float(i["x_mm"]),float(i["y_mm"])


def _spatial_sample(details,count):
    if count<=0 or count>=len(details): return list(details)
    pts=[(*_xy(d),i) for i,d in enumerate(details)]; xs=[p[0] for p in pts]; ys=[p[1] for p in pts]
    xmin,xmax=min(xs),max(xs); ymin,ymax=min(ys),max(ys); targets=[(xmin,ymin),(xmax,ymin),(xmin,ymax),(xmax,ymax),((xmin+xmax)/2,(ymin+ymax)/2)]
    chosen=[]
    for tx,ty in targets:
        if len(chosen)>=count: break
        c=[p for p in pts if p[2] not in chosen]
        if c: chosen.append(min(c,key=lambda p:(p[0]-tx)**2+(p[1]-ty)**2)[2])
    while len(chosen)<count:
        c=[p for p in pts if p[2] not in chosen]
        if not c: break
        chosen.append(max(c,key=lambda p:min((p[0]-pts[i][0])**2+(p[1]-pts[i][1])**2 for i in chosen))[2])
    return [details[i] for i in chosen]


def _infer_region_bounds(details):
    """Infer ERT 4-value ordering using how many observed filename coordinates fit."""
    rows=[]
    for d in details:
        rv=list(d.get("ert",{}).get("region_values",[]))
        if len(rv)>=4: rows.append((list(map(float,rv[:4])),*_xy(d)))
    if not rows: return None
    # Most recipes use one region for the whole panel. Compare XY vs YX interpretation.
    rv=rows[0][0]; candidates={
        "XY":(rv[0],rv[1],rv[2],rv[3]),
        "YX":(rv[1],rv[0],rv[3],rv[2]),
    }
    def score(b):
        xmin,ymin,xmax,ymax=b
        return sum(xmin-1e-6<=x<=xmax+1e-6 and ymin-1e-6<=y<=ymax+1e-6 for _r,x,y in rows)
    name=max(candidates,key=lambda k:score(candidates[k])); return name,candidates[name],score(candidates[name])


def _axis_swap_required(aoi_bounds,pnl_bounds):
    ax0,ay0,ax1,ay1=map(float,aoi_bounds); px0,py0,px1,py1=map(float,pnl_bounds)
    aw,ah=ax1-ax0,ay1-ay0; pw,ph=px1-px0,py1-py0
    direct=abs(aw-pw)+abs(ah-ph); swapped=abs(ah-pw)+abs(aw-ph)
    return swapped < direct


def _orientation_names(swapped):
    prefix="SWAP" if swapped else "DIRECT"
    return [f"{prefix}_X+_Y+",f"{prefix}_X-_Y+",f"{prefix}_X+_Y-",f"{prefix}_X-_Y-"]


def _map_aoi_to_pnl(x,y,aoi_bounds,pnl_bounds,orientation):
    ax0,ay0,ax1,ay1=map(float,aoi_bounds); px0,py0,px1,py1=map(float,pnl_bounds)
    swapped=orientation.startswith("SWAP"); flip_x="X-" in orientation; flip_y="Y-" in orientation
    aw,ah=ax1-ax0,ay1-ay0; pw,ph=px1-px0,py1-py0
    if swapped:
        sx=y-ay0; sy=x-ax0; sw,sh=ah,aw
    else:
        sx=x-ax0; sy=y-ay0; sw,sh=aw,ah
    mx=(pw-sw)/2.0; my=(ph-sh)/2.0
    ox=(px1-mx-sx) if flip_x else (px0+mx+sx)
    oy=(py1-my-sy) if flip_y else (py0+my+sy)
    return ox,oy


def _cache_key(odb,layer,x,y,um,w,h): return (str(Path(odb).resolve()),str(layer),round(float(x),6),round(float(y),6),round(float(um),6),int(w),int(h))


def _evaluate_stage(job,odb,layer,reference,native,panel_bounds,cx,cy,radius,step,cache,save_path=None):
    st=time.perf_counter(); um=_stage_resolution(native,step); sref=_resize_reference(reference,native,um); pw,ph=_patch_geometry(radius,sref,um)
    key=_cache_key(odb,layer,cx,cy,um,pw,ph); hit=key in cache
    if hit: patch,pmeta=cache[key]
    else:
        patch,pmeta=render_roi_cam(job,cx,cy,um,layer,width_px=pw,height_px=ph,signal_gv=255,drill_gv=125,return_components=False); cache[key]=(patch.copy(),dict(pmeta))
    if save_path is not None: patch.save(save_path)
    candidates=[q for q in _grid(cx,cy,radius,step) if _inside_panel(q[0],q[1],panel_bounds)]
    best=(-1.0,(cx,cy),{}); scored=skipped=0
    for x,y in candidates:
        crop=_candidate_crop(patch,cx,cy,x,y,um,sref.size)
        if crop is None: skipped+=1; continue
        score,sd=_score_crop(crop,sref); scored+=1
        if score>best[0]: best=(score,(x,y),sd)
    return {"best":best,"scored":scored,"skipped":skipped,"candidate_count":len(candidates),"um":um,"patch_size":[pw,ph],"cache_hit":hit,"elapsed":time.perf_counter()-st,"patch_meta":pmeta}


def main():
    p=argparse.ArgumentParser(description="Fast AOI panel-mm to ODB coordinate search")
    p.add_argument("validation_json",type=Path); p.add_argument("--output",type=Path,default=Path("local_coordinate_search")); p.add_argument("--limit",type=int,default=None)
    p.add_argument("--spatial-samples",type=int,default=None); p.add_argument("--level",action="append",default=None,metavar="RADIUS:STEP"); p.add_argument("--save-search-patches",action="store_true")
    args=p.parse_args(); levels=_parse_levels(args.level or ["10:2","2:0.5","0.5:0.1","0.1:0.02"])
    payload=json.loads(args.validation_json.resolve().read_text(encoding="utf-8")); all_details=list(payload.get("results",[]))
    if not all_details: raise ValueError("coordinate_validation.json has no results")
    details=_spatial_sample(all_details,max(1,args.spatial_samples)) if args.spatial_samples is not None else all_details[:(len(all_details) if args.limit is None else max(1,args.limit))]
    region_info=_infer_region_bounds(all_details)
    if region_info is None: raise ValueError("ERT region_values are required for AOI-to-PNL seed mapping")
    region_order,aoi_bounds,region_hits=region_info
    sample_pnl=list(details[0]["pnl_bounds_mm"]); swapped=_axis_swap_required(aoi_bounds,sample_pnl); orientations=_orientation_names(swapped)
    out=args.output.resolve(); out.mkdir(parents=True,exist_ok=True); report_path=out/"local_coordinate_search.json"
    report={"coordinate_assumption":"AOI filename X/Y are panel physical mm; origin/axis orientation inferred before local search","aoi_region_order":region_order,"aoi_region_bounds_mm":list(aoi_bounds),"region_observation_hits":region_hits,"axis_swap":swapped,"orientation_candidates":orientations,"locked_orientation":None,"selected_aoi_panel_mm":[list(_xy(d)) for d in details],"search_levels_mm":[{"radius":r,"step":s} for r,s in levels],"images":[],"failures":[]}
    cache={}; extracted={}; locked=None
    try:
        for ino,d in enumerate(details,1):
            started=time.perf_counter(); info,res,ert=d["image_context"],d["resources"],d["ert"]; g=Path(info["image_path"]); c=_find_reference(g); ax,ay=_xy(d); native=float(ert["resolution_um_per_px"]); pb=list(d["pnl_bounds_mm"]); odb=str(res["odb_path"]); layer=str(info["layer"])
            idir=out/f"{ino:02d}_{g.stem}"; idir.mkdir(parents=True,exist_ok=True)
            with Image.open(c) as src: reference=ImageOps.grayscale(src); reference.load()
            refpath=idir/"REFERENCE_C.png"; reference.save(refpath); print(f"[{ino}/{len(details)}] {g.name} AOI=({ax:.3f},{ay:.3f})",flush=True)
            if odb not in extracted: extracted[odb]=extract_input(Path(odb))
            job,_tmp=extracted[odb]
            try:
                orientation_trials=[]
                if locked is None:
                    radius,step=levels[0]
                    print(f"    calibrating orientation: swap={swapped} candidates={orientations}",flush=True)
                    for ori in orientations:
                        sx,sy=_map_aoi_to_pnl(ax,ay,aoi_bounds,pb,ori)
                        result=_evaluate_stage(job,odb,layer,reference,native,pb,sx,sy,radius,step,cache,idir/f"ORIENTATION_{ori}.png" if args.save_search_patches else None)
                        score=result["best"][0] if result["scored"] else -1.0
                        orientation_trials.append({"orientation":ori,"seed_odb_mm":[sx,sy],"best_odb_mm":list(result["best"][1]),"score":round(score,6),"scored_candidates":result["scored"],"elapsed_seconds":round(result["elapsed"],3)})
                        print(f"      {ori}: seed=({sx:.3f},{sy:.3f}) score={score:.4f} scored={result['scored']} elapsed={_fmt_seconds(result['elapsed'])}",flush=True)
                    valid=[r for r in orientation_trials if r["scored_candidates"]>0]
                    if not valid: raise RuntimeError("Orientation calibration produced no scoreable candidates")
                    locked=max(valid,key=lambda r:r["score"])["orientation"]; report["locked_orientation"]=locked; print(f"    LOCKED orientation: {locked}",flush=True)
                sx,sy=_map_aoi_to_pnl(ax,ay,aoi_bounds,pb,locked); cx,cy=sx,sy; stages=[]
                for sno,(radius,step) in enumerate(levels,1):
                    r=_evaluate_stage(job,odb,layer,reference,native,pb,cx,cy,radius,step,cache,idir/f"STAGE_{sno}_SEARCH_PATCH.png" if args.save_search_patches else None)
                    if not r["scored"]: raise RuntimeError(f"Stage {sno} produced no scoreable candidates from mapped seed ({cx:.3f},{cy:.3f})")
                    score,(cx,cy),sd=r["best"]
                    stages.append({"stage":sno,"radius_mm":radius,"step_mm":step,"raster_resolution_um_per_px":r["um"],"patch_size_px":r["patch_size"],"patch_cache_hit":r["cache_hit"],"candidate_count":r["candidate_count"],"scored_candidates":r["scored"],"skipped_crops":r["skipped"],"best_odb_mm":[cx,cy],"best_score":round(score,6),"score_detail":sd,"elapsed_seconds":round(r["elapsed"],3)})
                    print(f"    stage {sno}: best=({cx:.3f},{cy:.3f}) score={score:.4f} cache={'HIT' if r['cache_hit'] else 'MISS'} elapsed={_fmt_seconds(r['elapsed'])}",flush=True)
                    (idir/"INTERIM.json").write_text(json.dumps({"status":"running","orientation":locked,"mapped_seed_odb_mm":[sx,sy],"current_best_odb_mm":[cx,cy],"completed_stages":stages},indent=2),encoding="utf-8")
                fkey=_cache_key(odb,layer,cx,cy,native,reference.width,reference.height)
                if fkey in cache: final,pmeta=cache[fkey]; final_hit=True
                else:
                    final,pmeta=render_roi_cam(job,cx,cy,native,layer,width_px=reference.width,height_px=reference.height,signal_gv=255,drill_gv=125,return_components=False); cache[fkey]=(final.copy(),dict(pmeta)); final_hit=False
                fscore,fsd=_score_crop(final,reference); bestpath=idir/"BEST_CAM.png"; final.save(bestpath); dx,dy=cx-sx,cy-sy
                row={"g_image":str(g),"c_reference":str(c),"aoi_panel_mm":[ax,ay],"orientation":locked,"orientation_trials":orientation_trials,"mapped_seed_odb_mm":[sx,sy],"best_odb_mm":[cx,cy],"local_delta_from_seed_mm":[dx,dy],"final_score":round(fscore,6),"final_score_detail":fsd,"final_cache_hit":final_hit,"reference_output":str(refpath),"best_cam_output":str(bestpath),"panel_bounds_mm":pb,"elapsed_seconds":round(time.perf_counter()-started,3),"stages":stages}
                report["images"].append(row); report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
                print(f"    BEST=({cx:.6f},{cy:.6f}) seed=({sx:.6f},{sy:.6f}) local_delta=({dx:.6f},{dy:.6f}) score={fscore:.6f} total={_fmt_seconds(row['elapsed_seconds'])}",flush=True)
            except Exception as exc:
                failure={"g_image":str(g),"aoi_panel_mm":[ax,ay],"orientation":locked,"error":f"{type(exc).__name__}: {exc}"}; report["failures"].append(failure); report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8"); print(f"    FAILED: {failure['error']} -- continuing",flush=True)
    finally:
        for _job,tmp in extracted.values():
            if tmp is not None: tmp.cleanup()
    report["cache_entries"]=len(cache); report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8"); print(f"Report: {report_path}",flush=True); return 0 if not report["failures"] else 2


if __name__=="__main__": raise SystemExit(main())
