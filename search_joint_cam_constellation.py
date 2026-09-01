#!/usr/bin/env python3
"""Joint CAM search using several AOI references as one geometric constellation.

Unlike the old local matcher, only one anchor translation is searched. Neighbor
ODB positions are fixed by their AOI displacement through a locked orthogonal
orientation. Every candidate translation is scored by all C references jointly,
so a repeated PCB feature must agree at multiple physical positions to win.
"""
from __future__ import annotations
import argparse, json, math, time
from pathlib import Path
from PIL import Image, ImageOps
from odb_cam_renderer import extract_input
from render.roi import render_roi_cam
from search_local_coordinate_match import (_find_reference,_grid,_inside_panel,_score_crop,_stage_resolution,_resize_reference,_patch_geometry,_candidate_crop,_fmt_seconds)

ORIENTATIONS={
 "DIRECT_X+_Y+":((1,0),(0,1)),"DIRECT_X-_Y+":((-1,0),(0,1)),"DIRECT_X+_Y-":((1,0),(0,-1)),"DIRECT_X-_Y-":((-1,0),(0,-1)),
 "SWAP_X+_Y+":((0,1),(1,0)),"SWAP_X-_Y+":((0,-1),(1,0)),"SWAP_X+_Y-":((0,1),(-1,0)),"SWAP_X-_Y-":((0,-1),(-1,0))}

def _xy(d): i=d["image_context"]; return float(i["x_mm"]),float(i["y_mm"])
def _apply(m,p): x,y=p; return (m[0][0]*x+m[0][1]*y,m[1][0]*x+m[1][1]*y)
def _offset(m,a,b): q=_apply(m,(b[0]-a[0],b[1]-a[1])); return q

def _nearest_group(details,count,max_distance=None):
    """Choose a spatially central anchor and nearest distinct AOI neighbors."""
    pts=[(*_xy(d),i) for i,d in enumerate(details)]; cx=sum(p[0] for p in pts)/len(pts); cy=sum(p[1] for p in pts)/len(pts)
    anchor=min(pts,key=lambda p:(p[0]-cx)**2+(p[1]-cy)**2); ax,ay,ai=anchor
    rest=sorted((p for p in pts if p[2]!=ai),key=lambda p:(p[0]-ax)**2+(p[1]-ay)**2)
    if max_distance is not None: rest=[p for p in rest if math.hypot(p[0]-ax,p[1]-ay)<=max_distance]
    ids=[ai]+[p[2] for p in rest[:max(0,count-1)]]
    return [details[i] for i in ids]

def _parse_levels(raw):
    vals=raw or ["10:2","2:0.5","0.5:0.1","0.1:0.02"]; out=[]
    for s in vals: r,st=map(float,s.split(":",1)); out.append((r,st))
    return out

def _joint_score(crops,refs):
    scores=[]; details=[]
    for crop,ref in zip(crops,refs):
        s,d=_score_crop(crop,ref); scores.append(s); details.append(d)
    # Geometric mean strongly penalizes one bad member while keeping 0..1 range.
    eps=1e-6; joint=math.exp(sum(math.log(max(eps,s)) for s in scores)/len(scores))
    return joint,scores,details

def main():
    p=argparse.ArgumentParser(description="Joint multi-reference CAM constellation search")
    p.add_argument("validation_json",type=Path); p.add_argument("--output",type=Path,default=Path("joint_constellation_search")); p.add_argument("--orientation",default="SWAP_X-_Y-",choices=sorted(ORIENTATIONS)); p.add_argument("--group-size",type=int,default=4); p.add_argument("--max-neighbor-mm",type=float,default=None); p.add_argument("--level",action="append"); p.add_argument("--seed-json",type=Path,default=None,help="Optional coordinate_constellation.json; uses its orientation/translation as anchor seed")
    a=p.parse_args(); payload=json.loads(a.validation_json.resolve().read_text(encoding="utf-8")); all_details=list(payload.get("results",[]));
    if len(all_details)<2: raise ValueError("Need at least two validation images")
    group=_nearest_group(all_details,max(2,a.group_size),a.max_neighbor_mm); orientation=a.orientation; seed_t=None
    if a.seed_json and a.seed_json.exists():
        sj=json.loads(a.seed_json.read_text(encoding="utf-8")); orientation=sj.get("best_orientation",orientation); seed_t=sj.get("translation_mm")
    m=ORIENTATIONS[orientation]; anchor_xy=_xy(group[0]); levels=_parse_levels(a.level); out=a.output.resolve(); out.mkdir(parents=True,exist_ok=True)
    members=[]
    for d in group:
        g=Path(d["image_context"]["image_path"]); c=_find_reference(g)
        with Image.open(c) as im: ref=ImageOps.grayscale(im); ref.load()
        members.append({"detail":d,"g":g,"c":c,"xy":_xy(d),"reference":ref.copy()})
    # Seed: preferred global constellation translation; fallback old mapped seed if available, otherwise panel centre.
    pb=list(group[0]["pnl_bounds_mm"])
    if seed_t is not None:
        q=_apply(m,anchor_xy); cx,cy=q[0]+float(seed_t[0]),q[1]+float(seed_t[1])
    else:
        cx=(float(pb[0])+float(pb[2]))/2; cy=(float(pb[1])+float(pb[3]))/2
    odb=str(group[0]["resources"]["odb_path"]); layer=str(group[0]["image_context"]["layer"]); native=float(group[0]["ert"]["resolution_um_per_px"]); job,tmp=extract_input(Path(odb)); started=time.perf_counter(); stages=[]
    try:
        print(f"orientation={orientation} group={len(members)} anchor={anchor_xy} seed=({cx:.3f},{cy:.3f})",flush=True)
        for i,mem in enumerate(members): print(f"  member {i}: AOI={mem['xy']} file={mem['g'].name} offset={_offset(m,anchor_xy,mem['xy'])}",flush=True)
        for sno,(radius,step) in enumerate(levels,1):
            st=time.perf_counter(); um=_stage_resolution(native,step); stage_refs=[_resize_reference(x["reference"],native,um) for x in members]
            offsets=[_offset(m,anchor_xy,x["xy"]) for x in members]
            # One patch spans anchor search radius plus all member offsets.
            xs=[o[0] for o in offsets]; ys=[o[1] for o in offsets]; margin=max(max(r.width for r in stage_refs),max(r.height for r in stage_refs))*um/2000
            minx,maxx=min(xs)-radius-margin,max(xs)+radius+margin; miny,maxy=min(ys)-radius-margin,max(ys)+radius+margin
            pcx=cx+(minx+maxx)/2; pcy=cy+(miny+maxy)/2; w=max(8,math.ceil((maxx-minx)*1000/um)+4); h=max(8,math.ceil((maxy-miny)*1000/um)+4)
            patch,meta=render_roi_cam(job,pcx,pcy,um,layer,width_px=w,height_px=h,signal_gv=255,drill_gv=125,return_components=False)
            best=(-1,None,None,None); valid=0
            for axc,ayc in _grid(cx,cy,radius,step):
                crops=[]; ok=True
                for off,ref in zip(offsets,stage_refs):
                    x,y=axc+off[0],ayc+off[1]
                    if not _inside_panel(x,y,pb): ok=False; break
                    crop=_candidate_crop(patch,pcx,pcy,x,y,um,ref.size)
                    if crop is None: ok=False; break
                    crops.append(crop)
                if not ok: continue
                joint,scores,sd=_joint_score(crops,stage_refs); valid+=1
                if joint>best[0]: best=(joint,(axc,ayc),scores,sd)
            if best[1] is None: raise RuntimeError(f"stage {sno}: no valid joint candidates")
            cx,cy=best[1]; elapsed=time.perf_counter()-st; stages.append({"stage":sno,"radius_mm":radius,"step_mm":step,"resolution_um_per_px":um,"best_anchor_odb_mm":[cx,cy],"joint_score":best[0],"member_scores":best[2],"valid_candidates":valid,"elapsed_seconds":elapsed})
            print(f"stage {sno}: anchor=({cx:.3f},{cy:.3f}) joint={best[0]:.4f} members={[round(s,4) for s in best[2]]} valid={valid} elapsed={_fmt_seconds(elapsed)}",flush=True)
        final=[]; offsets=[_offset(m,anchor_xy,x["xy"]) for x in members]
        for idx,(mem,off) in enumerate(zip(members,offsets)):
            x,y=cx+off[0],cy+off[1]; ref=mem["reference"]; cam,meta=render_roi_cam(job,x,y,native,layer,width_px=ref.width,height_px=ref.height,signal_gv=255,drill_gv=125,return_components=False); score,sd=_score_crop(cam,ref); cam_path=out/f"MEMBER_{idx:02d}_BEST_CAM.png"; ref_path=out/f"MEMBER_{idx:02d}_REFERENCE_C.png"; cam.save(cam_path); ref.save(ref_path)
            final.append({"g_image":str(mem["g"]),"aoi_mm":list(mem["xy"]),"odb_mm":[x,y],"score":score,"score_detail":sd,"reference":str(ref_path),"cam":str(cam_path)})
        q=_apply(m,anchor_xy); t=[cx-q[0],cy-q[1]]; result={"algorithm":"joint CAM constellation search","orientation":orientation,"matrix_2x2":[list(m[0]),list(m[1])],"translation_mm":t,"anchor_aoi_mm":list(anchor_xy),"anchor_odb_mm":[cx,cy],"group_size":len(members),"final_joint_score":math.exp(sum(math.log(max(1e-6,x["score"])) for x in final)/len(final)),"members":final,"stages":stages,"elapsed_seconds":time.perf_counter()-started}
        op=out/"joint_constellation_search.json"; op.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8"); print(f"FINAL orientation={orientation} t={[round(v,6) for v in t]} joint={result['final_joint_score']:.4f} elapsed={_fmt_seconds(result['elapsed_seconds'])}"); print(f"Output: {op}")
    finally:
        if tmp is not None: tmp.cleanup()
    return 0
if __name__=="__main__": raise SystemExit(main())
