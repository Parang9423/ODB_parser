#!/usr/bin/env python3
"""Joint CAM constellation search using bounded small-ROI beam search.

Each candidate is a single global translation of the AOI constellation. Instead
of rendering one enormous patch spanning distant members, this version renders
only small ROI patches around each member position. A coarse beam keeps the best
anchor translations and progressively refines them. Candidate/member ROI renders
are cached within the run.
"""
from __future__ import annotations
import argparse, json, math, time
from pathlib import Path
from PIL import Image, ImageOps
from odb_cam_renderer import extract_input
from render.roi import render_roi_cam
from search_local_coordinate_match import (_find_reference,_inside_panel,_score_crop,_stage_resolution,_resize_reference,_fmt_seconds,_axis_offsets)

ORIENTATIONS={
 "DIRECT_X+_Y+":((1,0),(0,1)),"DIRECT_X-_Y+":((-1,0),(0,1)),"DIRECT_X+_Y-":((1,0),(0,-1)),"DIRECT_X-_Y-":((-1,0),(0,-1)),
 "SWAP_X+_Y+":((0,1),(1,0)),"SWAP_X-_Y+":((0,-1),(1,0)),"SWAP_X+_Y-":((0,1),(-1,0)),"SWAP_X-_Y-":((0,-1),(-1,0))}

def _xy(d): i=d["image_context"]; return float(i["x_mm"]),float(i["y_mm"])
def _apply(m,p): x,y=p; return (m[0][0]*x+m[0][1]*y,m[1][0]*x+m[1][1]*y)
def _offset(m,a,b): return _apply(m,(b[0]-a[0],b[1]-a[1]))

def _nearest_group(details,count,max_distance=None):
    pts=[(*_xy(d),i) for i,d in enumerate(details)]; cx=sum(p[0] for p in pts)/len(pts); cy=sum(p[1] for p in pts)/len(pts)
    anchor=min(pts,key=lambda p:(p[0]-cx)**2+(p[1]-cy)**2); ax,ay,ai=anchor
    rest=sorted((p for p in pts if p[2]!=ai),key=lambda p:(p[0]-ax)**2+(p[1]-ay)**2)
    if max_distance is not None: rest=[p for p in rest if math.hypot(p[0]-ax,p[1]-ay)<=max_distance]
    return [details[i] for i in [ai]+[p[2] for p in rest[:max(0,count-1)]]]

def _parse_levels(raw):
    vals=raw or ["10:2","2:0.5","0.5:0.1","0.1:0.02"]; return [tuple(map(float,s.split(":",1))) for s in vals]

def _joint_score_values(scores):
    eps=1e-6; return math.exp(sum(math.log(max(eps,s)) for s in scores)/len(scores))

def _candidate_centers(parents,radius,step):
    offsets=_axis_offsets(radius,step); seen=set(); out=[]
    for px,py in parents:
        for dy in offsets:
            for dx in offsets:
                q=(round(px+dx,9),round(py+dy,9))
                if q not in seen: seen.add(q); out.append(q)
    return out

def _render_score(job,layer,member,xy,native_um,stage_um,cache):
    ref=_resize_reference(member["reference"],native_um,stage_um); key=(round(xy[0],6),round(xy[1],6),round(stage_um,6),ref.width,ref.height,member["g"].name)
    if key in cache: return cache[key]
    cam,meta=render_roi_cam(job,xy[0],xy[1],stage_um,layer,width_px=ref.width,height_px=ref.height,signal_gv=255,drill_gv=125,return_components=False)
    score,detail=_score_crop(cam,ref); cache[key]=(score,detail); return score,detail

def main():
    p=argparse.ArgumentParser(description="Beam-search joint multi-reference CAM constellation")
    p.add_argument("validation_json",type=Path); p.add_argument("--output",type=Path,default=Path("joint_constellation_search")); p.add_argument("--orientation",default="SWAP_X-_Y-",choices=sorted(ORIENTATIONS)); p.add_argument("--group-size",type=int,default=4); p.add_argument("--max-neighbor-mm",type=float,default=None); p.add_argument("--level",action="append"); p.add_argument("--seed-json",type=Path,default=None); p.add_argument("--beam-width",type=int,default=3,help="Number of anchor candidates retained between stages"); p.add_argument("--coarse-members",type=int,default=2,help="Members scored at stage 1; later stages use all members")
    a=p.parse_args(); payload=json.loads(a.validation_json.resolve().read_text(encoding="utf-8")); all_details=list(payload.get("results",[]))
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
    pb=list(group[0]["pnl_bounds_mm"]); q=_apply(m,anchor_xy)
    if seed_t is not None: cx,cy=q[0]+float(seed_t[0]),q[1]+float(seed_t[1])
    else: cx=(float(pb[0])+float(pb[2]))/2; cy=(float(pb[1])+float(pb[3]))/2
    odb=str(group[0]["resources"]["odb_path"]); layer=str(group[0]["image_context"]["layer"]); native=float(group[0]["ert"]["resolution_um_per_px"]); job,tmp=extract_input(Path(odb)); started=time.perf_counter(); stages=[]; cache={}; offsets=[_offset(m,anchor_xy,x["xy"]) for x in members]; beam=[(cx,cy)]
    try:
        print(f"orientation={orientation} group={len(members)} anchor={anchor_xy} seed=({cx:.3f},{cy:.3f}) beam={a.beam_width}",flush=True)
        for i,mem in enumerate(members): print(f"  member {i}: AOI={mem['xy']} file={mem['g'].name} offset={offsets[i]}",flush=True)
        for sno,(radius,step) in enumerate(levels,1):
            st=time.perf_counter(); um=_stage_resolution(native,step); use_count=min(len(members),max(1,a.coarse_members)) if sno==1 else len(members); candidates=_candidate_centers(beam,radius,step); ranked=[]; skipped=0
            print(f"stage {sno}: parents={len(beam)} candidates={len(candidates)} members={use_count} raster={um:g}um/px",flush=True)
            for ci,(axc,ayc) in enumerate(candidates,1):
                scores=[]; details=[]; valid=True
                for mi in range(use_count):
                    x,y=axc+offsets[mi][0],ayc+offsets[mi][1]
                    if not _inside_panel(x,y,pb): valid=False; break
                    score,sd=_render_score(job,layer,members[mi],(x,y),native,um,cache); scores.append(score); details.append(sd)
                if not valid: skipped+=1; continue
                ranked.append((_joint_score_values(scores),axc,ayc,scores,details))
                if ci==1 or ci%10==0 or ci==len(candidates): print(f"  {ci}/{len(candidates)} current=({axc:.3f},{ayc:.3f}) joint={ranked[-1][0]:.4f}",flush=True)
            if not ranked: raise RuntimeError(f"stage {sno}: no valid beam candidates")
            ranked.sort(key=lambda r:r[0],reverse=True); kept=ranked[:max(1,a.beam_width)]; beam=[(r[1],r[2]) for r in kept]; best=kept[0]; elapsed=time.perf_counter()-st
            stages.append({"stage":sno,"radius_mm":radius,"step_mm":step,"resolution_um_per_px":um,"candidate_count":len(candidates),"valid_candidates":len(ranked),"skipped_candidates":skipped,"members_scored":use_count,"beam":[{"anchor_odb_mm":[r[1],r[2]],"joint_score":r[0],"member_scores":r[3]} for r in kept],"elapsed_seconds":elapsed})
            print(f"stage {sno} BEST=({best[1]:.3f},{best[2]:.3f}) joint={best[0]:.4f} members={[round(s,4) for s in best[3]]} elapsed={_fmt_seconds(elapsed)}",flush=True)
            (out/"INTERIM.json").write_text(json.dumps({"orientation":orientation,"completed_stages":stages},ensure_ascii=False,indent=2),encoding="utf-8")
        # Re-score final beam with every member at native resolution, then choose winner.
        finals=[]
        for axc,ayc in beam:
            scores=[]; sds=[]; valid=True
            for mem,off in zip(members,offsets):
                x,y=axc+off[0],ayc+off[1]
                if not _inside_panel(x,y,pb): valid=False; break
                s,sd=_render_score(job,layer,mem,(x,y),native,native,cache); scores.append(s); sds.append(sd)
            if valid: finals.append((_joint_score_values(scores),axc,ayc,scores,sds))
        if not finals: raise RuntimeError("No valid final beam candidates")
        finals.sort(key=lambda r:r[0],reverse=True); winner=finals[0]; cx,cy=winner[1],winner[2]; final=[]
        for idx,(mem,off) in enumerate(zip(members,offsets)):
            x,y=cx+off[0],cy+off[1]; ref=mem["reference"]; cam,meta=render_roi_cam(job,x,y,native,layer,width_px=ref.width,height_px=ref.height,signal_gv=255,drill_gv=125,return_components=False); score,sd=_score_crop(cam,ref); cp=out/f"MEMBER_{idx:02d}_BEST_CAM.png"; rp=out/f"MEMBER_{idx:02d}_REFERENCE_C.png"; cam.save(cp); ref.save(rp); final.append({"g_image":str(mem["g"]),"aoi_mm":list(mem["xy"]),"odb_mm":[x,y],"score":score,"score_detail":sd,"reference":str(rp),"cam":str(cp)})
        q=_apply(m,anchor_xy); t=[cx-q[0],cy-q[1]]; result={"algorithm":"small-ROI beam joint CAM constellation search","orientation":orientation,"matrix_2x2":[list(m[0]),list(m[1])],"translation_mm":t,"anchor_aoi_mm":list(anchor_xy),"anchor_odb_mm":[cx,cy],"group_size":len(members),"beam_width":a.beam_width,"coarse_members":a.coarse_members,"final_joint_score":winner[0],"members":final,"stages":stages,"cache_entries":len(cache),"elapsed_seconds":time.perf_counter()-started}
        op=out/"joint_constellation_search.json"; op.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8"); print(f"FINAL orientation={orientation} t={[round(v,6) for v in t]} joint={result['final_joint_score']:.4f} elapsed={_fmt_seconds(result['elapsed_seconds'])}",flush=True); print(f"Output: {op}",flush=True)
    finally:
        if tmp is not None: tmp.cleanup()
    return 0
if __name__=="__main__": raise SystemExit(main())
