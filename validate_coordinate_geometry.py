#!/usr/bin/env python3
"""Geometry-only AOI/ERT inspection-frame vs ODB PNL validation.
Tests all 8 unit-scale orthogonal transforms. Translation is deterministic by
center-aligning the transformed ERT frame to PNL profile or STRIP-array bounds.
No CAM image matching is used.
"""
from __future__ import annotations
import argparse,json
from collections import defaultdict
from pathlib import Path
from hierarchy_renderer import FastODBRenderer
from odb_cam_renderer import contours_bounds,extract_input
ORIENTATIONS={"DIRECT_X+_Y+":((1,0),(0,1)),"DIRECT_X-_Y+":((-1,0),(0,1)),"DIRECT_X+_Y-":((1,0),(0,-1)),"DIRECT_X-_Y-":((-1,0),(0,-1)),"SWAP_X+_Y+":((0,1),(1,0)),"SWAP_X-_Y+":((0,-1),(1,0)),"SWAP_X+_Y-":((0,1),(-1,0)),"SWAP_X-_Y-":((0,-1),(-1,0))}
def _apply(m,p):x,y=p;return m[0][0]*x+m[0][1]*y,m[1][0]*x+m[1][1]*y
def _bounds(ps):xs=[p[0] for p in ps];ys=[p[1] for p in ps];return [min(xs),min(ys),max(xs),max(ys)]
def _center(b):return ((b[0]+b[2])/2,(b[1]+b[3])/2)
def _inside(p,b,e=1e-6):return b[0]-e<=p[0]<=b[2]+e and b[1]-e<=p[1]<=b[3]+e
def _union(bs):return [min(b[0] for b in bs),min(b[1] for b in bs),max(b[2] for b in bs),max(b[3] for b in bs)]
def _instance_bounds_mm(r,i):return [v*25.4 for v in contours_bounds(r.transformed_profile(i))]
def _corners(b):return [(b[0],b[1]),(b[0],b[3]),(b[2],b[1]),(b[2],b[3])]
def _candidate(name,m,frame,target_name,target,rows,sbs):
 tb=_bounds([_apply(m,p) for p in _corners(frame)]);ac=_center(tb);tc=_center(target);t=(tc[0]-ac[0],tc[1]-ac[1]);fb=[tb[0]+t[0],tb[1]+t[1],tb[2]+t[0],tb[3]+t[1]];imgs=[];inside=strip=0
 for d in rows:
  ic=d["image_context"];a=(float(ic["x_mm"]),float(ic["y_mm"]));q0=_apply(m,a);q=(q0[0]+t[0],q0[1]+t[1]);pin=_inside(q,target);hits=[i for i,b in enumerate(sbs) if _inside(q,b)];inside+=pin;strip+=bool(hits);imgs.append({"image":ic["image_path"],"aoi_mm":list(a),"odb_pnl_mm":list(q),"inside_target":pin,"strip_instances":hits})
 fw,fh=fb[2]-fb[0],fb[3]-fb[1];tw,th=target[2]-target[0],target[3]-target[1]
 return {"orientation":name,"matrix_2x2":[list(m[0]),list(m[1])],"target":target_name,"translation_mm":list(t),"transformed_aoi_frame_bounds_mm":fb,"transformed_frame_size_mm":[fw,fh],"target_size_mm":[tw,th],"size_abs_error_mm":abs(fw-tw)+abs(fh-th),"size_overflow_mm":max(0,fw-tw)+max(0,fh-th),"image_inside_ratio":inside/len(rows),"image_strip_hit_ratio":strip/len(rows),"images":imgs}
def validate(path):
 payload=json.loads(Path(path).resolve().read_text(encoding="utf-8"));groups=defaultdict(list)
 for d in payload.get("results",[]):groups[str(d["resources"]["odb_path"])].append(d)
 if not groups:raise ValueError("coordinate_validation.json has no results")
 out={"method":"ERT frame vs ODB PNL/STRIP; 8 unit-scale orthogonal transforms; center-aligned translation","groups":[]};temps=[]
 try:
  for odb,rows in groups.items():
   job,tmp=extract_input(Path(odb));temps.append(tmp);r=FastODBRenderer(job,72);steps={p.name.lower() for p in (job/"steps").iterdir() if p.is_dir()};root="pnl" if "pnl" in steps else sorted(steps)[0];pnl=[v*25.4 for v in r.profile_bounds(root)];sbs=[_instance_bounds_mm(r,i) for i in r.collect_instances(root) if i.step=="strip"];su=_union(sbs) if sbs else pnl;rv=list(map(float,rows[0]["ert"]["region_values"]));frames={"ERT_XY":[min(rv[0],rv[2]),min(rv[1],rv[3]),max(rv[0],rv[2]),max(rv[1],rv[3])],"ERT_YX":[min(rv[1],rv[3]),min(rv[0],rv[2]),max(rv[1],rv[3]),max(rv[0],rv[2])]};cs=[]
   for fn,f in frames.items():
    for on,m in ORIENTATIONS.items():
     for tn,target in (("PNL_PROFILE",pnl),("STRIP_ARRAY",su)):
      c=_candidate(on,m,f,tn,target,rows,sbs);c["frame_interpretation"]=fn;c["aoi_frame_bounds_mm"]=f;cs.append(c)
   cs.sort(key=lambda c:(-c["image_inside_ratio"],-c["image_strip_hit_ratio"],c["size_overflow_mm"],c["size_abs_error_mm"]));obs=[min(float(d["image_context"]["x_mm"]) for d in rows),min(float(d["image_context"]["y_mm"]) for d in rows),max(float(d["image_context"]["x_mm"]) for d in rows),max(float(d["image_context"]["y_mm"]) for d in rows)];out["groups"].append({"odb_path":odb,"ert_region_values":rv,"pnl_bounds_mm":pnl,"strip_array_bounds_mm":su,"aoi_observed_bounds_mm":obs,"best_candidates":cs[:8],"all_candidates":cs})
  return out
 finally:
  for t in temps:
   if t is not None:t.cleanup()
def main():
 p=argparse.ArgumentParser();p.add_argument("validation_json",type=Path);p.add_argument("--output",type=Path,default=Path("coordinate_geometry_validation.json"));a=p.parse_args();res=validate(a.validation_json);op=a.output.resolve();op.write_text(json.dumps(res,ensure_ascii=False,indent=2),encoding="utf-8")
 for gi,g in enumerate(res["groups"],1):
  print(f"[{gi}] {Path(g['odb_path']).name}\n  ERT region : {g['ert_region_values']}\n  AOI observed: {['%.3f'%v for v in g['aoi_observed_bounds_mm']]}\n  PNL bounds : {['%.3f'%v for v in g['pnl_bounds_mm']]}\n  STRIP array: {['%.3f'%v for v in g['strip_array_bounds_mm']]}\n  TOP CANDIDATES")
  for i,c in enumerate(g["best_candidates"][:6],1):print(f"   {i}. {c['frame_interpretation']} {c['orientation']} -> {c['target']} t={['%.3f'%v for v in c['translation_mm']]} inside={c['image_inside_ratio']:.3f} strip={c['image_strip_hit_ratio']:.3f} size_err={c['size_abs_error_mm']:.3f} overflow={c['size_overflow_mm']:.3f}")
 print(f"Output: {op}");return 0
if __name__=="__main__":raise SystemExit(main())
