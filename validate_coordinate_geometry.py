#!/usr/bin/env python3
"""Validate whether AOI filename millimetres are PNL- or STRIP-origin coordinates.

This is geometry-only: no CAM raster matching is performed. AOI coordinates are
assumed to start at a top-left origin (+X right, +Y down). The script evaluates:
  1) PNL_TOP_LEFT: AOI origin is the PNL profile top-left.
  2) STRIP_TOP_LEFT: AOI origin is each concrete STRIP instance's *transformed
     profile* top-left in PNL space.

For STRIP candidates, the transformed STRIP profile bounds define the physical
left/top edges after STEP-REPEAT rotation/mirror. This deliberately avoids
assuming that the STRIP local (0,0) or datum is its visual top-left.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from hierarchy_renderer import FastODBRenderer
from odb_cam_renderer import contours_bounds, extract_input


def _map_top_left(bounds_mm, x_mm, y_mm):
    xmin,ymin,xmax,ymax=map(float,bounds_mm)
    return xmin+x_mm, ymax-y_mm


def _inside(point,bounds,eps=1e-6):
    x,y=point; xmin,ymin,xmax,ymax=map(float,bounds)
    return xmin-eps<=x<=xmax+eps and ymin-eps<=y<=ymax+eps


def _instance_bounds_mm(renderer,instance):
    b=contours_bounds(renderer.transformed_profile(instance))
    return [v*25.4 for v in b]


def _transform_summary(instance):
    t=instance.transform
    # Local unit axes expressed in PNL axes. Matrix is dimensionless.
    return {"a":t.a,"b":t.b,"c":t.c,"d":t.d,"tx_mm":t.tx*25.4,"ty_mm":t.ty*25.4,
            "local_x_axis_in_pnl":[t.a,t.c],"local_y_axis_in_pnl":[t.b,t.d]}


def _sample_details(details,limit):
    if limit is None or limit<=0 or limit>=len(details): return details
    # Evenly sample the sorted AOI X range so validation is not concentrated locally.
    ordered=sorted(details,key=lambda d:(float(d["image_context"]["x_mm"]),float(d["image_context"]["y_mm"])))
    if limit==1:return [ordered[len(ordered)//2]]
    ids=sorted({round(i*(len(ordered)-1)/(limit-1)) for i in range(limit)})
    return [ordered[i] for i in ids]


def validate(validation_json,limit=None):
    payload=json.loads(Path(validation_json).resolve().read_text(encoding="utf-8")); details=_sample_details(list(payload.get("results",[])),limit)
    if not details: raise ValueError("coordinate_validation.json has no results")
    jobs={}; groups=defaultdict(list)
    for d in details: groups[str(d["resources"]["odb_path"])].append(d)
    output={"assumption":"AOI filename coordinates are physical mm from a top-left origin (+X right, +Y down)","groups":[]}
    try:
        for odb_path,rows in groups.items():
            job,tmp=extract_input(Path(odb_path)); jobs[odb_path]=(job,tmp); renderer=FastODBRenderer(job,72.0)
            steps={p.name.lower() for p in (job/"steps").iterdir() if p.is_dir()}; root="pnl" if "pnl" in steps else sorted(steps)[0]
            pnl_bounds=[v*25.4 for v in renderer.profile_bounds(root)]
            instances=renderer.collect_instances(root); strips=[i for i in instances if i.step=="strip"]
            strip_meta=[]
            for idx,inst in enumerate(strips):
                b=_instance_bounds_mm(renderer,inst); strip_meta.append({"instance_index":idx,"bounds_mm":b,"top_left_pnl_mm":[b[0],b[3]],"width_mm":b[2]-b[0],"height_mm":b[3]-b[1],"transform":_transform_summary(inst)})
            image_results=[]; pnl_hits=0; strip_unique=0; strip_any=0
            for d in rows:
                info=d["image_context"]; ax=float(info["x_mm"]); ay=float(info["y_mm"]); pnl_point=_map_top_left(pnl_bounds,ax,ay); pnl_ok=_inside(pnl_point,pnl_bounds); pnl_hits+=int(pnl_ok)
                candidates=[]
                for sm in strip_meta:
                    point=_map_top_left(sm["bounds_mm"],ax,ay)
                    if _inside(point,sm["bounds_mm"]):
                        candidates.append({"strip_instance_index":sm["instance_index"],"odb_pnl_mm":list(point),"strip_bounds_mm":sm["bounds_mm"],"distance_to_strip_center_mm":math.hypot(point[0]-(sm["bounds_mm"][0]+sm["bounds_mm"][2])/2,point[1]-(sm["bounds_mm"][1]+sm["bounds_mm"][3])/2)})
                strip_any+=int(bool(candidates)); strip_unique+=int(len(candidates)==1)
                image_results.append({"image":info["image_path"],"aoi_mm":[ax,ay],"pnl_top_left":{"odb_pnl_mm":list(pnl_point),"inside_pnl":pnl_ok},"strip_top_left":{"candidate_count":len(candidates),"candidates":candidates}})
            n=len(rows); output["groups"].append({"odb_path":odb_path,"root_step":root,"pnl_bounds_mm":pnl_bounds,"pnl_top_left_mm":[pnl_bounds[0],pnl_bounds[3]],"strip_instance_count":len(strip_meta),"strip_instances":strip_meta,"summary":{"images":n,"pnl_origin_inside_ratio":pnl_hits/n,"strip_origin_any_candidate_ratio":strip_any/n,"strip_origin_unique_candidate_ratio":strip_unique/n},"images":image_results})
        return output
    finally:
        for job,tmp in jobs.values():
            if tmp is not None: tmp.cleanup()


def main():
    p=argparse.ArgumentParser(description="Geometry-only PNL-vs-STRIP AOI origin validator")
    p.add_argument("validation_json",type=Path); p.add_argument("--output",type=Path,default=Path("coordinate_geometry_validation.json")); p.add_argument("--limit",type=int,default=20,help="Spatially spread sample count; <=0 means all")
    a=p.parse_args(); result=validate(a.validation_json,a.limit); op=a.output.resolve(); op.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    for gi,g in enumerate(result["groups"],1):
        s=g["summary"]; print(f"[{gi}] {Path(g['odb_path']).name}"); print(f"  PNL bounds       : {g['pnl_bounds_mm']}"); print(f"  STRIP instances  : {g['strip_instance_count']}")
        for sm in g["strip_instances"]: print(f"    strip#{sm['instance_index']} bounds={['%.3f'%v for v in sm['bounds_mm']]} top-left={['%.3f'%v for v in sm['top_left_pnl_mm']]} axes X={sm['transform']['local_x_axis_in_pnl']} Y={sm['transform']['local_y_axis_in_pnl']}")
        print(f"  PNL-origin inside: {s['pnl_origin_inside_ratio']:.3f}"); print(f"  STRIP any hit    : {s['strip_origin_any_candidate_ratio']:.3f}"); print(f"  STRIP unique hit : {s['strip_origin_unique_candidate_ratio']:.3f}")
    print(f"Output: {op}"); return 0

if __name__=="__main__": raise SystemExit(main())
