#!/usr/bin/env python3
"""Check whether failed AOI references use an unexpected ODB matrix layer.

This diagnostic keeps the existing AOI->ODB coordinate fixed. It first inspects
all ODB matrix layers for primitives intersecting the exact C-reference ROI, then
rasterizes only candidate layers with intersecting primitives. This avoids the
very expensive approach of rendering every layer blindly.

Use a failed sample (e.g. S03) together with a known-good sample (e.g. S05).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from app_core import inspect_job
from hierarchy_renderer import FastODBRenderer
from odb_cam_renderer import extract_input
from render.roi import _feature_diagnostics, _render_layer_mask, roi_bounds_in, select_roi_layers
from validation_utils import _score_crop

VISIBLE_STEPS=("pnl","strip","unit")


def _root_step(job:Path)->str:
    steps=sorted(p.name.lower() for p in (job/"steps").iterdir() if p.is_dir())
    if not steps: raise RuntimeError("ODB has no steps")
    return "pnl" if "pnl" in steps else steps[0]


def _visible(job:Path)->tuple[str,...]:
    available={p.name.lower() for p in (job/"steps").iterdir() if p.is_dir()}
    return tuple(s for s in VISIBLE_STEPS if s in available)


def _primitive_total(diag:dict)->int:
    c=diag.get("roi_primitive_counts",{})
    return int(c.get("pads",0))+int(c.get("lines",0))+int(c.get("surfaces",0))


def _nonzero(im:Image.Image)->int:
    h=im.histogram(); return int(sum(h[1:])) if h else 0


def _safe_name(name:str)->str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)


def _save_montage(reference:Image.Image, images:list[tuple[str,Image.Image,float]], path:Path)->None:
    ref=ImageOps.grayscale(reference); tiles=[("REFERENCE",ref,None)]+[(n,ImageOps.grayscale(i),s) for n,i,s in images]
    label_h=24; tile_w=max(i.width for _,i,_ in tiles); tile_h=max(i.height for _,i,_ in tiles)
    cols=min(4,len(tiles)); rows=(len(tiles)+cols-1)//cols
    canvas=Image.new("L",(cols*tile_w,rows*(tile_h+label_h)),0); draw=ImageDraw.Draw(canvas); font=ImageFont.load_default()
    for idx,(name,img,score) in enumerate(tiles):
        cx=(idx%cols)*tile_w; cy=(idx//cols)*(tile_h+label_h)
        canvas.paste(img,(cx,cy+label_h)); label=name if score is None else f"{name} score={score:.4f}"
        draw.text((cx+3,cy+5),label,fill=255,font=font)
    canvas.save(path)


def main()->int:
    ap=argparse.ArgumentParser(description="Sweep all ODB matrix layers at fixed AOI->ODB coordinates")
    ap.add_argument("validation_json",type=Path,help="final_aoi_odb_validation.json")
    ap.add_argument("--samples",default="3,5",help="Comma-separated sample numbers; default failed S03 + good S05")
    ap.add_argument("--output",type=Path,default=Path("layer_sweep_validation"))
    ap.add_argument("--max-render-layers",type=int,default=30,help="Safety cap after primitive prefilter")
    args=ap.parse_args()

    src=args.validation_json.resolve(); payload=json.loads(src.read_text(encoding="utf-8")); all_rows=list(payload.get("results",[]))
    wanted={int(v.strip()) for v in str(args.samples).split(",") if v.strip()}; rows=[r for r in all_rows if int(r["sample"]) in wanted]
    if not rows: raise ValueError("No requested samples found")
    out=args.output.resolve(); out.mkdir(parents=True,exist_ok=True); started=time.perf_counter(); results=[]; failures=[]

    for pos,row in enumerate(rows,1):
        sample=int(row["sample"]); print(f"[{pos}/{len(rows)}] S{sample:02d} layer sweep",flush=True)
        try:
            with Image.open(Path(row["reference_c"])) as im:
                reference=ImageOps.grayscale(im); reference.load(); reference=reference.copy()
            odb_path=Path(row["odb_path"]); odb_x=float(row["odb_x_mm"]); odb_y=float(row["odb_y_mm"]); resolution=float(row["resolution_um_per_px"]); recipe_layer=str(row["layer"])
            job,tmp=extract_input(odb_path)
            try:
                info=inspect_job(job); renderer=FastODBRenderer.from_um_per_pixel(job,resolution,resolution); root=_root_step(job); visible=_visible(job)
                bounds=roi_bounds_in(odb_x,odb_y,resolution,reference.width,reference.height)
                selected=select_roi_layers(job,recipe_layer)
                layer_rows=[]
                print(f"  matrix layers={len(info.layers)}; prefiltering exact {reference.width}x{reference.height} ROI...",flush=True)
                for li,layer in enumerate(info.layers,1):
                    diag=_feature_diagnostics(renderer,root,layer.name,visible,bounds,resolution,max_samples=5)
                    primitives=_primitive_total(diag)
                    layer_rows.append({"name":layer.name,"type":layer.layer_type,"context":layer.context,"side":layer.side,"polarity":layer.polarity,"primitive_count":primitives,"primitive_counts":diag.get("roi_primitive_counts",{}),"unsupported_symbols":diag.get("unsupported_symbols",{})})
                    if primitives: print(f"    candidate {layer.name} [{layer.layer_type}/{layer.context}/{layer.side}] primitives={primitives}",flush=True)
                candidates=[r for r in layer_rows if r["primitive_count"]>0]
                candidates.sort(key=lambda r:(-r["primitive_count"],r["name"]))
                if len(candidates)>args.max_render_layers:
                    print(f"  WARNING candidates={len(candidates)} > cap={args.max_render_layers}; rendering top by primitive count",flush=True)
                    render_candidates=candidates[:args.max_render_layers]
                else: render_candidates=candidates
                rendered=[]
                for ci,cand in enumerate(render_candidates,1):
                    name=cand["name"]; print(f"  render {ci}/{len(render_candidates)} {name}",flush=True)
                    mask=_render_layer_mask(renderer,root,name,visible,bounds,reference.width,reference.height)
                    nz=_nonzero(mask); score,detail=_score_crop(mask,reference)
                    op=out/f"S{sample:02d}_{_safe_name(name)}.png"; mask.save(op)
                    cand.update({"rendered":True,"nonzero_pixels":nz,"score":float(score),"edge_score":float(detail.get("edge_score") or 0.0),"occupancy_score":float(detail.get("occupancy_score") or 0.0),"reference_mode":detail.get("reference_mode"),"output":str(op)})
                    rendered.append((name,mask,float(score)))
                for lr in layer_rows:
                    lr.setdefault("rendered",False)
                rendered.sort(key=lambda x:x[2],reverse=True)
                montage=out/f"S{sample:02d}_LAYER_SWEEP_MONTAGE.png"; _save_montage(reference,rendered[:15],montage)
                best=max((r for r in layer_rows if r.get("rendered")),key=lambda r:r.get("score",-1.0),default=None)
                result={"sample":sample,"g_image":row["g_image"],"reference_c":row["reference_c"],"odb_path":str(odb_path),"aoi_mm":[row["aoi_x_mm"],row["aoi_y_mm"]],"odb_mm":[odb_x,odb_y],"resolution_um_per_px":resolution,"roi_size_px":[reference.width,reference.height],"roi_physical_size_mm":[reference.width*resolution/1000.0,reference.height*resolution/1000.0],"current_recipe_layer":recipe_layer,"current_selected_signal_layer":selected.signal_layer,"current_selected_drill_layers":list(selected.drill_layers),"candidate_count":len(candidates),"rendered_candidate_count":len(render_candidates),"best_layer":best["name"] if best else None,"best_score":best.get("score") if best else None,"montage_output":str(montage),"layers":layer_rows}
                results.append(result)
                print(f"  candidates={len(candidates)} best={result['best_layer']} score={result['best_score']}",flush=True)
            finally:
                if tmp is not None: tmp.cleanup()
        except Exception as exc:
            failures.append({"sample":sample,"error":f"{type(exc).__name__}: {exc}"}); print(f"  FAIL {type(exc).__name__}: {exc}",flush=True)

    output={"summary":{"algorithm":"all ODB matrix-layer primitive prefilter + exact ROI render at fixed coordinate; no coordinate search","source_validation_json":str(src),"sample_count":len(rows),"success_count":len(results),"failure_count":len(failures),"elapsed_seconds":time.perf_counter()-started},"results":results,"failures":failures}
    jp=out/"layer_sweep_validation.json"; jp.write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding="utf-8"); print(f"Output: {jp}"); return 0

if __name__=="__main__": raise SystemExit(main())
