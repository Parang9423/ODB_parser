#!/usr/bin/env python3
"""Visual final diagnostic for fixed SWAP_X+_Y- AOI -> ODB mapping.

Consumes final_aoi_odb_validation.json. For each unique ODB/layer pair it renders
one low-resolution full PNL CAM image, then overlays every validation coordinate.
The expensive full-panel render is therefore performed once per ODB/layer, not
once per sample. Existing C reference images are not modified.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from hierarchy_renderer import FastODBRenderer, adaptive_preview_dpi
from odb_cam_renderer import extract_input


def _pnl_bounds_mm(renderer: FastODBRenderer, root: str) -> list[float]:
    return [float(v) * 25.4 for v in renderer.profile_bounds(root)]


def _root_step(job: Path) -> str:
    steps = sorted(p.name.lower() for p in (job / "steps").iterdir() if p.is_dir())
    if not steps:
        raise RuntimeError("ODB has no steps")
    return "pnl" if "pnl" in steps else steps[0]


def _point_to_pixel(x_mm: float, y_mm: float, bounds_mm: list[float], size: tuple[int, int]) -> tuple[float, float]:
    xmin, ymin, xmax, ymax = bounds_mm
    w, h = size
    px = (x_mm - xmin) / max(1e-12, xmax - xmin) * max(1, w - 1)
    py = (ymax - y_mm) / max(1e-12, ymax - ymin) * max(1, h - 1)
    return px, py


def _draw_marker(draw: ImageDraw.ImageDraw, p: tuple[float, float], label: str, matched: bool) -> None:
    x, y = p
    r = 10
    # Shape, not colour, carries status so the output remains interpretable in grayscale.
    if matched:
        draw.ellipse((x-r, y-r, x+r, y+r), outline=255, width=3)
    else:
        draw.line((x-r, y-r, x+r, y+r), fill=255, width=4)
        draw.line((x-r, y+r, x+r, y-r), fill=255, width=4)
    draw.rectangle((x+r+3, y-9, x+r+47, y+10), fill=0, outline=255, width=1)
    draw.text((x+r+6, y-7), label, fill=255, font=ImageFont.load_default())


def _crop_context(panel: Image.Image, p: tuple[float, float], radius_px: int) -> Image.Image:
    x, y = map(int, map(round, p))
    left=max(0,x-radius_px); top=max(0,y-radius_px); right=min(panel.width,x+radius_px); bottom=min(panel.height,y+radius_px)
    crop=panel.crop((left,top,right,bottom))
    draw=ImageDraw.Draw(crop); cx=x-left; cy=y-top
    draw.line((max(0,cx-12),cy,min(crop.width-1,cx+12),cy),fill=255,width=2)
    draw.line((cx,max(0,cy-12),cx,min(crop.height-1,cy+12)),fill=255,width=2)
    return crop


def main() -> int:
    ap=argparse.ArgumentParser(description="Render one full PNL CAM and overlay final AOI->ODB validation points")
    ap.add_argument("validation_json",type=Path,help="final_aoi_odb_validation.json")
    ap.add_argument("--output",type=Path,default=Path("full_panel_mapping_diagnostic"))
    ap.add_argument("--preview-um-per-px",type=float,default=100.0,help="Requested diagnostic full-panel resolution; default 100 um/px")
    ap.add_argument("--max-pixels",type=int,default=12_000_000,help="Safety cap for each full-panel image")
    ap.add_argument("--match-threshold",type=float,default=0.8)
    ap.add_argument("--context-mm",type=float,default=20.0,help="Optional context crop physical width/height")
    args=ap.parse_args()

    payload=json.loads(args.validation_json.resolve().read_text(encoding="utf-8"))
    rows=list(payload.get("results",[]))
    if not rows: raise ValueError("Validation JSON has no results")
    out=args.output.resolve(); out.mkdir(parents=True,exist_ok=True)

    groups=defaultdict(list)
    for row in rows:
        key=(str(Path(row["odb_path"]).resolve()).casefold(),str(row["layer"]).casefold())
        groups[key].append(row)

    report={"source_json":str(args.validation_json.resolve()),"mapping":payload.get("summary",{}).get("mapping"),"groups":[]}
    total_started=time.perf_counter()
    for gi,((_odb_key,layer_key),items) in enumerate(groups.items(),1):
        odb=Path(items[0]["odb_path"]); layer=str(items[0]["layer"])
        print(f"[{gi}/{len(groups)}] Full panel render: {odb.name} / {layer}",flush=True)
        job,tmp=extract_input(odb)
        try:
            probe=FastODBRenderer(job,72.0); root=_root_step(job); bounds_mm=_pnl_bounds_mm(probe,root)
            requested_dpi=25.4*1000.0/max(1e-9,args.preview_um_per_px)
            bounds_in=tuple(v/25.4 for v in bounds_mm)
            dpi=adaptive_preview_dpi(bounds_in,requested_dpi,max_pixels=max(1,args.max_pixels),min_dpi=20.0)
            t0=time.perf_counter(); renderer=FastODBRenderer(job,dpi); panel=renderer.render(root,layer); render_seconds=time.perf_counter()-t0
        finally:
            if tmp is not None: tmp.cleanup()

        base_name=f"G{gi:02d}_{odb.stem}_{layer}"
        raw_path=out/f"{base_name}_FULL_PANEL.png"; panel.save(raw_path)
        marked=panel.copy(); draw=ImageDraw.Draw(marked)
        point_rows=[]
        mm_per_px_x=(bounds_mm[2]-bounds_mm[0])/max(1,panel.width-1)
        mm_per_px_y=(bounds_mm[3]-bounds_mm[1])/max(1,panel.height-1)
        context_radius=max(20,int(round((args.context_mm/2.0)/max(mm_per_px_x,mm_per_px_y))))
        context_radius=min(context_radius,max(panel.width,panel.height))
        for row in items:
            p=_point_to_pixel(float(row["odb_x_mm"]),float(row["odb_y_mm"]),bounds_mm,panel.size)
            matched=float(row.get("score",0.0))>=args.match_threshold
            label=f"S{int(row['sample']):02d} {'OK' if matched else 'NG'}"
            _draw_marker(draw,p,label,matched)
            context=_crop_context(panel,p,context_radius)
            context_path=out/f"S{int(row['sample']):02d}_{Path(row['g_image']).stem}_PANEL_CONTEXT.png"; context.save(context_path)
            point_rows.append({"sample":row["sample"],"g_image":row["g_image"],"aoi_mm":[row["aoi_x_mm"],row["aoi_y_mm"]],"odb_mm":[row["odb_x_mm"],row["odb_y_mm"]],"panel_pixel":[p[0],p[1]],"score":row.get("score"),"status":"MATCH" if matched else "MISMATCH","context_output":str(context_path)})
        marked_path=out/f"{base_name}_FULL_PANEL_MARKED.png"; marked.save(marked_path)
        report["groups"].append({"odb_path":str(odb),"layer":layer,"root_step":root,"pnl_bounds_mm":bounds_mm,"panel_size_px":list(panel.size),"effective_dpi":dpi,"effective_um_per_px_x":mm_per_px_x*1000.0,"effective_um_per_px_y":mm_per_px_y*1000.0,"full_panel_render_seconds":render_seconds,"full_panel_output":str(raw_path),"marked_panel_output":str(marked_path),"points":point_rows})
        print(f"  size={panel.width}x{panel.height} dpi={dpi:.2f} render={render_seconds:.1f}s -> {marked_path}",flush=True)

    report["elapsed_seconds"]=time.perf_counter()-total_started
    report_path=out/"full_panel_mapping_diagnostic.json"; report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"Output: {report_path}")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
