#!/usr/bin/env python3
"""Fast coarse-to-fine AOI(panel-mm) -> ODB coordinate search.

AOI filename X/Y values are panel physical millimetres and are used directly as
initial centres. Spatial sampling can automatically choose panel-distributed
references. CAM search patches are cached by ODB/layer/resolution/physical
window so repeated searches can reuse identical renders when possible.
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
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Matching C_ reference not found for {g_path.name}")


def _axis_offsets(radius_mm: float, step_mm: float) -> list[float]:
    if radius_mm < 0 or step_mm <= 0:
        raise ValueError("radius must be >= 0 and step must be > 0")
    count = int(math.floor(radius_mm / step_mm + 1e-9))
    values = [i * step_mm for i in range(-count, count + 1)]
    if radius_mm > 0 and (not values or abs(values[0] + radius_mm) > 1e-9): values.insert(0, -radius_mm)
    if radius_mm > 0 and abs(values[-1] - radius_mm) > 1e-9: values.append(radius_mm)
    return sorted({round(v, 9) for v in values})


def _grid(cx: float, cy: float, radius: float, step: float):
    offsets = _axis_offsets(radius, step)
    return [(cx + dx, cy + dy) for dy in offsets for dx in offsets]


def _inside_panel(x: float, y: float, bounds):
    xmin, ymin, xmax, ymax = map(float, bounds)
    return xmin <= x <= xmax and ymin <= y <= ymax


def _parse_levels(values):
    out = []
    for raw in values:
        try: r, s = map(float, raw.split(":", 1))
        except Exception as exc: raise ValueError(f"Invalid search level {raw!r}; expected RADIUS:STEP") from exc
        if r < 0 or s <= 0: raise ValueError(f"Invalid search level {raw!r}")
        out.append((r, s))
    if not out: raise ValueError("At least one search level is required")
    return out


def _fmt_seconds(v):
    v = max(0.0, float(v))
    return f"{v:.1f}s" if v < 60 else (f"{v/60:.1f}m" if v < 3600 else f"{v/3600:.2f}h")


def _stage_resolution(native_um, step_mm): return max(float(native_um), min(20.0, float(step_mm) * 20.0))


def _resize_reference(reference, native_um, stage_um):
    scale = float(native_um) / float(stage_um)
    return ImageOps.grayscale(reference).resize((max(8, round(reference.width*scale)), max(8, round(reference.height*scale))), Image.Resampling.BILINEAR)


def _binary(image, invert=False):
    gray = ImageOps.autocontrast(ImageOps.grayscale(image))
    if invert: gray = ImageOps.invert(gray)
    return gray.point(lambda v: 255 if v >= 128 else 0, mode="L")


def _edge_mask(binary): return binary.filter(ImageFilter.FIND_EDGES).point(lambda v: 255 if v >= 32 else 0, mode="L")
def _count_on(binary): return int(binary.histogram()[255])
def _occupancy(binary): return _count_on(binary) / float(max(1, binary.width * binary.height))


def _tolerant_edge_dice(a, b):
    na, nb = _count_on(a), _count_on(b)
    if na + nb == 0: return 0.0
    da, db = a.filter(ImageFilter.MaxFilter(3)), b.filter(ImageFilter.MaxFilter(3))
    oa = _count_on(ImageChops.multiply(a, db)); ob = _count_on(ImageChops.multiply(b, da))
    return min(1.0, (oa + ob) / float(na + nb))


def _score_crop(cam_crop, reference_stage):
    cam_bin = _binary(cam_crop); cam_edge = _edge_mask(cam_bin); cam_occ = _occupancy(cam_bin)
    best_score, best_detail = -1.0, {}
    for mode, invert in (("normal", False), ("inverted", True)):
        ref_bin = _binary(reference_stage, invert); ref_edge = _edge_mask(ref_bin); ref_occ = _occupancy(ref_bin)
        occ_score = max(0.0, 1.0 - abs(cam_occ-ref_occ)); edge_score = _tolerant_edge_dice(cam_edge, ref_edge)
        score = .35*occ_score + .65*edge_score
        cam_solid = cam_occ <= .005 or cam_occ >= .995; ref_solid = ref_occ <= .005 or ref_occ >= .995
        if cam_solid and not ref_solid: score *= .02
        if score > best_score:
            best_score = score; best_detail = {"reference_mode":mode,"occupancy_score":round(occ_score,6),"edge_score":round(edge_score,6),"cam_occupancy":round(cam_occ,6),"reference_occupancy":round(ref_occ,6),"cam_solid_rejected":bool(cam_solid and not ref_solid)}
    return best_score, best_detail


def _patch_geometry(radius, ref, um):
    wmm=2*radius+ref.width*um/1000+2*um/1000; hmm=2*radius+ref.height*um/1000+2*um/1000
    return max(ref.width, math.ceil(wmm*1000/um)), max(ref.height, math.ceil(hmm*1000/um))


def _candidate_crop(patch, pcx, pcy, x, y, um, crop_size):
    ppm=1000/float(um); cx=patch.width/2+(x-pcx)*ppm; cy=patch.height/2-(y-pcy)*ppm; w,h=crop_size
    left=round(cx-w/2); top=round(cy-h/2); right=left+w; bottom=top+h
    if left<0 or top<0 or right>patch.width or bottom>patch.height: return None
    return patch.crop((left,top,right,bottom))


def _xy(detail):
    info=detail["image_context"]
    return float(info["x_mm"]), float(info["y_mm"])


def _spatial_sample(details, count):
    """Pick samples near four data-cloud corners plus centre, then farthest-point fill."""
    if count <= 0 or count >= len(details): return list(details)
    pts=[(_xy(d)[0],_xy(d)[1],i) for i,d in enumerate(details)]
    xs=[p[0] for p in pts]; ys=[p[1] for p in pts]; xmin,xmax=min(xs),max(xs); ymin,ymax=min(ys),max(ys)
    targets=[(xmin,ymin),(xmax,ymin),(xmin,ymax),(xmax,ymax),((xmin+xmax)/2,(ymin+ymax)/2)]
    chosen=[]
    for tx,ty in targets:
        if len(chosen)>=count: break
        candidates=[p for p in pts if p[2] not in chosen]
        if candidates: chosen.append(min(candidates,key=lambda p:(p[0]-tx)**2+(p[1]-ty)**2)[2])
    while len(chosen)<count:
        candidates=[p for p in pts if p[2] not in chosen]
        if not candidates: break
        def mind2(p):
            return min((p[0]-pts[i][0])**2+(p[1]-pts[i][1])**2 for i in chosen)
        chosen.append(max(candidates,key=mind2)[2])
    return [details[i] for i in chosen]


def _cache_key(odb_path, layer, x, y, um, w, h):
    return (str(Path(odb_path).resolve()),str(layer),round(float(x),6),round(float(y),6),round(float(um),6),int(w),int(h))


def main():
    p=argparse.ArgumentParser(description="Fast search of ODB coordinates around AOI panel-mm coordinates")
    p.add_argument("validation_json",type=Path); p.add_argument("--output",type=Path,default=Path("local_coordinate_search")); p.add_argument("--limit",type=int,default=None)
    p.add_argument("--spatial-samples",type=int,default=None,metavar="N",help="Automatically select N spatially distributed AOI images (corners + centre + farthest fill)")
    p.add_argument("--level",action="append",default=None,metavar="RADIUS:STEP"); p.add_argument("--save-search-patches",action="store_true")
    args=p.parse_args(); levels=_parse_levels(args.level or ["10:2","2:0.5","0.5:0.1","0.1:0.02"])
    payload=json.loads(args.validation_json.resolve().read_text(encoding="utf-8")); all_details=list(payload.get("results",[]))
    if not all_details: raise ValueError("coordinate_validation.json has no results")
    if args.spatial_samples is not None:
        details=_spatial_sample(all_details,max(1,args.spatial_samples)); selection_mode=f"spatial:{len(details)}"
    else:
        n=len(all_details) if args.limit is None else max(1,args.limit); details=all_details[:n]; selection_mode=f"sequential:{len(details)}"
    out=args.output.resolve(); out.mkdir(parents=True,exist_ok=True); report_path=out/"local_coordinate_search.json"
    report={"coordinate_assumption":"filename X/Y are panel physical millimetres and are used as the initial search centre","algorithm":"one CAM patch per stage with in-memory candidate scoring; shared identical-patch cache; one final native render","selection_mode":selection_mode,"selected_aoi_panel_mm":[list(_xy(d)) for d in details],"search_levels_mm":[{"radius":r,"step":s} for r,s in levels],"images":[]}
    patch_cache={}; extracted={}
    try:
        for image_no,detail in enumerate(details,1):
            started=time.perf_counter(); info,resources,ert=detail["image_context"],detail["resources"],detail["ert"]; g=Path(info["image_path"]); c=_find_reference(g)
            ax,ay=_xy(detail); native=float(ert["resolution_um_per_px"]); bounds=list(detail["pnl_bounds_mm"]); odb=str(resources["odb_path"]); layer=str(info["layer"])
            idir=out/f"{image_no:02d}_{g.stem}"; idir.mkdir(parents=True,exist_ok=True)
            with Image.open(c) as src: reference=ImageOps.grayscale(src); reference.load()
            refpath=idir/"REFERENCE_C.png"; reference.save(refpath)
            print(f"[{image_no}/{len(details)}] {g.name}  AOI=({ax:.3f},{ay:.3f})",flush=True)
            if odb not in extracted: extracted[odb]=extract_input(Path(odb))
            job,_tmp=extracted[odb]; cx,cy=ax,ay; stages=[]
            for sno,(radius,step) in enumerate(levels,1):
                st=time.perf_counter(); um=_stage_resolution(native,step); sref=_resize_reference(reference,native,um); pw,ph=_patch_geometry(radius,sref,um); cin=(cx,cy)
                key=_cache_key(odb,layer,cx,cy,um,pw,ph); cache_hit=key in patch_cache
                if cache_hit: patch,pmeta=patch_cache[key]
                else:
                    patch,pmeta=render_roi_cam(job,cx,cy,um,layer,width_px=pw,height_px=ph,signal_gv=255,drill_gv=125,return_components=False); patch_cache[key]=(patch.copy(),dict(pmeta))
                if args.save_search_patches: patch.save(idir/f"STAGE_{sno}_SEARCH_PATCH.png")
                candidates=[q for q in _grid(cx,cy,radius,step) if _inside_panel(q[0],q[1],bounds)]; best=(-1.0,cin,{}) ; scored=skipped=0
                for x,y in candidates:
                    crop=_candidate_crop(patch,cx,cy,x,y,um,sref.size)
                    if crop is None: skipped+=1; continue
                    score,sd=_score_crop(crop,sref); scored+=1
                    if score>best[0]: best=(score,(x,y),sd)
                if not scored: raise RuntimeError(f"Stage {sno} produced no scoreable candidates")
                cx,cy=best[1]; elapsed=time.perf_counter()-st
                stages.append({"stage":sno,"radius_mm":radius,"step_mm":step,"raster_resolution_um_per_px":um,"search_center_input_mm":list(cin),"patch_size_px":[pw,ph],"patch_cache_hit":cache_hit,"candidate_count":len(candidates),"scored_candidates":scored,"skipped_crops":skipped,"best_odb_mm":[cx,cy],"best_score":round(best[0],6),"score_detail":best[2],"elapsed_seconds":round(elapsed,3)})
                print(f"    stage {sno}: best=({cx:.3f},{cy:.3f}) score={best[0]:.4f} cache={'HIT' if cache_hit else 'MISS'} elapsed={_fmt_seconds(elapsed)}",flush=True)
                (idir/"INTERIM.json").write_text(json.dumps({"status":"running","aoi_panel_mm":[ax,ay],"current_best_odb_mm":[cx,cy],"completed_stages":stages},indent=2),encoding="utf-8")
            fkey=_cache_key(odb,layer,cx,cy,native,reference.width,reference.height); final_cache=fkey in patch_cache
            if final_cache: final,pmeta=patch_cache[fkey]
            else:
                final,pmeta=render_roi_cam(job,cx,cy,native,layer,width_px=reference.width,height_px=reference.height,signal_gv=255,drill_gv=125,return_components=False); patch_cache[fkey]=(final.copy(),dict(pmeta))
            score,sd=_score_crop(final,reference); bestpath=idir/"BEST_CAM.png"; final.save(bestpath); dx,dy=cx-ax,cy-ay
            row={"g_image":str(g),"c_reference":str(c),"reference_output":str(refpath),"best_cam_output":str(bestpath),"aoi_panel_mm":[ax,ay],"best_odb_mm":[cx,cy],"delta_mm":[dx,dy],"distance_from_aoi_mm":math.hypot(dx,dy),"resolution_um_per_px":native,"reference_size_px":[reference.width,reference.height],"final_score":round(score,6),"final_score_detail":sd,"final_cache_hit":final_cache,"signal_layer":pmeta["signal_layer"],"physical_signal_layer":pmeta["physical_signal_layer"],"drill_layers":list(pmeta["drill_layers_considered"]),"signal_nonzero":int(pmeta["signal_nonzero_pixels"]),"drill_nonzero":int(pmeta["drill_nonzero_pixels"]),"final_nonzero":int(pmeta["final_nonzero_pixels"]),"panel_bounds_mm":bounds,"elapsed_seconds":round(time.perf_counter()-started,3),"stages":stages}
            report["images"].append(row); report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
            print(f"    BEST=({cx:.6f},{cy:.6f}) delta=({dx:.6f},{dy:.6f}) score={score:.6f} total={_fmt_seconds(row['elapsed_seconds'])}",flush=True)
    finally:
        for _job,tmp in extracted.values():
            if tmp is not None: tmp.cleanup()
    report["cache_entries"]=len(patch_cache); report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8"); print(f"Report: {report_path}",flush=True); return 0


if __name__=="__main__": raise SystemExit(main())
