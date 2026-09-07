#!/usr/bin/env python3
"""Analyze GIDS G/C filename suffixes without CAM rendering.

Parses G_<Y>_<X>_<suffix> image names and reports whether the final token is
associated with spatial zones, panel folders, AOI layers, or final mapping
success/failure. This is diagnostic only; it does not assume suffix semantics.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

PATTERN = re.compile(r"^[GC]_(-?\d+(?:\.\d+)?)_(-?\d+(?:\.\d+)?)_([^._]+)\.(?:jpg|jpeg|png|bmp)$", re.I)


def parse_name(path: Path):
    m=PATTERN.match(path.name)
    if not m: return None
    return {"y_mm":float(m.group(1)),"x_mm":float(m.group(2)),"suffix":m.group(3)}


def _stats(vals):
    vals=list(vals)
    if not vals: return None
    return {"min":min(vals),"max":max(vals),"mean":sum(vals)/len(vals),"span":max(vals)-min(vals)}


def _path_context(root:Path,p:Path):
    try: parts=p.relative_to(root).parts[:-1]
    except ValueError: parts=p.parts[:-1]
    return {"relative_parent":"/".join(parts),"parent":p.parent.name,"grandparent":p.parent.parent.name if len(p.parents)>1 else ""}


def main():
    ap=argparse.ArgumentParser(description="Analyze spatial meaning of GIDS filename suffix")
    ap.add_argument("gids_root",type=Path,help="Root containing GIDS images")
    ap.add_argument("--validation-json",type=Path,help="Optional final_aoi_odb_validation.json to correlate score/status")
    ap.add_argument("--output",type=Path,default=Path("gids_suffix_analysis"))
    ap.add_argument("--match-threshold",type=float,default=0.8)
    args=ap.parse_args(); root=args.gids_root.resolve(); out=args.output.resolve(); out.mkdir(parents=True,exist_ok=True)

    rows=[]
    for p in root.rglob("*"):
        if not p.is_file() or not p.name.upper().startswith("G_"): continue
        parsed=parse_name(p)
        if parsed is None: continue
        ctx=_path_context(root,p)
        rows.append({"path":str(p),**parsed,**ctx})
    if not rows: raise ValueError(f"No parseable G images under {root}")

    validation={}
    if args.validation_json:
        payload=json.loads(args.validation_json.resolve().read_text(encoding="utf-8"))
        for r in payload.get("results",[]):
            key=Path(r["g_image"]).name.casefold(); score=float(r.get("score",0.0))
            validation[key]={"sample":r.get("sample"),"score":score,"status":"MATCH" if score>=args.match_threshold else "MISMATCH"}
    for r in rows:
        v=validation.get(Path(r["path"]).name.casefold(),{})
        r.update({"validation_sample":v.get("sample"),"validation_score":v.get("score"),"validation_status":v.get("status")})

    by_suffix=defaultdict(list)
    for r in rows: by_suffix[r["suffix"]].append(r)
    suffix_summary=[]
    for suffix,items in sorted(by_suffix.items(),key=lambda kv:(-len(kv[1]),kv[0])):
        status=Counter(r["validation_status"] for r in items if r["validation_status"])
        parents=Counter(r["parent"] for r in items); grands=Counter(r["grandparent"] for r in items)
        suffix_summary.append({"suffix":suffix,"count":len(items),"x_mm":_stats(r["x_mm"] for r in items),"y_mm":_stats(r["y_mm"] for r in items),"unique_parent_count":len(parents),"top_parents":parents.most_common(10),"unique_grandparent_count":len(grands),"top_grandparents":grands.most_common(10),"validation_match":status.get("MATCH",0),"validation_mismatch":status.get("MISMATCH",0)})

    # Spatial bins expose whether suffixes partition the AOI frame.
    xmin=min(r["x_mm"] for r in rows); xmax=max(r["x_mm"] for r in rows); ymin=min(r["y_mm"] for r in rows); ymax=max(r["y_mm"] for r in rows)
    bins=defaultdict(Counter)
    for r in rows:
        bx=min(4,max(0,int((r["x_mm"]-xmin)/max(1e-12,xmax-xmin)*5)))
        by=min(4,max(0,int((r["y_mm"]-ymin)/max(1e-12,ymax-ymin)*5)))
        bins[f"Y{by}_X{bx}"][r["suffix"]]+=1
    spatial_bins={k:v.most_common() for k,v in sorted(bins.items())}

    with (out/"gids_suffix_rows.csv").open("w",newline="",encoding="utf-8-sig") as f:
        fields=list(rows[0].keys()); w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    payload={"summary":{"root":str(root),"image_count":len(rows),"suffix_count":len(by_suffix),"observed_bounds_mm":{"x":[xmin,xmax],"y":[ymin,ymax]},"validation_json":str(args.validation_json.resolve()) if args.validation_json else None,"note":"Suffix meaning is not assumed; distributions are observational."},"suffixes":suffix_summary,"spatial_5x5_bins":spatial_bins}
    jp=out/"gids_suffix_analysis.json"; jp.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")

    print(f"G images: {len(rows)}  suffixes: {len(by_suffix)}")
    print(f"AOI bounds X={xmin:.3f}..{xmax:.3f} Y={ymin:.3f}..{ymax:.3f}")
    print("\nTop suffix groups:")
    for s in suffix_summary[:30]:
        print(f"  {s['suffix']:>6} n={s['count']:5d} X={s['x_mm']['min']:.3f}..{s['x_mm']['max']:.3f} Y={s['y_mm']['min']:.3f}..{s['y_mm']['max']:.3f} validation={s['validation_match']} OK/{s['validation_mismatch']} NG")
    print(f"\nOutput: {jp}")
    return 0

if __name__=="__main__": raise SystemExit(main())
