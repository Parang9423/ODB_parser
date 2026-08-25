#!/usr/bin/env python3
from __future__ import annotations

import math
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Dict, List, Optional

import app_core as core
from app_core import App as _CoreApp, LayerInfo
from hierarchy_renderer import FastODBRenderer, adaptive_preview_dpi
from odb_cam_renderer import CompositeLayer
from step_composite_renderer import render_selected_steps_composite

core.ODBRenderer = FastODBRenderer

HIER_COLORS = {"pnl": "#3b82f6", "strip": "#22c55e", "unit": "#f59e0b"}
MAX_PREVIEW_PIXELS = 12_000_000
ZOOM_STEP = 1.25


def default_spec(layer: LayerInfo) -> CompositeLayer:
    kind, name = layer.layer_type.upper(), layer.name.upper()
    if "DRILL" in kind or "DRILL" in name or name.startswith(("UV_", "TH_", "GDRILL_")):
        return CompositeLayer(layer.name, "REPLACE", 96)
    if "SOLDER_MASK" in kind or name.startswith("SM-"):
        return CompositeLayer(layer.name, "ADD", 160)
    if kind == "MIXED":
        return CompositeLayer(layer.name, "ADD", 220)
    return CompositeLayer(layer.name, "REPLACE", 255)


class LayerDialog(tk.Toplevel):
    """Metadata-only layer picker. Feature files are not opened here."""

    def __init__(self, parent: "App", layers: List[LayerInfo], selected: set[str]):
        super().__init__(parent)
        self.title("렌더링 레이어 선택")
        self.geometry("780x620")
        self.minsize(680, 460)
        self.transient(parent)
        self.grab_set()
        self.layers = layers
        self.result: Optional[List[str]] = None
        self.vars = {layer.name: tk.BooleanVar(value=layer.name in selected) for layer in layers}
        self.type_var = tk.StringVar(value="전체")
        self.search_var = tk.StringVar()

        root = ttk.Frame(self, padding=12)
        root.pack(fill="both", expand=True)
        ttk.Label(root, text="Matrix 메타데이터만 조회합니다. 체크한 Layer의 feature만 실제 렌더링 시 로드/캐시됩니다.").pack(anchor="w", pady=(0, 10))

        bar = ttk.Frame(root); bar.pack(fill="x", pady=(0, 8))
        ttk.Label(bar, text="Type").pack(side="left")
        types = ["전체"] + sorted({x.layer_type for x in layers})
        cb = ttk.Combobox(bar, textvariable=self.type_var, values=types, state="readonly", width=18)
        cb.pack(side="left", padx=(6, 14)); cb.bind("<<ComboboxSelected>>", lambda _e: self.refresh())
        ttk.Label(bar, text="검색").pack(side="left")
        ttk.Entry(bar, textvariable=self.search_var).pack(side="left", fill="x", expand=True, padx=(6, 0))
        self.search_var.trace_add("write", lambda *_: self.refresh())

        tools = ttk.Frame(root); tools.pack(fill="x", pady=(0, 6))
        ttk.Button(tools, text="표시 목록 전체 선택", command=lambda: self.set_visible(True)).pack(side="left")
        ttk.Button(tools, text="표시 목록 전체 해제", command=lambda: self.set_visible(False)).pack(side="left", padx=6)

        head = ttk.Frame(root); head.pack(fill="x")
        for col, (title, width) in enumerate([("사용", 6), ("Layer", 22), ("Type", 18), ("Context", 12), ("Side", 10), ("Polarity", 10)]):
            ttk.Label(head, text=title, width=width, anchor="w").grid(row=0, column=col, sticky="w")

        body = ttk.Frame(root); body.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(body, highlightthickness=0)
        scroll = ttk.Scrollbar(body, orient="vertical", command=self.canvas.yview)
        self.rows = ttk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.rows, anchor="nw", tags="rows")
        self.rows.bind("<Configure>", lambda _e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfigure("rows", width=e.width))
        self.canvas.configure(yscrollcommand=scroll.set)
        self.canvas.pack(side="left", fill="both", expand=True); scroll.pack(side="right", fill="y")

        buttons = ttk.Frame(root); buttons.pack(fill="x", pady=(10, 0))
        ttk.Button(buttons, text="취소", command=self.destroy).pack(side="right")
        ttk.Button(buttons, text="선택 레이어 로드", command=self.apply).pack(side="right", padx=6)
        self.refresh()

    def filtered(self) -> List[LayerInfo]:
        rows = self.layers
        if self.type_var.get() != "전체":
            rows = [x for x in rows if x.layer_type == self.type_var.get()]
        q = self.search_var.get().strip().lower()
        if q:
            rows = [x for x in rows if q in " ".join((x.name, x.layer_type, x.context, x.side, x.polarity)).lower()]
        return rows

    def refresh(self) -> None:
        for child in self.rows.winfo_children(): child.destroy()
        for r, layer in enumerate(self.filtered()):
            ttk.Checkbutton(self.rows, variable=self.vars[layer.name]).grid(row=r, column=0, sticky="w", pady=2)
            for c, (value, width) in enumerate([(layer.name,22),(layer.layer_type,18),(layer.context,12),(layer.side,10),(layer.polarity,10)], start=1):
                ttk.Label(self.rows, text=value, width=width).grid(row=r, column=c, sticky="w")

    def set_visible(self, value: bool) -> None:
        for layer in self.filtered(): self.vars[layer.name].set(value)

    def apply(self) -> None:
        selected = [x.name for x in self.layers if self.vars[x.name].get()]
        if not selected:
            messagebox.showwarning("레이어 선택", "하나 이상의 Layer를 선택하세요.", parent=self); return
        self.result = selected; self.destroy()


class App(_CoreApp):
    """PNL-root AOI viewer using lazy layer selection and Step visibility checkboxes."""

    def _ui(self) -> None:
        super()._ui()
        self.selected_specs: Dict[str, CompositeLayer] = {}
        self.show_pnl = tk.BooleanVar(value=True); self.show_strip = tk.BooleanVar(value=True); self.show_unit = tk.BooleanVar(value=True)
        self.edit_op = tk.StringVar(value="REPLACE"); self.edit_gv = tk.IntVar(value=255)
        self.pnl_coord_text = tk.StringVar(value="PNL X: - mm   Y: - mm")
        self._view_bounds = self._view_dpi_x = self._view_dpi_y = None; self._hier_profiles = []

        left = self.step_cb.master
        for widget in (self.step_cb, self.target_cb, self.layer_cb, self.comp_frame): widget.grid_remove()
        for child in left.winfo_children():
            if isinstance(child, ttk.Label):
                try:
                    if child.cget("text") in {"Step", "렌더링 대상", "Layer"}: child.grid_remove()
                except tk.TclError: pass

        ttk.Label(left, text="Step 표시", font=("TkDefaultFont", 11, "bold")).grid(row=1, column=0, columnspan=2, sticky="w")
        steps = ttk.Frame(left); steps.grid(row=2, column=0, columnspan=2, sticky="w", pady=(5, 8))
        self.pnl_check = ttk.Checkbutton(steps, text="PNL", variable=self.show_pnl, command=self.step_changed); self.pnl_check.pack(side="left")
        self.strip_check = ttk.Checkbutton(steps, text="STRIP", variable=self.show_strip, command=self.step_changed); self.strip_check.pack(side="left", padx=8)
        self.unit_check = ttk.Checkbutton(steps, text="UNIT", variable=self.show_unit, command=self.step_changed); self.unit_check.pack(side="left")

        layer_box = ttk.LabelFrame(left, text="선택 레이어", padding=6); layer_box.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        self.layer_tree = ttk.Treeview(layer_box, columns=("name","type","side","op","gv"), show="headings", height=6, selectmode="browse")
        for col, title, width in [("name","Layer",105),("type","Type",85),("side","Side",55),("op","Operation",75),("gv","GV",40)]:
            self.layer_tree.heading(col, text=title); self.layer_tree.column(col, width=width, anchor="center" if col in {"side","op","gv"} else "w")
        self.layer_tree.pack(fill="x"); self.layer_tree.bind("<<TreeviewSelect>>", lambda _e: self.load_edit())
        edit = ttk.Frame(layer_box); edit.pack(fill="x", pady=(5,0))
        ttk.Combobox(edit, textvariable=self.edit_op, values=["ADD","REPLACE","SUBTRACT"], state="readonly", width=10).pack(side="left")
        ttk.Spinbox(edit, from_=0, to=255, textvariable=self.edit_gv, width=6).pack(side="left", padx=4)
        ttk.Button(edit, text="설정 적용", command=self.apply_edit).pack(side="left")

        top = self.file_label.master
        self.layer_pick_btn = ttk.Button(top, text="레이어 선택/변경", command=self.choose_layers, state="disabled")
        self.layer_pick_btn.pack(side="left", padx=(0,8), before=self.file_label)
        ttk.Label(top, textvariable=self.pnl_coord_text).pack(side="right")

        self.canvas.bind("<Motion>", self.on_motion, add="+")
        self.canvas.bind("<Leave>", lambda _e: self.pnl_coord_text.set("PNL X: - mm   Y: - mm"), add="+")
        self.render_btn.configure(text="Composite 결과 저장")

    def apply_loaded(self, path, job, tmp, info) -> None:
        if self.tmp: self.tmp.cleanup()
        self.job, self.tmp, self.info, self.source = job, tmp, info, path
        self.selected_specs.clear(); self.preview = None; self._hier_profiles = []
        self.file_label.configure(text=path.name); self.layer_pick_btn.configure(state="normal")
        available = {x.lower() for x in info.steps}
        for name, var, widget in [("pnl",self.show_pnl,self.pnl_check),("strip",self.show_strip,self.strip_check),("unit",self.show_unit,self.unit_check)]:
            exists = name in available; var.set(exists); widget.configure(state="normal" if exists else "disabled")
        self.render_btn.configure(state="disabled"); self.refresh_layer_tree(); self.update_info()
        self.status.set(f"메타데이터 로드 완료: {info.name} / {len(info.layers)} layers. 필요한 Layer를 선택하세요.")
        self.after_idle(self.choose_layers)

    def choose_layers(self) -> None:
        if not self.info: return
        dialog = LayerDialog(self, self.info.layers, set(self.selected_specs)); self.wait_window(dialog)
        if dialog.result is None: return
        old, meta = self.selected_specs, {x.name:x for x in self.info.layers}; new: Dict[str, CompositeLayer] = {}
        for name in dialog.result: new[name] = old.get(name, default_spec(meta[name]))
        self.selected_specs = new; self.refresh_layer_tree(); self.update_info(); self.render_btn.configure(state="normal")
        self.schedule_preview()

    def refresh_layer_tree(self) -> None:
        for item in self.layer_tree.get_children(): self.layer_tree.delete(item)
        if not self.info: return
        meta = {x.name:x for x in self.info.layers}
        for name, spec in self.selected_specs.items():
            layer = meta[name]; self.layer_tree.insert("", "end", iid=name, values=(name,layer.layer_type,layer.side,spec.operation,spec.gv))

    def load_edit(self) -> None:
        sel = self.layer_tree.selection()
        if sel:
            spec = self.selected_specs[sel[0]]; self.edit_op.set(spec.operation); self.edit_gv.set(spec.gv)

    def apply_edit(self) -> None:
        sel = self.layer_tree.selection()
        if not sel: messagebox.showwarning("레이어 설정", "수정할 Layer를 선택하세요."); return
        try:
            gv = int(self.edit_gv.get()); assert 0 <= gv <= 255
            spec = CompositeLayer(sel[0], self.edit_op.get().upper(), gv).normalized()
        except Exception: messagebox.showerror("GV 오류", "GV는 0~255 사이의 정수여야 합니다."); return
        self.selected_specs[sel[0]] = spec; self.refresh_layer_tree(); self.layer_tree.selection_set(sel[0]); self.schedule_preview()

    def visible_steps(self) -> set[str]:
        if not self.info: return set()
        available = {x.lower() for x in self.info.steps}; visible = set()
        if self.show_pnl.get() and "pnl" in available: visible.add("pnl")
        if self.show_strip.get() and "strip" in available: visible.add("strip")
        if self.show_unit.get() and "unit" in available: visible.add("unit")
        return visible

    def root_step(self) -> str:
        available = {x.lower() for x in self.info.steps} if self.info else set()
        return next((x for x in ("pnl","strip","unit") if x in available), next(iter(available), ""))

    def step_changed(self) -> None:
        if not self.visible_steps() and self.info:
            for name, var in (("pnl",self.show_pnl),("strip",self.show_strip),("unit",self.show_unit)):
                if name in {x.lower() for x in self.info.steps}: var.set(True); break
        self.update_info()
        if self.selected_specs: self.schedule_preview()

    def schedule_preview(self) -> None:
        if not self.job or not self.selected_specs: return
        root, visible = self.root_step(), self.visible_steps()
        if not root or not visible: return
        self.token += 1; token = self.token; self.render_btn.configure(state="disabled")
        requested = float(self.preview_dpi); probe = FastODBRenderer(self.job, requested); bounds = probe.profile_bounds(root)
        effective = adaptive_preview_dpi(bounds, requested, MAX_PREVIEW_PIXELS); specs = list(self.selected_specs.values()); job = self.job
        self.status.set(f"미리보기 생성 중 | {len(specs)} layers | {effective:.0f} DPI" + (f" (요청 {requested:g})" if effective < requested-.5 else ""))
        def work() -> None:
            try:
                renderer = FastODBRenderer(job, effective); image = render_selected_steps_composite(renderer, root, specs, visible)
                profiles = [(i.step,i.depth,renderer.transformed_profile(i)) for i in renderer.collect_instances(root) if i.step in visible]
                if token == self.token:
                    self._view_bounds, self._view_dpi_x, self._view_dpi_y, self._hier_profiles = bounds, renderer.dpi_x, renderer.dpi_y, profiles
                self.q.put(("preview", token, image, renderer.stats, effective))
            except Exception as exc: self.q.put(("error", token, f"미리보기 생성 실패: {exc}"))
        threading.Thread(target=work, daemon=True).start()

    def render(self) -> None:
        if not self.job or not self.selected_specs: return
        root, visible, specs = self.root_step(), self.visible_steps(), list(self.selected_specs.values())
        try:
            if self.output_mode.get() == core.MODE_DPI:
                dpi = int(self.dpi.get()); assert core.MIN_DPI <= dpi <= core.MAX_DPI
                args=("dpi",dpi,dpi); suffix=f"{dpi}dpi"; desc=f"DPI: {dpi}"
            else:
                x,y=self.valid_um(); args=("aoi",x,y); suffix=f"{x:g}x{y:g}umpp".replace(".","p"); desc=f"AOI X={x:g}, Y={y:g} µm/pixel"
        except Exception: messagebox.showerror("출력 설정 오류", "출력 DPI/AOI 해상도를 확인하세요."); return
        step_suffix="-".join(step.upper() for step in ("pnl","strip","unit") if step in visible)
        output=filedialog.asksaveasfilename(title="CAM Image 저장",defaultextension=".png",initialfile=f"{self.info.name}_{step_suffix}_composite_{suffix}.png",filetypes=[("PNG Image","*.png")])
        if not output:return
        output_path=Path(output); job=self.job; token=self.token; self.render_btn.configure(state="disabled"); self.status.set(f"최종 Composite CAM 렌더링 중... Steps={step_suffix}")
        def work() -> None:
            try:
                renderer=FastODBRenderer(job,args[1]) if args[0]=="dpi" else FastODBRenderer.from_um_per_pixel(job,args[1],args[2])
                image=render_selected_steps_composite(renderer,root,specs,visible); output_path.parent.mkdir(parents=True,exist_ok=True); image.save(output_path,optimize=True,dpi=(renderer.dpi_x,renderer.dpi_y))
                self.q.put(("rendered",image,output_path,renderer.stats,desc+f" | Steps: {step_suffix}"))
            except Exception as exc:self.q.put(("error",token,f"렌더링 실패: {exc}"))
        threading.Thread(target=work,daemon=True).start()

    def draw(self) -> None:
        super().draw()
        if self.preview and self._hier_profiles and self._view_bounds: self.draw_profiles()

    def image_origin(self):
        w,h=self.preview.width*self.zoom,self.preview.height*self.zoom; cw,ch=max(1,self.canvas.winfo_width()),max(1,self.canvas.winfo_height())
        return cw/2+self.offx-w/2,ch/2+self.offy-h/2,w,h

    def root_to_canvas(self,x,y):
        xmin,_ymin,_xmax,ymax=self._view_bounds; left,top,_w,_h=self.image_origin(); return left+(x-xmin)*self._view_dpi_x*self.zoom, top+(ymax-y)*self._view_dpi_y*self.zoom

    def draw_profiles(self) -> None:
        self.canvas.delete("hier_profile")
        for step,depth,contours in self._hier_profiles:
            color=HIER_COLORS.get(step,"#a855f7"); width=2 if depth==0 else 1
            for _kind,points in contours:
                if len(points)<2:continue
                xy=[]
                for x,y in points: xy.extend(self.root_to_canvas(x,y))
                if points[0]!=points[-1]:xy.extend(self.root_to_canvas(*points[0]))
                self.canvas.create_line(*xy,fill=color,width=width,tags="hier_profile")
        self.canvas.tag_raise("hier_profile")

    def on_motion(self,event) -> None:
        if not self.preview or not self._view_bounds:self.pnl_coord_text.set("PNL X: - mm   Y: - mm");return
        left,top,w,h=self.image_origin()
        if not(left<=event.x<=left+w and top<=event.y<=top+h):self.pnl_coord_text.set("PNL X: - mm   Y: - mm");return
        ix,iy=(event.x-left)/self.zoom,(event.y-top)/self.zoom; xmin,_ymin,_xmax,ymax=self._view_bounds
        x=xmin+ix/self._view_dpi_x; y=ymax-iy/self._view_dpi_y; self.pnl_coord_text.set(f"PNL X: {x*25.4:.3f} mm   Y: {y*25.4:.3f} mm")

    def update_info(self) -> None:
        if not getattr(self,"info_txt",None) or not self.info:return
        visible=", ".join(x.upper() for x in ("pnl","strip","unit") if x in self.visible_steps())
        types: Dict[str,int]={}
        for layer in self.info.layers:
            if layer.name in self.selected_specs:types[layer.layer_type]=types.get(layer.layer_type,0)+1
        lines=[f"Job: {self.info.name}",f"Source: {self.source.name if self.source else '-'}",f"Steps: {', '.join(x.upper() for x in self.info.steps)}",f"Matrix layers: {len(self.info.layers)}",f"Selected layers: {len(self.selected_specs)}",f"Visible steps: {visible or '-'}", "", "Selected types: "+(", ".join(f"{k}={v}" for k,v in types.items()) or "-"), "", "Lazy loading:", "Matrix/Step metadata → Layer 선택 → 선택 feature만 로드/캐시"]
        self.info_txt.configure(state="normal");self.info_txt.delete("1.0","end");self.info_txt.insert("1.0","\n".join(lines));self.info_txt.configure(state="disabled")


if __name__ == "__main__": App().mainloop()
