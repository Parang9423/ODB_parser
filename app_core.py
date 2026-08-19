#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import queue
import threading
import tkinter as tk
from dataclasses import asdict, dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk
from typing import Dict, List, Optional

from PIL import Image, ImageOps, ImageTk

from odb_cam_renderer import (
    CompositeLayer,
    ODBError,
    ODBRenderer,
    contours_bounds,
    extract_input,
    parse_kv_blocks,
    parse_profile_contours,
)

PREVIEW_DPI_DEFAULT = 600
OUTPUT_DPI_DEFAULT = 1200
MIN_DPI = 72
MAX_DPI = 10000
MIN_UM = 0.1
MAX_UM = 1000.0
ZOOM_MIN = 0.05
ZOOM_MAX = 16.0
ZOOM_STEP = 1.25
MODE_DPI = "DPI"
MODE_AOI = "AOI 해상도"
TARGET_SINGLE = "단일 Layer"
TARGET_COMPOSITE = "Composite"
CUSTOM = "사용자 지정"
CONFIG_DIR = Path.home() / ".odb_cam_renderer"
PROFILE_FILE = CONFIG_DIR / "aoi_profiles.json"
COMPOSITE_FILE = CONFIG_DIR / "composite_presets.json"


@dataclass(frozen=True)
class LayerInfo:
    name: str
    layer_type: str = "?"
    context: str = "?"
    side: str = "?"
    polarity: str = "?"

    @property
    def label(self) -> str:
        side = f" / {self.side}" if self.side not in {"", "?"} else ""
        return f"{self.name}  [{self.layer_type}{side}]"


@dataclass
class JobInfo:
    name: str
    steps: List[str]
    layers: List[LayerInfo]


def inspect_job(job: Path) -> JobInfo:
    steps_dir = job / "steps"
    if not steps_dir.is_dir():
        raise ODBError("ODB++ steps directory is missing")
    steps = sorted(p.name for p in steps_dir.iterdir() if p.is_dir())
    layers: List[LayerInfo] = []
    matrix = job / "matrix" / "matrix"
    if matrix.exists():
        for block in parse_kv_blocks(matrix, "LAYER"):
            if block.get("NAME"):
                layers.append(LayerInfo(block["NAME"], block.get("TYPE", "?"), block.get("CONTEXT", "?"), block.get("SIDE", "?"), block.get("POLARITY", "?")))
    return JobInfo(job.name, steps, layers)


def profile_size_mm(job: Path, step: str) -> tuple[float, float]:
    bounds = contours_bounds(parse_profile_contours(job / "steps" / step.lower() / "profile"))
    return (bounds[2] - bounds[0]) * 25.4, (bounds[3] - bounds[1]) * 25.4


def load_json(path: Path) -> Dict[str, dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except Exception:
        return {}


def save_json(path: Path, data: Dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("ODB++ CAM Image Renderer")
        self.geometry("1480x900")
        self.minsize(1180, 720)
        self.job: Optional[Path] = None
        self.tmp = None
        self.info: Optional[JobInfo] = None
        self.source: Optional[Path] = None
        self.preview: Optional[Image.Image] = None
        self.photo: Optional[ImageTk.PhotoImage] = None
        self.zoom = 1.0
        self.offx = self.offy = 0.0
        self.pan = None
        self.preview_dpi = PREVIEW_DPI_DEFAULT
        self.profiles = load_json(PROFILE_FILE)
        self.composite_presets = load_json(COMPOSITE_FILE)
        self.token = 0
        self.q: queue.Queue = queue.Queue()
        self.step = tk.StringVar()
        self.layer = tk.StringVar()
        self.output_mode = tk.StringVar(value=MODE_DPI)
        self.target_mode = tk.StringVar(value=TARGET_SINGLE)
        self.dpi = tk.IntVar(value=OUTPUT_DPI_DEFAULT)
        self.profile = tk.StringVar(value=CUSTOM)
        self.umx = tk.DoubleVar(value=10.0)
        self.umy = tk.DoubleVar(value=10.0)
        self.lock = tk.BooleanVar(value=True)
        self.zoom_text = tk.StringVar(value="100%")
        self.status = tk.StringVar(value="ODB++ TGZ 파일을 열어주세요.")
        self.comp_preset = tk.StringVar(value=CUSTOM)
        self.comp_layer = tk.StringVar()
        self.comp_op = tk.StringVar(value="REPLACE")
        self.comp_gv = tk.IntVar(value=255)
        self._menu()
        self._ui()
        self._output_mode_changed()
        self._target_mode_changed()
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.after(100, self.poll)

    def _menu(self) -> None:
        menu = tk.Menu(self)
        file_menu = tk.Menu(menu, tearoff=False)
        file_menu.add_command(label="ODB++ 파일 열기...", accelerator="Ctrl+O", command=self.open_file)
        file_menu.add_separator()
        file_menu.add_command(label="종료", command=self.close)
        menu.add_cascade(label="파일", menu=file_menu)
        settings = tk.Menu(menu, tearoff=False)
        settings.add_command(label="미리보기 설정...", command=self.preview_settings)
        menu.add_cascade(label="설정", menu=settings)
        self.config(menu=menu)
        self.bind_all("<Control-o>", lambda _e: self.open_file())

    def _ui(self) -> None:
        root = ttk.Frame(self, padding=12)
        root.pack(fill="both", expand=True)
        top = ttk.Frame(root)
        top.pack(fill="x", pady=(0, 10))
        ttk.Button(top, text="파일 업로드", command=self.open_file).pack(side="left")
        self.file_label = ttk.Label(top, text="선택된 파일 없음")
        self.file_label.pack(side="left", padx=12)
        panes = ttk.Panedwindow(root, orient="horizontal")
        panes.pack(fill="both", expand=True)
        left = ttk.Frame(panes, padding=(4, 8))
        right = ttk.Frame(panes, padding=(8, 8))
        panes.add(left, weight=0)
        panes.add(right, weight=1)
        ttk.Label(left, text="렌더링 설정", font=("TkDefaultFont", 12, "bold")).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 12))
        ttk.Label(left, text="Step").grid(row=1, column=0, sticky="w", pady=5)
        self.step_cb = ttk.Combobox(left, textvariable=self.step, state="disabled", width=28)
        self.step_cb.grid(row=1, column=1, sticky="ew")
        self.step_cb.bind("<<ComboboxSelected>>", lambda _e: self.selection())
        ttk.Label(left, text="렌더링 대상").grid(row=2, column=0, sticky="w", pady=5)
        self.target_cb = ttk.Combobox(left, textvariable=self.target_mode, values=[TARGET_SINGLE, TARGET_COMPOSITE], state="readonly", width=18)
        self.target_cb.grid(row=2, column=1, sticky="w")
        self.target_cb.bind("<<ComboboxSelected>>", lambda _e: self._target_mode_changed())
        ttk.Label(left, text="Layer").grid(row=3, column=0, sticky="w", pady=5)
        self.layer_cb = ttk.Combobox(left, textvariable=self.layer, state="disabled", width=28)
        self.layer_cb.grid(row=3, column=1, sticky="ew")
        self.layer_cb.bind("<<ComboboxSelected>>", lambda _e: self.selection())

        self.comp_frame = ttk.LabelFrame(left, text="Composite 설정", padding=8)
        self.comp_frame.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(8, 6))
        ttk.Label(self.comp_frame, text="Preset").grid(row=0, column=0, sticky="w")
        self.comp_preset_cb = ttk.Combobox(self.comp_frame, textvariable=self.comp_preset, state="readonly", width=21)
        self.comp_preset_cb.grid(row=0, column=1, columnspan=2, sticky="ew")
        self.comp_preset_cb.bind("<<ComboboxSelected>>", lambda _e: self.use_composite_preset())
        ttk.Button(self.comp_frame, text="저장", width=6, command=self.save_composite_preset).grid(row=0, column=3, padx=(4, 0))
        cols = ("order", "layer", "op", "gv")
        self.comp_tree = ttk.Treeview(self.comp_frame, columns=cols, show="headings", height=7, selectmode="browse")
        for col, title in zip(cols, ["#", "Layer", "Operation", "GV"]):
            self.comp_tree.heading(col, text=title)
        self.comp_tree.column("order", width=35, anchor="center", stretch=False)
        self.comp_tree.column("layer", width=130)
        self.comp_tree.column("op", width=80, anchor="center")
        self.comp_tree.column("gv", width=45, anchor="center")
        self.comp_tree.grid(row=1, column=0, columnspan=4, sticky="ew", pady=(8, 6))
        self.comp_tree.bind("<<TreeviewSelect>>", lambda _e: self._load_selected_composite_row())
        self.comp_layer_cb = ttk.Combobox(self.comp_frame, textvariable=self.comp_layer, state="readonly", width=18)
        self.comp_layer_cb.grid(row=2, column=0, sticky="ew")
        self.comp_op_cb = ttk.Combobox(self.comp_frame, textvariable=self.comp_op, values=["ADD", "REPLACE", "SUBTRACT"], state="readonly", width=10)
        self.comp_op_cb.grid(row=2, column=1, padx=4)
        self.comp_gv_sp = ttk.Spinbox(self.comp_frame, from_=0, to=255, textvariable=self.comp_gv, width=6)
        self.comp_gv_sp.grid(row=2, column=2)
        ttk.Button(self.comp_frame, text="추가/수정", command=self.upsert_composite_row).grid(row=2, column=3, padx=(4, 0))
        row_btns = ttk.Frame(self.comp_frame)
        row_btns.grid(row=3, column=0, columnspan=4, sticky="ew", pady=(6, 0))
        ttk.Button(row_btns, text="삭제", command=self.remove_composite_row).pack(side="left")
        ttk.Button(row_btns, text="↑", width=3, command=lambda: self.move_composite_row(-1)).pack(side="left", padx=(5, 2))
        ttk.Button(row_btns, text="↓", width=3, command=lambda: self.move_composite_row(1)).pack(side="left")
        ttk.Label(row_btns, text="  ADD=max / REPLACE=덮어쓰기 / SUBTRACT=0", foreground="#666666").pack(side="left")

        ttk.Label(left, text="출력 방식").grid(row=5, column=0, sticky="w", pady=5)
        self.mode_cb = ttk.Combobox(left, textvariable=self.output_mode, values=[MODE_DPI, MODE_AOI], state="readonly", width=18)
        self.mode_cb.grid(row=5, column=1, sticky="w")
        self.mode_cb.bind("<<ComboboxSelected>>", lambda _e: self._output_mode_changed())
        ttk.Label(left, text="출력 DPI").grid(row=6, column=0, sticky="w", pady=5)
        self.dpi_sp = ttk.Spinbox(left, from_=MIN_DPI, to=MAX_DPI, increment=50, textvariable=self.dpi, width=12)
        self.dpi_sp.grid(row=6, column=1, sticky="w")
        self.aoi = ttk.LabelFrame(left, text="AOI 해상도 설정", padding=8)
        self.aoi.grid(row=7, column=0, columnspan=2, sticky="ew", pady=(8, 4))
        ttk.Label(self.aoi, text="설비 프로파일").grid(row=0, column=0, sticky="w")
        self.profile_cb = ttk.Combobox(self.aoi, textvariable=self.profile, state="readonly", width=20)
        self.profile_cb.grid(row=0, column=1)
        self.profile_cb.bind("<<ComboboxSelected>>", lambda _e: self.use_profile())
        ttk.Label(self.aoi, text="X (µm/pixel)").grid(row=1, column=0, sticky="w", pady=3)
        self.xsp = ttk.Spinbox(self.aoi, from_=MIN_UM, to=MAX_UM, increment=.1, textvariable=self.umx, width=12)
        self.xsp.grid(row=1, column=1, sticky="w")
        ttk.Label(self.aoi, text="Y (µm/pixel)").grid(row=2, column=0, sticky="w", pady=3)
        self.ysp = ttk.Spinbox(self.aoi, from_=MIN_UM, to=MAX_UM, increment=.1, textvariable=self.umy, width=12)
        self.ysp.grid(row=2, column=1, sticky="w")
        ttk.Checkbutton(self.aoi, text="X/Y 동일 해상도", variable=self.lock, command=self.lock_changed).grid(row=3, column=0, columnspan=2, sticky="w")
        ttk.Button(self.aoi, text="현재 값 프로파일 저장", command=self.save_profile).grid(row=4, column=0, columnspan=2, sticky="ew", pady=(5, 0))
        self.umx.trace_add("write", lambda *_: self.sync_y())
        self.refresh_profiles()
        self.refresh_composite_presets()
        self.render_btn = ttk.Button(left, text="렌더링 결과 저장", command=self.render, state="disabled")
        self.render_btn.grid(row=8, column=0, columnspan=2, sticky="ew", pady=(14, 12))
        ttk.Separator(left).grid(row=9, column=0, columnspan=2, sticky="ew")
        ttk.Label(left, text="ODB 정보", font=("TkDefaultFont", 10, "bold")).grid(row=10, column=0, columnspan=2, sticky="w", pady=(8, 4))
        self.info_txt = tk.Text(left, width=43, height=13, wrap="word", state="disabled", relief="flat")
        self.info_txt.grid(row=11, column=0, columnspan=2, sticky="nsew")
        left.rowconfigure(11, weight=1)

        hdr = ttk.Frame(right)
        hdr.pack(fill="x", pady=(0, 8))
        ttk.Label(hdr, text="CAM 미리보기", font=("TkDefaultFont", 12, "bold")).pack(side="left")
        tools = ttk.Frame(hdr)
        tools.pack(side="right")
        ttk.Button(tools, text="−", width=3, command=lambda: self.change_zoom(1 / ZOOM_STEP)).pack(side="left")
        ttk.Label(tools, textvariable=self.zoom_text, width=7, anchor="center").pack(side="left")
        ttk.Button(tools, text="+", width=3, command=lambda: self.change_zoom(ZOOM_STEP)).pack(side="left")
        ttk.Button(tools, text="맞춤", command=self.fit).pack(side="left", padx=(6, 0))
        ttk.Button(tools, text="100%", command=self.actual).pack(side="left", padx=(4, 0))
        canvas_frame = ttk.Frame(right)
        canvas_frame.pack(fill="both", expand=True)
        canvas_frame.rowconfigure(0, weight=1)
        canvas_frame.columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(canvas_frame, background="#202020", highlightthickness=0, cursor="fleur")
        self.canvas.grid(row=0, column=0, sticky="nsew")
        hs = ttk.Scrollbar(canvas_frame, orient="horizontal", command=self.canvas.xview)
        vs = ttk.Scrollbar(canvas_frame, orient="vertical", command=self.canvas.yview)
        hs.grid(row=1, column=0, sticky="ew")
        vs.grid(row=0, column=1, sticky="ns")
        self.canvas.configure(xscrollcommand=hs.set, yscrollcommand=vs.set)
        self.canvas.bind("<Configure>", lambda _e: self.draw())
        self.canvas.bind("<MouseWheel>", self.wheel)
        self.canvas.bind("<Button-4>", self.wheel)
        self.canvas.bind("<Button-5>", self.wheel)
        self.canvas.bind("<ButtonPress-1>", self.pan_start)
        self.canvas.bind("<B1-Motion>", self.pan_move)
        self.canvas.bind("<ButtonRelease-1>", self.pan_end)
        ttk.Label(root, textvariable=self.status, anchor="w", relief="sunken").pack(fill="x", pady=(10, 0))

    def refresh_profiles(self) -> None:
        values = [CUSTOM] + sorted(self.profiles)
        self.profile_cb["values"] = values
        if self.profile.get() not in values:
            self.profile.set(CUSTOM)

    def valid_um(self) -> tuple[float, float]:
        try:
            x, y = float(self.umx.get()), float(self.umy.get())
        except Exception as exc:
            raise ValueError("AOI 해상도는 숫자여야 합니다.") from exc
        if not (MIN_UM <= x <= MAX_UM and MIN_UM <= y <= MAX_UM):
            raise ValueError(f"AOI 해상도는 {MIN_UM}~{MAX_UM} µm/pixel 범위여야 합니다.")
        return x, y

    def sync_y(self) -> None:
        if self.lock.get():
            try:
                self.umy.set(self.umx.get())
            except tk.TclError:
                pass
        self.update_info()

    def lock_changed(self) -> None:
        if self.lock.get():
            self.umy.set(self.umx.get())
            self.ysp.configure(state="disabled")
        elif self.output_mode.get() == MODE_AOI:
            self.ysp.configure(state="normal")
        self.update_info()

    def _output_mode_changed(self) -> None:
        aoi_mode = self.output_mode.get() == MODE_AOI
        self.dpi_sp.configure(state="disabled" if aoi_mode else "normal")
        for widget in self.aoi.winfo_children():
            try:
                widget.configure(state="normal" if aoi_mode else "disabled")
            except tk.TclError:
                pass
        if aoi_mode:
            self.profile_cb.configure(state="readonly")
            self.lock_changed()
        self.update_info()

    def save_profile(self) -> None:
        try:
            x, y = self.valid_um()
        except ValueError as exc:
            messagebox.showerror("AOI 해상도 오류", str(exc))
            return
        name = simpledialog.askstring("설비 프로파일 저장", "설비/프로파일 이름을 입력하세요.", parent=self)
        if not name:
            return
        name = name.strip()
        self.profiles[name] = {"um_per_pixel_x": x, "um_per_pixel_y": y}
        save_json(PROFILE_FILE, self.profiles)
        self.refresh_profiles()
        self.profile.set(name)
        self.status.set(f"프로파일 저장: {name}")

    def use_profile(self) -> None:
        preset = self.profiles.get(self.profile.get())
        if not preset:
            return
        x = float(preset["um_per_pixel_x"])
        y = float(preset.get("um_per_pixel_y", x))
        self.lock.set(abs(x - y) < 1e-12)
        self.umx.set(x)
        self.umy.set(y)
        self.lock_changed()

    def refresh_composite_presets(self) -> None:
        values = [CUSTOM] + sorted(self.composite_presets)
        self.comp_preset_cb["values"] = values
        if self.comp_preset.get() not in values:
            self.comp_preset.set(CUSTOM)

    def _target_mode_changed(self) -> None:
        composite = self.target_mode.get() == TARGET_COMPOSITE
        self.layer_cb.configure(state="disabled" if composite or not self.info else "readonly")
        for child in self.comp_frame.winfo_children():
            try:
                child.configure(state="normal" if composite else "disabled")
            except tk.TclError:
                pass
        if composite:
            self.comp_preset_cb.configure(state="readonly")
            self.comp_layer_cb.configure(state="readonly")
            self.comp_op_cb.configure(state="readonly")
            if not self.comp_tree.get_children() and self.selected_layer():
                self._append_composite_row(self.selected_layer().name, "REPLACE", 255)
        self.update_info()
        if self.job:
            self.schedule_preview()

    def _append_composite_row(self, layer: str, op: str, gv: int) -> None:
        self.comp_tree.insert("", "end", values=(len(self.comp_tree.get_children()) + 1, layer, op, gv))

    def _renumber_composite_rows(self) -> None:
        for idx, item in enumerate(self.comp_tree.get_children(), start=1):
            values = list(self.comp_tree.item(item, "values"))
            values[0] = idx
            self.comp_tree.item(item, values=values)

    def _load_selected_composite_row(self) -> None:
        selected = self.comp_tree.selection()
        if not selected:
            return
        values = self.comp_tree.item(selected[0], "values")
        self.comp_layer.set(values[1])
        self.comp_op.set(values[2])
        self.comp_gv.set(int(values[3]))

    def upsert_composite_row(self) -> None:
        layer = self.comp_layer.get().strip()
        if not layer:
            messagebox.showwarning("Composite", "추가할 Layer를 선택하세요.")
            return
        try:
            gv = int(self.comp_gv.get())
            if not 0 <= gv <= 255:
                raise ValueError
        except Exception:
            messagebox.showerror("GV 오류", "GV는 0~255 사이의 정수여야 합니다.")
            return
        op = self.comp_op.get().upper()
        selected = self.comp_tree.selection()
        if selected:
            order = self.comp_tree.item(selected[0], "values")[0]
            self.comp_tree.item(selected[0], values=(order, layer, op, gv))
        else:
            self._append_composite_row(layer, op, gv)
        self.comp_preset.set(CUSTOM)
        self._renumber_composite_rows()
        self.update_info()
        self.schedule_preview()

    def remove_composite_row(self) -> None:
        selected = self.comp_tree.selection()
        if not selected:
            return
        self.comp_tree.delete(selected[0])
        self._renumber_composite_rows()
        self.comp_preset.set(CUSTOM)
        self.update_info()
        self.schedule_preview()

    def move_composite_row(self, delta: int) -> None:
        selected = self.comp_tree.selection()
        if not selected:
            return
        item = selected[0]
        children = list(self.comp_tree.get_children())
        index = children.index(item)
        new_index = max(0, min(len(children) - 1, index + delta))
        if new_index == index:
            return
        self.comp_tree.move(item, "", new_index)
        self.comp_tree.selection_set(item)
        self._renumber_composite_rows()
        self.comp_preset.set(CUSTOM)
        self.update_info()
        self.schedule_preview()

    def composite_specs(self) -> List[CompositeLayer]:
        specs: List[CompositeLayer] = []
        for item in self.comp_tree.get_children():
            values = self.comp_tree.item(item, "values")
            specs.append(CompositeLayer(str(values[1]), str(values[2]), int(values[3])).normalized())
        return specs

    def save_composite_preset(self) -> None:
        specs = self.composite_specs()
        if not specs:
            messagebox.showwarning("Composite", "저장할 Composite Layer가 없습니다.")
            return
        name = simpledialog.askstring("Composite Preset 저장", "Preset 이름을 입력하세요.", parent=self)
        if not name:
            return
        name = name.strip()
        self.composite_presets[name] = {"layers": [asdict(spec) for spec in specs]}
        save_json(COMPOSITE_FILE, self.composite_presets)
        self.refresh_composite_presets()
        self.comp_preset.set(name)
        self.status.set(f"Composite preset 저장: {name}")

    def use_composite_preset(self) -> None:
        preset = self.composite_presets.get(self.comp_preset.get())
        if not preset:
            return
        for item in self.comp_tree.get_children():
            self.comp_tree.delete(item)
        valid_layers = {layer.name for layer in self.info.layers} if self.info else None
        for row in preset.get("layers", []):
            layer = str(row.get("layer", ""))
            if valid_layers is not None and layer not in valid_layers:
                continue
            self._append_composite_row(layer, str(row.get("operation", "ADD")).upper(), int(row.get("gv", 255)))
        self._renumber_composite_rows()
        self.update_info()
        self.schedule_preview()

    def preview_settings(self) -> None:
        dialog = tk.Toplevel(self)
        dialog.title("설정")
        dialog.transient(self)
        dialog.grab_set()
        frame = ttk.Frame(dialog, padding=18)
        frame.pack()
        ttk.Label(frame, text="미리보기 DPI").grid(row=0, column=0, padx=(0, 16))
        value = tk.StringVar(value=str(self.preview_dpi))
        spin = ttk.Spinbox(frame, from_=MIN_DPI, to=MAX_DPI, increment=50, textvariable=value, width=12)
        spin.grid(row=0, column=1)
        def apply() -> None:
            try:
                dpi = int(value.get())
                if not MIN_DPI <= dpi <= MAX_DPI:
                    raise ValueError
            except Exception:
                messagebox.showerror("DPI 오류", f"{MIN_DPI}~{MAX_DPI} 범위의 정수를 입력하세요.", parent=dialog)
                return
            changed = dpi != self.preview_dpi
            self.preview_dpi = dpi
            dialog.destroy()
            self.update_info()
            if changed and self.job:
                self.schedule_preview()
        buttons = ttk.Frame(frame)
        buttons.grid(row=1, column=0, columnspan=2, sticky="e", pady=(14, 0))
        ttk.Button(buttons, text="취소", command=dialog.destroy).pack(side="right")
        ttk.Button(buttons, text="적용", command=apply).pack(side="right", padx=6)

    def open_file(self) -> None:
        path = filedialog.askopenfilename(title="ODB++ TGZ 파일 선택", filetypes=[("ODB++ TGZ", "*.tgz"), ("Tar GZip", "*.tar.gz"), ("모든 파일", "*.*")], defaultextension=".tgz")
        if path:
            self.load(Path(path))

    def load(self, path: Path) -> None:
        self.status.set("ODB++ 파일을 분석하는 중...")
        self.token += 1
        token = self.token
        def work() -> None:
            try:
                job, tmp = extract_input(path)
                self.q.put(("loaded", token, path, job, tmp, inspect_job(job)))
            except Exception as exc:
                self.q.put(("error", token, f"파일 로드 실패: {exc}"))
        threading.Thread(target=work, daemon=True).start()

    def apply_loaded(self, path: Path, job: Path, tmp, info: JobInfo) -> None:
        if self.tmp:
            self.tmp.cleanup()
        self.job, self.tmp, self.info, self.source = job, tmp, info, path
        self.file_label.configure(text=path.name)
        self.step_cb["values"] = info.steps
        self.step.set("unit" if "unit" in info.steps else info.steps[0])
        self.step_cb.configure(state="readonly")
        labels = [layer.label for layer in info.layers]
        self.layer_cb["values"] = labels
        index = next((i for i, layer in enumerate(info.layers) if layer.name.lower() == "l1"), 0)
        if labels:
            self.layer_cb.current(index)
        self.comp_layer_cb["values"] = [layer.name for layer in info.layers]
        if info.layers:
            self.comp_layer.set(info.layers[index].name)
        self.render_btn.configure(state="normal")
        self._target_mode_changed()
        self.update_info()
        self.schedule_preview()

    def selected_layer(self) -> Optional[LayerInfo]:
        return next((layer for layer in self.info.layers if layer.label == self.layer.get()), None) if self.info else None

    def selection(self) -> None:
        self.update_info()
        self.schedule_preview()

    def schedule_preview(self) -> None:
        if not self.job:
            return
        step = self.step.get()
        if not step:
            return
        if self.target_mode.get() == TARGET_COMPOSITE and not self.comp_tree.get_children():
            return
        self.token += 1
        token = self.token
        dpi = self.preview_dpi
        self.render_btn.configure(state="disabled")
        target = "Composite" if self.target_mode.get() == TARGET_COMPOSITE else (self.selected_layer().name if self.selected_layer() else "-")
        self.status.set(f"미리보기 생성 중: {step}/{target} @ {dpi} DPI")
        specs = self.composite_specs() if self.target_mode.get() == TARGET_COMPOSITE else None
        layer_name = self.selected_layer().name if self.selected_layer() else None
        job = self.job
        def work() -> None:
            try:
                renderer = ODBRenderer(job, dpi)
                image = renderer.render_composite(step, specs) if specs is not None else renderer.render(step, layer_name)
                self.q.put(("preview", token, image, renderer.stats, dpi))
            except Exception as exc:
                self.q.put(("error", token, f"미리보기 생성 실패: {exc}"))
        threading.Thread(target=work, daemon=True).start()

    def poll(self) -> None:
        try:
            while True:
                result = self.q.get_nowait()
                kind = result[0]
                if kind == "loaded":
                    _, token, path, job, tmp, info = result
                    if token == self.token:
                        self.apply_loaded(path, job, tmp, info)
                    elif tmp:
                        tmp.cleanup()
                elif kind == "preview":
                    _, token, image, stats, dpi = result
                    if token == self.token:
                        self.preview = image
                        self.fit()
                        self.render_btn.configure(state="normal")
                        self.status.set(f"미리보기 완료 | {image.width}×{image.height}px @ {dpi} DPI | warnings={stats.unsupported}")
                elif kind == "rendered":
                    _, image, output, stats, desc = result
                    self.render_btn.configure(state="normal")
                    self.status.set(f"렌더링 완료: {output}")
                    messagebox.showinfo("렌더링 완료", f"{output}\n\n{desc}\n크기: {image.width} × {image.height}px\nWarnings: {stats.unsupported}")
                elif kind == "error":
                    _, token, msg = result
                    if token == self.token:
                        self.render_btn.configure(state="normal" if self.job else "disabled")
                        self.status.set(msg)
                        messagebox.showerror("오류", msg)
        except queue.Empty:
            pass
        self.after(100, self.poll)

    def fit(self) -> None:
        if not self.preview:
            self.draw()
            return
        self.zoom = max(ZOOM_MIN, min(ZOOM_MAX, min(max(1, self.canvas.winfo_width() - 32) / self.preview.width, max(1, self.canvas.winfo_height() - 32) / self.preview.height)))
        self.offx = self.offy = 0.0
        self.draw()

    def actual(self) -> None:
        self.zoom = 1.0
        self.offx = self.offy = 0.0
        self.draw()

    def change_zoom(self, factor: float, x=None, y=None) -> None:
        if not self.preview:
            return
        old = self.zoom
        new = max(ZOOM_MIN, min(ZOOM_MAX, old * factor))
        cw, ch = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
        ax, ay = (cw / 2 if x is None else x), (ch / 2 if y is None else y)
        ix = (ax - cw / 2 - self.offx) / old
        iy = (ay - ch / 2 - self.offy) / old
        self.zoom = new
        self.offx = ax - cw / 2 - ix * new
        self.offy = ay - ch / 2 - iy * new
        self.draw()

    def wheel(self, event) -> str:
        self.change_zoom(ZOOM_STEP if getattr(event, "num", None) == 4 or getattr(event, "delta", 0) > 0 else 1 / ZOOM_STEP, event.x, event.y)
        return "break"

    def pan_start(self, event) -> None:
        self.pan = (event.x, event.y)

    def pan_move(self, event) -> None:
        if not self.pan:
            return
        x, y = self.pan
        self.offx += event.x - x
        self.offy += event.y - y
        self.pan = (event.x, event.y)
        self.draw()

    def pan_end(self, _event) -> None:
        self.pan = None

    def draw(self) -> None:
        self.canvas.delete("all")
        if not self.preview:
            self.canvas.create_text(max(10, self.canvas.winfo_width() // 2), max(10, self.canvas.winfo_height() // 2), text="ODB++ 파일을 열고 Layer를 선택하면 미리보기가 표시됩니다.", fill="white")
            return
        width = max(1, round(self.preview.width * self.zoom))
        height = max(1, round(self.preview.height * self.zoom))
        resample = Image.Resampling.NEAREST if self.zoom >= 1 else Image.Resampling.LANCZOS
        image = self.preview.resize((width, height), resample)
        self.photo = ImageTk.PhotoImage(ImageOps.colorize(image, black="black", white="white"))
        cw, ch = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
        self.canvas.create_image(cw / 2 + self.offx, ch / 2 + self.offy, image=self.photo, anchor="center", tags="p")
        bbox = self.canvas.bbox("p")
        if bbox:
            self.canvas.configure(scrollregion=(min(0, bbox[0] - 100), min(0, bbox[1] - 100), max(cw, bbox[2] + 100), max(ch, bbox[3] + 100)))
        self.zoom_text.set(f"{self.zoom * 100:.0f}%")

    def update_info(self) -> None:
        if not hasattr(self, "info_txt") or not self.job or not self.info:
            return
        step = self.step.get()
        lines = [f"Job: {self.info.name}", f"Source: {self.source.name}", f"Step: {step}"]
        wh = None
        try:
            wh = profile_size_mm(self.job, step)
            lines.append(f"Profile: {wh[0]:.3f} × {wh[1]:.3f} mm")
        except Exception:
            pass
        lines.extend(["", f"Target: {self.target_mode.get()}"])
        if self.target_mode.get() == TARGET_SINGLE:
            layer = self.selected_layer()
            if layer:
                lines.extend([f"Layer: {layer.name}", f"Type: {layer.layer_type}", f"Side: {layer.side}"])
        else:
            specs = self.composite_specs()
            lines.append(f"Composite layers: {len(specs)}")
            for index, spec in enumerate(specs, start=1):
                lines.append(f"  {index}. {spec.layer} / {spec.operation} / GV {spec.gv}")
        lines.extend(["", f"Preview DPI: {self.preview_dpi}", f"Output Mode: {self.output_mode.get()}"])
        if self.output_mode.get() == MODE_DPI:
            lines.append(f"Output DPI: {self.dpi.get()}")
        else:
            try:
                x, y = self.valid_um()
                lines.extend([f"AOI X: {x:g} µm/pixel", f"AOI Y: {y:g} µm/pixel", f"Equivalent DPI: X={25400/x:.2f}, Y={25400/y:.2f}"])
                if wh:
                    lines.append(f"Expected pixels: ≈ {math.ceil(wh[0]*1000/x)} × {math.ceil(wh[1]*1000/y)}")
            except ValueError:
                lines.append("AOI resolution: invalid")
        self.info_txt.configure(state="normal")
        self.info_txt.delete("1.0", "end")
        self.info_txt.insert("1.0", "\n".join(lines))
        self.info_txt.configure(state="disabled")

    def render(self) -> None:
        if not self.job or not self.info:
            return
        step = self.step.get()
        specs = self.composite_specs() if self.target_mode.get() == TARGET_COMPOSITE else None
        layer = self.selected_layer()
        if specs is not None and not specs:
            messagebox.showwarning("Composite", "Composite Layer를 하나 이상 추가하세요.")
            return
        if specs is None and not layer:
            return
        try:
            if self.output_mode.get() == MODE_DPI:
                dpi = int(self.dpi.get())
                if not MIN_DPI <= dpi <= MAX_DPI:
                    raise ValueError(f"DPI는 {MIN_DPI}~{MAX_DPI} 범위여야 합니다.")
                args = ("dpi", dpi, dpi)
                suffix = f"{dpi}dpi"
                desc = f"DPI: {dpi}"
            else:
                x, y = self.valid_um()
                args = ("aoi", x, y)
                suffix = f"{x:g}x{y:g}umpp".replace(".", "p")
                desc = f"AOI 해상도: X={x:g}, Y={y:g} µm/pixel"
        except Exception as exc:
            messagebox.showerror("출력 설정 오류", str(exc) or "출력 설정값을 확인하세요.")
            return
        target_name = "composite" if specs is not None else layer.name
        output = filedialog.asksaveasfilename(title="CAM Image 저장", defaultextension=".png", initialfile=f"{self.info.name}_{step}_{target_name}_{suffix}.png", filetypes=[("PNG Image", "*.png")])
        if not output:
            return
        output_path = Path(output)
        token = self.token
        self.render_btn.configure(state="disabled")
        self.status.set(f"렌더링 중: {step}/{target_name} | {desc}")
        job = self.job
        layer_name = layer.name if layer else None
        def work() -> None:
            try:
                renderer = ODBRenderer(job, args[1]) if args[0] == "dpi" else ODBRenderer.from_um_per_pixel(job, args[1], args[2])
                image = renderer.render_composite(step, specs) if specs is not None else renderer.render(step, layer_name)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                image.save(output_path, optimize=True, dpi=(renderer.dpi_x, renderer.dpi_y))
                composite_desc = ""
                if specs is not None:
                    composite_desc = "\nComposite: " + ", ".join(f"{s.layer}:{s.operation}:GV{s.gv}" for s in specs)
                self.q.put(("rendered", image, output_path, renderer.stats, desc + composite_desc))
            except Exception as exc:
                self.q.put(("error", token, f"렌더링 실패: {exc}"))
        threading.Thread(target=work, daemon=True).start()

    def close(self) -> None:
        if self.tmp:
            self.tmp.cleanup()
        self.destroy()


if __name__ == "__main__":
    App().mainloop()
