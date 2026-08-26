#!/usr/bin/env python3
"""Main Tk window for ODB CAM inspection/alignment workflows.

This module owns UI orchestration only. ERT parsing, guide-drill extraction and
rendering are imported through their package boundaries so future batch/ROI
workflows can reuse them without importing Tk.
"""
from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Dict, Optional

import app_core as core
from alignment import collect_guide_drill_candidates, drill_layer_names
from aoi import ERTMetadata, parse_ert
from app_core import App as _CoreApp
from odb_cam_renderer import CompositeLayer
from render import FastODBRenderer, adaptive_preview_dpi, render_selected_steps_composite

from .defaults import default_spec
from .layer_dialog import LayerDialog
from .platform import configure_linux_fonts

core.ODBRenderer = FastODBRenderer

HIER_COLORS = {"pnl": "#3b82f6", "strip": "#22c55e", "unit": "#f59e0b"}
ALIGN_COLORS = ("#ff3b30", "#00d084", "#ffd60a")
MAX_PREVIEW_PIXELS = 12_000_000


class App(_CoreApp):
    """PNL-root CAM viewer with lazy layers, ERT metadata and guide alignment."""

    def _ui(self) -> None:
        configure_linux_fonts(self)
        super()._ui()
        configure_linux_fonts(self)

        self.selected_specs: Dict[str, CompositeLayer] = {}
        self.show_pnl = tk.BooleanVar(value=True)
        self.show_strip = tk.BooleanVar(value=True)
        self.show_unit = tk.BooleanVar(value=True)
        self.edit_op = tk.StringVar(value="REPLACE")
        self.edit_gv = tk.IntVar(value=255)
        self.pnl_coord_text = tk.StringVar(value="PNL X: - mm   Y: - mm")

        self._view_bounds = None
        self._view_dpi_x = None
        self._view_dpi_y = None
        self._hier_profiles = []

        self.ert_metadata: Optional[ERTMetadata] = None
        self.ert_path: Optional[Path] = None
        self.guide_candidates = []
        self.alignment_points = []
        self.guide_pick_mode = False

        self._configure_step_controls()
        self._configure_layer_panel()
        self._configure_top_actions()
        self._configure_canvas_events()
        self.render_btn.configure(text="Composite 결과 저장")

    def _configure_step_controls(self) -> None:
        left = self.step_cb.master
        for widget in (self.step_cb, self.target_cb, self.layer_cb, self.comp_frame):
            widget.grid_remove()
        for child in left.winfo_children():
            if isinstance(child, ttk.Label):
                try:
                    if child.cget("text") in {"Step", "렌더링 대상", "Layer"}:
                        child.grid_remove()
                except tk.TclError:
                    pass

        ttk.Label(left, text="Step 표시", font=("TkDefaultFont", 11, "bold")).grid(
            row=1, column=0, columnspan=2, sticky="w"
        )
        steps = ttk.Frame(left)
        steps.grid(row=2, column=0, columnspan=2, sticky="w", pady=(5, 8))
        self.pnl_check = ttk.Checkbutton(
            steps, text="PNL", variable=self.show_pnl, command=self.step_changed
        )
        self.pnl_check.pack(side="left")
        self.strip_check = ttk.Checkbutton(
            steps, text="STRIP", variable=self.show_strip, command=self.step_changed
        )
        self.strip_check.pack(side="left", padx=8)
        self.unit_check = ttk.Checkbutton(
            steps, text="UNIT", variable=self.show_unit, command=self.step_changed
        )
        self.unit_check.pack(side="left")

    def _configure_layer_panel(self) -> None:
        left = self.step_cb.master
        box = ttk.LabelFrame(left, text="선택 레이어", padding=6)
        box.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        self.layer_tree = ttk.Treeview(
            box,
            columns=("name", "type", "side", "op", "gv"),
            show="headings",
            height=6,
            selectmode="browse",
        )
        columns = [
            ("name", "Layer", 105),
            ("type", "Type", 85),
            ("side", "Side", 55),
            ("op", "Operation", 75),
            ("gv", "GV", 40),
        ]
        for column, title, width in columns:
            self.layer_tree.heading(column, text=title)
            self.layer_tree.column(
                column,
                width=width,
                anchor="center" if column in {"side", "op", "gv"} else "w",
            )
        self.layer_tree.pack(fill="x")
        self.layer_tree.bind("<<TreeviewSelect>>", lambda _event: self.load_edit())

        edit = ttk.Frame(box)
        edit.pack(fill="x", pady=(5, 0))
        ttk.Combobox(
            edit,
            textvariable=self.edit_op,
            values=["ADD", "REPLACE", "SUBTRACT"],
            state="readonly",
            width=10,
        ).pack(side="left")
        ttk.Spinbox(edit, from_=0, to=255, textvariable=self.edit_gv, width=6).pack(
            side="left", padx=4
        )
        ttk.Button(edit, text="설정 적용", command=self.apply_edit).pack(side="left")

    def _configure_top_actions(self) -> None:
        top = self.file_label.master
        self.layer_pick_btn = ttk.Button(
            top, text="레이어 선택/변경", command=self.choose_layers, state="disabled"
        )
        self.layer_pick_btn.pack(side="left", padx=(0, 8), before=self.file_label)
        self.ert_load_btn = ttk.Button(
            top, text="ERT 불러오기", command=self.load_ert, state="disabled"
        )
        self.ert_load_btn.pack(side="left", padx=(0, 6), before=self.file_label)
        self.guide_pick_btn = ttk.Button(
            top,
            text="가이드홀 3점 선택",
            command=self.toggle_guide_pick,
            state="disabled",
        )
        self.guide_pick_btn.pack(side="left", padx=(0, 6), before=self.file_label)
        self.guide_clear_btn = ttk.Button(
            top, text="Align 초기화", command=self.clear_alignment, state="disabled"
        )
        self.guide_clear_btn.pack(side="left", padx=(0, 8), before=self.file_label)
        ttk.Label(top, textvariable=self.pnl_coord_text).pack(side="right")

    def _configure_canvas_events(self) -> None:
        self.canvas.bind("<Motion>", self.on_motion, add="+")
        self.canvas.bind(
            "<Leave>",
            lambda _event: self.pnl_coord_text.set("PNL X: - mm   Y: - mm"),
            add="+",
        )
        self.canvas.bind("<Button-1>", self.on_alignment_click, add="+")

    def apply_loaded(self, path, job, tmp, info) -> None:
        if self.tmp:
            self.tmp.cleanup()
        self.job, self.tmp, self.info, self.source = job, tmp, info, path
        self.selected_specs.clear()
        self.preview = None
        self._hier_profiles = []
        self.ert_metadata = None
        self.ert_path = None
        self.guide_candidates = []
        self.alignment_points = []
        self.guide_pick_mode = False

        self.file_label.configure(text=path.name)
        for button in (
            self.layer_pick_btn,
            self.ert_load_btn,
            self.guide_pick_btn,
            self.guide_clear_btn,
        ):
            button.configure(state="normal")

        available = {step.lower() for step in info.steps}
        step_controls = [
            ("pnl", self.show_pnl, self.pnl_check),
            ("strip", self.show_strip, self.strip_check),
            ("unit", self.show_unit, self.unit_check),
        ]
        for name, variable, widget in step_controls:
            exists = name in available
            variable.set(exists)
            widget.configure(state="normal" if exists else "disabled")

        self.render_btn.configure(state="disabled")
        self.refresh_layer_tree()
        self.update_info()
        self.status.set(
            f"메타데이터 로드 완료: {info.name} / {len(info.layers)} layers. 필요한 Layer를 선택하세요."
        )
        self.after(120, self.choose_layers)

    def choose_layers(self) -> None:
        if not self.info:
            return
        dialog = LayerDialog(self, self.info.layers, set(self.selected_specs))
        self.wait_window(dialog)
        if dialog.result is None:
            return

        old = self.selected_specs
        metadata = {layer.name: layer for layer in self.info.layers}
        self.selected_specs = {
            name: old.get(name, default_spec(metadata[name]))
            for name in dialog.result
        }
        self.refresh_layer_tree()
        self.update_info()
        self.render_btn.configure(state="normal")
        self.schedule_preview()

    def refresh_layer_tree(self) -> None:
        for item in self.layer_tree.get_children():
            self.layer_tree.delete(item)
        if not self.info:
            return
        metadata = {layer.name: layer for layer in self.info.layers}
        for name, spec in self.selected_specs.items():
            layer = metadata[name]
            self.layer_tree.insert(
                "",
                "end",
                iid=name,
                values=(name, layer.layer_type, layer.side, spec.operation, spec.gv),
            )

    def load_edit(self) -> None:
        selected = self.layer_tree.selection()
        if not selected:
            return
        spec = self.selected_specs[selected[0]]
        self.edit_op.set(spec.operation)
        self.edit_gv.set(spec.gv)

    def apply_edit(self) -> None:
        selected = self.layer_tree.selection()
        if not selected:
            messagebox.showwarning("레이어 설정", "수정할 Layer를 선택하세요.")
            return
        try:
            gv = int(self.edit_gv.get())
            if not 0 <= gv <= 255:
                raise ValueError
            spec = CompositeLayer(
                selected[0], self.edit_op.get().upper(), gv
            ).normalized()
        except Exception:
            messagebox.showerror("GV 오류", "GV는 0~255 사이의 정수여야 합니다.")
            return
        self.selected_specs[selected[0]] = spec
        self.refresh_layer_tree()
        self.layer_tree.selection_set(selected[0])
        self.schedule_preview()

    def visible_steps(self) -> set[str]:
        if not self.info:
            return set()
        available = {step.lower() for step in self.info.steps}
        visible: set[str] = set()
        if self.show_pnl.get() and "pnl" in available:
            visible.add("pnl")
        if self.show_strip.get() and "strip" in available:
            visible.add("strip")
        if self.show_unit.get() and "unit" in available:
            visible.add("unit")
        return visible

    def root_step(self) -> str:
        available = {step.lower() for step in self.info.steps} if self.info else set()
        return next(
            (step for step in ("pnl", "strip", "unit") if step in available),
            next(iter(available), ""),
        )

    def step_changed(self) -> None:
        if not self.visible_steps() and self.info:
            available = {step.lower() for step in self.info.steps}
            for name, variable in (
                ("pnl", self.show_pnl),
                ("strip", self.show_strip),
                ("unit", self.show_unit),
            ):
                if name in available:
                    variable.set(True)
                    break
        self.update_info()
        if self.selected_specs:
            self.schedule_preview()

    def load_ert(self) -> None:
        if not self.job:
            return
        path = filedialog.askopenfilename(
            title="ERT 파일 선택",
            filetypes=[("ERT", "*.ERT *.ert"), ("모든 파일", "*.*")],
        )
        if not path:
            return
        try:
            self.ert_metadata = parse_ert(path)
            self.ert_path = Path(path)
        except Exception as exc:
            messagebox.showerror("ERT 로드 실패", str(exc))
            return
        self.update_info()
        self.status.set(
            "ERT 로드 완료 | "
            f"해상도 {self.ert_metadata.resolution_um_per_px:g} µm/px | "
            f"Guide 후보 {self.ert_metadata.guide_reference_candidate}"
        )

    def _load_guide_candidates(self):
        if not self.job or not self.info:
            return []
        layers = drill_layer_names(self.info.layers)
        if not layers:
            return []
        renderer = FastODBRenderer(self.job, max(72.0, float(self.preview_dpi)))
        rows = collect_guide_drill_candidates(
            renderer, self.root_step(), layers, {"pnl"}
        )
        if not rows:
            rows = collect_guide_drill_candidates(
                renderer, self.root_step(), layers, self.visible_steps()
            )
        rows.sort(key=lambda row: (-row.diameter_in, row.x_in, row.y_in))
        return rows

    def toggle_guide_pick(self) -> None:
        if not self.job:
            return
        if self.guide_pick_mode:
            self.guide_pick_mode = False
            self.guide_pick_btn.configure(text="가이드홀 3점 선택")
            self.draw()
            return
        if len(self.alignment_points) >= 3:
            self.alignment_points = []
        if not self.guide_candidates:
            self.status.set("Guide Drill 후보를 읽는 중...")
            self.update_idletasks()
            self.guide_candidates = self._load_guide_candidates()
        if not self.guide_candidates:
            messagebox.showwarning(
                "Guide Drill",
                "PNL에서 원형 Drill/Pad 후보를 찾지 못했습니다. Drill 계열 Layer가 있는지 확인하세요.",
            )
            return
        self.guide_pick_mode = True
        self.guide_pick_btn.configure(
            text=f"P{len(self.alignment_points) + 1} 선택 중..."
        )
        self.status.set(
            f"Guide Drill 후보 {len(self.guide_candidates)}개 | 확대 후 홀을 클릭하세요."
        )
        self.draw()

    def clear_alignment(self) -> None:
        self.alignment_points = []
        self.guide_pick_mode = False
        self.guide_pick_btn.configure(text="가이드홀 3점 선택")
        self.update_info()
        self.draw()

    def _nearest_guide_on_screen(self, x: float, y: float, max_px: float = 28):
        if not self.preview or not self._view_bounds:
            return None
        best = None
        best_distance = max_px * max_px
        for row in self.guide_candidates:
            center_x, center_y = self.root_to_canvas(row.x_in, row.y_in)
            distance = (center_x - x) ** 2 + (center_y - y) ** 2
            if distance <= best_distance:
                best = row
                best_distance = distance
        return best

    def on_alignment_click(self, event):
        if not self.guide_pick_mode:
            return None
        row = self._nearest_guide_on_screen(event.x, event.y)
        if row is None:
            self.status.set(
                "근처 Guide Drill 심볼이 없습니다. 더 확대해서 심볼 중심 근처를 클릭하세요."
            )
            return "break"
        if any(
            abs(point.x_in - row.x_in) < 1e-9
            and abs(point.y_in - row.y_in) < 1e-9
            for point in self.alignment_points
        ):
            self.status.set("이미 선택한 Guide Drill입니다.")
            return "break"

        self.alignment_points.append(row)
        if len(self.alignment_points) >= 3:
            self.guide_pick_mode = False
            self.guide_pick_btn.configure(text="가이드홀 3점 다시 선택")
            self.status.set(
                "Guide Drill 3점 선택 완료. ERT 기준점 후보와 ODB 좌표를 비교할 수 있습니다."
            )
        else:
            self.guide_pick_btn.configure(
                text=f"P{len(self.alignment_points) + 1} 선택 중..."
            )
            self.status.set(
                f"P{len(self.alignment_points)} 선택: "
                f"({row.x_mm:.3f}, {row.y_mm:.3f}) mm"
            )
        self.update_info()
        self.draw()
        return "break"

    def schedule_preview(self) -> None:
        if not self.job or not self.selected_specs:
            return
        root = self.root_step()
        visible = self.visible_steps()
        if not root or not visible:
            return

        self.token += 1
        token = self.token
        self.render_btn.configure(state="disabled")
        requested = float(self.preview_dpi)
        probe = FastODBRenderer(self.job, requested)
        bounds = probe.profile_bounds(root)
        effective = adaptive_preview_dpi(
            bounds, requested, MAX_PREVIEW_PIXELS
        )
        specs = list(self.selected_specs.values())
        job = self.job
        suffix = f" (요청 {requested:g})" if effective < requested - 0.5 else ""
        self.status.set(
            f"미리보기 생성 중 | {len(specs)} layers | {effective:.0f} DPI{suffix}"
        )

        def work() -> None:
            try:
                renderer = FastODBRenderer(job, effective)
                image = render_selected_steps_composite(
                    renderer, root, specs, visible
                )
                profiles = [
                    (instance.step, instance.depth, renderer.transformed_profile(instance))
                    for instance in renderer.collect_instances(root)
                    if instance.step in visible
                ]
                if token == self.token:
                    self._view_bounds = bounds
                    self._view_dpi_x = renderer.dpi_x
                    self._view_dpi_y = renderer.dpi_y
                    self._hier_profiles = profiles
                self.q.put(("preview", token, image, renderer.stats, effective))
            except Exception as exc:
                self.q.put(("error", token, f"미리보기 생성 실패: {exc}"))

        threading.Thread(target=work, daemon=True).start()

    def render(self) -> None:
        if not self.job or not self.selected_specs:
            return
        root = self.root_step()
        visible = self.visible_steps()
        specs = list(self.selected_specs.values())
        try:
            if self.output_mode.get() == core.MODE_DPI:
                dpi = int(self.dpi.get())
                if not core.MIN_DPI <= dpi <= core.MAX_DPI:
                    raise ValueError
                args = ("dpi", dpi, dpi)
                suffix = f"{dpi}dpi"
                description = f"DPI: {dpi}"
            else:
                x_um, y_um = self.valid_um()
                args = ("aoi", x_um, y_um)
                suffix = f"{x_um:g}x{y_um:g}umpp".replace(".", "p")
                description = f"AOI X={x_um:g}, Y={y_um:g} µm/pixel"
        except Exception:
            messagebox.showerror(
                "출력 설정 오류", "출력 DPI/AOI 해상도를 확인하세요."
            )
            return

        step_suffix = "-".join(
            step.upper()
            for step in ("pnl", "strip", "unit")
            if step in visible
        )
        output = filedialog.asksaveasfilename(
            title="CAM Image 저장",
            defaultextension=".png",
            initialfile=(
                f"{self.info.name}_{step_suffix}_composite_{suffix}.png"
            ),
            filetypes=[("PNG Image", "*.png")],
        )
        if not output:
            return

        output_path = Path(output)
        job = self.job
        token = self.token
        self.render_btn.configure(state="disabled")
        self.status.set(
            f"최종 Composite CAM 렌더링 중... Steps={step_suffix}"
        )

        def work() -> None:
            try:
                if args[0] == "dpi":
                    renderer = FastODBRenderer(job, args[1])
                else:
                    renderer = FastODBRenderer.from_um_per_pixel(
                        job, args[1], args[2]
                    )
                image = render_selected_steps_composite(
                    renderer, root, specs, visible
                )
                output_path.parent.mkdir(parents=True, exist_ok=True)
                image.save(
                    output_path,
                    format="PNG",
                    compress_level=1,
                    optimize=False,
                    dpi=(renderer.dpi_x, renderer.dpi_y),
                )
                self.q.put(
                    (
                        "rendered",
                        image,
                        output_path,
                        renderer.stats,
                        description
                        + f" | Steps: {step_suffix} | PNG fast/lossless",
                    )
                )
            except Exception as exc:
                self.q.put(("error", token, f"렌더링 실패: {exc}"))

        threading.Thread(target=work, daemon=True).start()

    def draw(self) -> None:
        super().draw()
        if self.preview and self._hier_profiles and self._view_bounds:
            self.draw_profiles()
        if self.preview and self._view_bounds:
            self.draw_alignment_overlay()

    def image_origin(self):
        width = self.preview.width * self.zoom
        height = self.preview.height * self.zoom
        canvas_width = max(1, self.canvas.winfo_width())
        canvas_height = max(1, self.canvas.winfo_height())
        return (
            canvas_width / 2 + self.offx - width / 2,
            canvas_height / 2 + self.offy - height / 2,
            width,
            height,
        )

    def root_to_canvas(self, x: float, y: float):
        xmin, _ymin, _xmax, ymax = self._view_bounds
        left, top, _width, _height = self.image_origin()
        return (
            left + (x - xmin) * self._view_dpi_x * self.zoom,
            top + (ymax - y) * self._view_dpi_y * self.zoom,
        )

    def draw_profiles(self) -> None:
        self.canvas.delete("hier_profile")
        for step, depth, contours in self._hier_profiles:
            color = HIER_COLORS.get(step, "#a855f7")
            width = 2 if depth == 0 else 1
            for _kind, points in contours:
                if len(points) < 2:
                    continue
                coordinates = []
                for x, y in points:
                    coordinates.extend(self.root_to_canvas(x, y))
                if points[0] != points[-1]:
                    coordinates.extend(self.root_to_canvas(*points[0]))
                self.canvas.create_line(
                    *coordinates,
                    fill=color,
                    width=width,
                    tags="hier_profile",
                )
        self.canvas.tag_raise("hier_profile")

    def draw_alignment_overlay(self) -> None:
        self.canvas.delete("guide_candidate")
        self.canvas.delete("align_point")
        if self.guide_pick_mode:
            left, top, width, height = self.image_origin()
            shown = 0
            for row in self.guide_candidates:
                center_x, center_y = self.root_to_canvas(row.x_in, row.y_in)
                if not (
                    left - 8 <= center_x <= left + width + 8
                    and top - 8 <= center_y <= top + height + 8
                ):
                    continue
                radius = max(
                    2,
                    min(
                        7,
                        row.diameter_in
                        * max(self._view_dpi_x, self._view_dpi_y)
                        * self.zoom
                        / 2,
                    ),
                )
                self.canvas.create_oval(
                    center_x - radius,
                    center_y - radius,
                    center_x + radius,
                    center_y + radius,
                    outline="#00e5ff",
                    width=1,
                    tags="guide_candidate",
                )
                shown += 1
                if shown >= 2500:
                    break

        for index, row in enumerate(self.alignment_points):
            center_x, center_y = self.root_to_canvas(row.x_in, row.y_in)
            color = ALIGN_COLORS[index % len(ALIGN_COLORS)]
            arm = 10
            self.canvas.create_line(
                center_x - arm,
                center_y,
                center_x + arm,
                center_y,
                fill=color,
                width=2,
                tags="align_point",
            )
            self.canvas.create_line(
                center_x,
                center_y - arm,
                center_x,
                center_y + arm,
                fill=color,
                width=2,
                tags="align_point",
            )
            self.canvas.create_text(
                center_x + 14,
                center_y - 12,
                text=f"P{index + 1}",
                fill=color,
                anchor="w",
                tags="align_point",
            )
        self.canvas.tag_raise("guide_candidate")
        self.canvas.tag_raise("align_point")

    def on_motion(self, event) -> None:
        if not self.preview or not self._view_bounds:
            self.pnl_coord_text.set("PNL X: - mm   Y: - mm")
            return
        left, top, width, height = self.image_origin()
        if not (
            left <= event.x <= left + width
            and top <= event.y <= top + height
        ):
            self.pnl_coord_text.set("PNL X: - mm   Y: - mm")
            return
        image_x = (event.x - left) / self.zoom
        image_y = (event.y - top) / self.zoom
        xmin, _ymin, _xmax, ymax = self._view_bounds
        x = xmin + image_x / self._view_dpi_x
        y = ymax - image_y / self._view_dpi_y
        self.pnl_coord_text.set(
            f"PNL X: {x * 25.4:.3f} mm   Y: {y * 25.4:.3f} mm"
        )

    def update_info(self) -> None:
        if not getattr(self, "info_txt", None) or not self.info:
            return
        visible = ", ".join(
            step.upper()
            for step in ("pnl", "strip", "unit")
            if step in self.visible_steps()
        )
        type_counts: Dict[str, int] = {}
        for layer in self.info.layers:
            if layer.name in self.selected_specs:
                type_counts[layer.layer_type] = type_counts.get(layer.layer_type, 0) + 1

        lines = [
            f"Job: {self.info.name}",
            f"Source: {self.source.name if self.source else '-'}",
            f"Steps: {', '.join(step.upper() for step in self.info.steps)}",
            f"Matrix layers: {len(self.info.layers)}",
            f"Selected layers: {len(self.selected_specs)}",
            f"Visible steps: {visible or '-'}",
            "",
            "Selected types: "
            + (
                ", ".join(f"{kind}={count}" for kind, count in type_counts.items())
                or "-"
            ),
            "",
        ]

        if self.ert_metadata:
            metadata = self.ert_metadata
            roi_width, roi_height = metadata.roi_size_mm_for_pixels(100, 100)
            lines.extend(
                [
                    f"ERT: {self.ert_path.name if self.ert_path else '-'}",
                    f"ERT resolution: {metadata.resolution_um_per_px:g} µm/pixel",
                    f"ERT region values: {metadata.region_values}",
                    "ERT guide reference candidate: "
                    f"{metadata.guide_reference_candidate}",
                    f"100×100 physical ROI: {roi_width:.3f} × {roi_height:.3f} mm",
                    "",
                ]
            )

        if self.alignment_points:
            lines.append("ODB Guide Drill points (PNL root):")
            for index, row in enumerate(self.alignment_points, start=1):
                lines.append(
                    f" P{index}: X={row.x_mm:.3f} mm, Y={row.y_mm:.3f} mm, "
                    f"D={row.diameter_mm:.3f} mm, {row.layer}/{row.step}"
                )
            lines.append("")

        lines.extend(
            [
                "Lazy loading:",
                "Matrix/Step metadata → Layer 선택 → 선택 feature만 로드/캐시",
                "Guide Drill 후보는 Align 모드를 켤 때만 Drill feature를 읽음",
            ]
        )
        self.info_txt.configure(state="normal")
        self.info_txt.delete("1.0", "end")
        self.info_txt.insert("1.0", "\n".join(lines))
        self.info_txt.configure(state="disabled")
