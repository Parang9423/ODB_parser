#!/usr/bin/env python3
from __future__ import annotations

import threading
import tkinter as tk
from tkinter import messagebox, ttk

import app_core as core
from app_core import App as _CoreApp
from hierarchy_renderer import FastODBRenderer, adaptive_preview_dpi

# Keep the existing app behavior, but use the cached renderer for normal/final renders too.
core.ODBRenderer = FastODBRenderer

HIER_COLORS = {
    "pnl": "#3b82f6",
    "strip": "#22c55e",
    "unit": "#f59e0b",
}
MAX_PREVIEW_PIXELS = 12_000_000


class App(_CoreApp):
    """ODB CAM app with Composite UX fixes and PNL-coordinate hierarchy overlay."""

    def _ui(self) -> None:
        super()._ui()

        # Composite: separate Add and Edit so selecting an existing row never blocks adding.
        add_button = None
        for child in self.comp_frame.winfo_children():
            try:
                if child.cget("text") == "추가/수정":
                    add_button = child
                    break
            except tk.TclError:
                continue
        if add_button is None:
            raise RuntimeError("Composite 추가/수정 버튼을 찾을 수 없습니다.")
        add_button.configure(text="+ 레이어 추가", command=self.add_composite_row)
        add_button.grid_configure(sticky="ew")
        self.comp_tree.grid_configure(columnspan=5)
        ttk.Button(
            self.comp_frame,
            text="선택 레이어 수정",
            command=self.update_composite_row,
        ).grid(row=2, column=4, padx=(4, 0), sticky="ew")

        # Hierarchy controls are placed above the existing preview canvas.
        self.hierarchy_enabled = tk.BooleanVar(value=False)
        self.hier_pnl = tk.BooleanVar(value=True)
        self.hier_strip = tk.BooleanVar(value=True)
        self.hier_unit = tk.BooleanVar(value=True)
        self.pnl_coord_text = tk.StringVar(value="PNL X: - mm   Y: - mm")
        self._view_bounds = None
        self._view_dpi_x = None
        self._view_dpi_y = None
        self._hier_profiles = []
        self._hier_root = None

        canvas_frame = self.canvas.master
        right = canvas_frame.master
        hierarchy_bar = ttk.Frame(right)
        hierarchy_bar.pack(fill="x", pady=(0, 6), before=canvas_frame)
        ttk.Checkbutton(
            hierarchy_bar,
            text="Hierarchy Overlay",
            variable=self.hierarchy_enabled,
            command=self._hierarchy_changed,
        ).pack(side="left")
        ttk.Separator(hierarchy_bar, orient="vertical").pack(side="left", fill="y", padx=8)
        self.pnl_check = ttk.Checkbutton(
            hierarchy_bar,
            text="PNL",
            variable=self.hier_pnl,
            command=self._hierarchy_changed,
        )
        self.pnl_check.pack(side="left")
        self.strip_check = ttk.Checkbutton(
            hierarchy_bar,
            text="STRIP",
            variable=self.hier_strip,
            command=self._hierarchy_changed,
        )
        self.strip_check.pack(side="left", padx=(6, 0))
        self.unit_check = ttk.Checkbutton(
            hierarchy_bar,
            text="UNIT",
            variable=self.hier_unit,
            command=self._hierarchy_changed,
        )
        self.unit_check.pack(side="left", padx=(6, 0))

        ttk.Label(hierarchy_bar, text="PNL", foreground=HIER_COLORS["pnl"]).pack(side="left", padx=(14, 2))
        ttk.Label(hierarchy_bar, text="STRIP", foreground=HIER_COLORS["strip"]).pack(side="left", padx=2)
        ttk.Label(hierarchy_bar, text="UNIT", foreground=HIER_COLORS["unit"]).pack(side="left", padx=2)
        ttk.Label(hierarchy_bar, textvariable=self.pnl_coord_text).pack(side="right")

        self.canvas.bind("<Motion>", self._on_canvas_motion, add="+")
        self.canvas.bind("<Leave>", lambda _e: self.pnl_coord_text.set("PNL X: - mm   Y: - mm"), add="+")

    def add_composite_row(self) -> None:
        selected = self.comp_tree.selection()
        if selected:
            self.comp_tree.selection_remove(*selected)
        _CoreApp.upsert_composite_row(self)

    def update_composite_row(self) -> None:
        if not self.comp_tree.selection():
            messagebox.showwarning("Composite", "수정할 Composite Layer를 목록에서 선택하세요.")
            return
        _CoreApp.upsert_composite_row(self)

    def apply_loaded(self, path, job, tmp, info) -> None:
        super().apply_loaded(path, job, tmp, info)
        available = {name.lower() for name in info.steps}
        self.pnl_check.configure(state="normal" if "pnl" in available else "disabled")
        self.strip_check.configure(state="normal" if "strip" in available else "disabled")
        self.unit_check.configure(state="normal" if "unit" in available else "disabled")
        if "pnl" not in available:
            self.hierarchy_enabled.set(False)
        self._hierarchy_changed(render=False)

    def _hierarchy_changed(self, render: bool = True) -> None:
        enabled = bool(self.hierarchy_enabled.get()) and self.info is not None
        self.step_cb.configure(state="disabled" if enabled else ("readonly" if self.info else "disabled"))
        if enabled and not self._visible_hierarchy_steps():
            self.hier_pnl.set(True)
        self.update_info()
        if render and self.job:
            self.schedule_preview()
        else:
            self.draw()

    def _visible_hierarchy_steps(self) -> set[str]:
        if not self.info:
            return set()
        available = {name.lower() for name in self.info.steps}
        visible = set()
        if self.hier_pnl.get() and "pnl" in available:
            visible.add("pnl")
        if self.hier_strip.get() and "strip" in available:
            visible.add("strip")
        if self.hier_unit.get() and "unit" in available:
            visible.add("unit")
        return visible

    def _root_step(self) -> str:
        if self.hierarchy_enabled.get() and self.info:
            available = {name.lower() for name in self.info.steps}
            if "pnl" in available:
                return "pnl"
        return self.step.get()

    def schedule_preview(self) -> None:
        if not self.job:
            return
        root_step = self._root_step()
        if not root_step:
            return
        hierarchy = bool(self.hierarchy_enabled.get()) and root_step == "pnl"
        visible_steps = self._visible_hierarchy_steps() if hierarchy else set()
        if hierarchy and not visible_steps:
            return
        if self.target_mode.get() == core.TARGET_COMPOSITE and not self.comp_tree.get_children():
            return

        self.token += 1
        token = self.token
        self.render_btn.configure(state="disabled")
        job = self.job
        requested_dpi = float(self.preview_dpi)
        specs = self.composite_specs() if self.target_mode.get() == core.TARGET_COMPOSITE else None
        layer = self.selected_layer()
        layer_name = layer.name if layer else None
        target = "Composite" if specs is not None else (layer_name or "-")

        # Profile parsing is cached; using it here lets us avoid huge PNL preview rasters.
        probe = FastODBRenderer(job, requested_dpi)
        bounds = probe.profile_bounds(root_step)
        effective_dpi = adaptive_preview_dpi(bounds, requested_dpi, MAX_PREVIEW_PIXELS)
        if effective_dpi < requested_dpi - 0.5:
            self.status.set(
                f"미리보기 생성 중: {root_step}/{target} | 요청 {requested_dpi:g} DPI → 자동 {effective_dpi:.0f} DPI"
            )
        else:
            self.status.set(f"미리보기 생성 중: {root_step}/{target} @ {effective_dpi:g} DPI")

        def work() -> None:
            try:
                renderer = FastODBRenderer(job, effective_dpi)
                if hierarchy:
                    if specs is not None:
                        image = renderer.render_composite_hierarchy(root_step, specs, visible_steps)
                    else:
                        image = renderer.render_hierarchy(root_step, layer_name, visible_steps)
                    profiles = []
                    for instance in renderer.collect_instances(root_step):
                        if instance.step in visible_steps:
                            profiles.append((instance.step, instance.depth, renderer.transformed_profile(instance)))
                else:
                    image = renderer.render_composite(root_step, specs) if specs is not None else renderer.render(root_step, layer_name)
                    profiles = []
                if token == self.token:
                    self._view_bounds = bounds
                    self._view_dpi_x = renderer.dpi_x
                    self._view_dpi_y = renderer.dpi_y
                    self._hier_profiles = profiles
                    self._hier_root = root_step if hierarchy else None
                self.q.put(("preview", token, image, renderer.stats, effective_dpi))
            except Exception as exc:
                self.q.put(("error", token, f"미리보기 생성 실패: {exc}"))

        threading.Thread(target=work, daemon=True).start()

    def draw(self) -> None:
        super().draw()
        if self.hierarchy_enabled.get() and self.preview and self._hier_profiles and self._view_bounds:
            self._draw_hierarchy_profiles()

    def _image_canvas_origin(self) -> tuple[float, float, float, float]:
        width = self.preview.width * self.zoom
        height = self.preview.height * self.zoom
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        left = cw / 2 + self.offx - width / 2
        top = ch / 2 + self.offy - height / 2
        return left, top, width, height

    def _root_point_to_canvas(self, x_in: float, y_in: float) -> tuple[float, float]:
        xmin, ymin, xmax, ymax = self._view_bounds
        left, top, _width, _height = self._image_canvas_origin()
        px = (x_in - xmin) * self._view_dpi_x
        py = (ymax - y_in) * self._view_dpi_y
        return left + px * self.zoom, top + py * self.zoom

    def _draw_hierarchy_profiles(self) -> None:
        self.canvas.delete("hier_profile")
        depth_colors = [HIER_COLORS["pnl"], HIER_COLORS["strip"], HIER_COLORS["unit"], "#a855f7"]
        for step, depth, contours in self._hier_profiles:
            color = HIER_COLORS.get(step, depth_colors[min(depth, len(depth_colors) - 1)])
            width = 2 if depth == 0 else 1
            for _kind, points in contours:
                if len(points) < 2:
                    continue
                xy = []
                for x_in, y_in in points:
                    cx, cy = self._root_point_to_canvas(x_in, y_in)
                    xy.extend((cx, cy))
                if points[0] != points[-1]:
                    cx, cy = self._root_point_to_canvas(*points[0])
                    xy.extend((cx, cy))
                self.canvas.create_line(*xy, fill=color, width=width, tags=("hier_profile",))
        self.canvas.tag_raise("hier_profile")

    def _on_canvas_motion(self, event) -> None:
        if not self.hierarchy_enabled.get() or not self.preview or not self._view_bounds:
            self.pnl_coord_text.set("PNL X: - mm   Y: - mm")
            return
        left, top, width, height = self._image_canvas_origin()
        if event.x < left or event.x > left + width or event.y < top or event.y > top + height:
            self.pnl_coord_text.set("PNL X: - mm   Y: - mm")
            return
        image_x = (event.x - left) / self.zoom
        image_y = (event.y - top) / self.zoom
        xmin, _ymin, _xmax, ymax = self._view_bounds
        x_in = xmin + image_x / self._view_dpi_x
        y_in = ymax - image_y / self._view_dpi_y
        self.pnl_coord_text.set(f"PNL X: {x_in * 25.4:.3f} mm   Y: {y_in * 25.4:.3f} mm")

    def update_info(self) -> None:
        super().update_info()
        if not getattr(self, "info_txt", None) or not self.job or not self.info:
            return
        if not getattr(self, "hierarchy_enabled", None) or not self.hierarchy_enabled.get():
            return
        visible = ", ".join(name.upper() for name in ("pnl", "strip", "unit") if name in self._visible_hierarchy_steps())
        self.info_txt.configure(state="normal")
        self.info_txt.insert("end", f"\n\nHierarchy root: PNL\nVisible Steps: {visible or '-'}\nBorder: PNL / STRIP / UNIT 색상 구분")
        self.info_txt.configure(state="disabled")


if __name__ == "__main__":
    App().mainloop()
