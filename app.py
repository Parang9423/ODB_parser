#!/usr/bin/env python3
from __future__ import annotations

import queue
import threading
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import List, Optional

from PIL import Image, ImageOps, ImageTk

from odb_cam_renderer import (
    ODBError,
    ODBRenderer,
    contours_bounds,
    extract_input,
    parse_kv_blocks,
    parse_profile_contours,
)

PREVIEW_DPI = 600
DEFAULT_RENDER_DPI = 1200
MIN_ZOOM = 0.05
MAX_ZOOM = 16.0
ZOOM_STEP = 1.25


@dataclass(frozen=True)
class LayerInfo:
    name: str
    layer_type: str = "?"
    context: str = "?"
    side: str = "?"
    polarity: str = "?"

    @property
    def label(self) -> str:
        details = [self.layer_type]
        if self.side not in {"", "?"}:
            details.append(self.side)
        return f"{self.name}  [{' / '.join(details)}]"


@dataclass
class JobInfo:
    name: str
    steps: List[str]
    layers: List[LayerInfo]


def inspect_job(job_dir: Path) -> JobInfo:
    steps_dir = job_dir / "steps"
    if not steps_dir.is_dir():
        raise ODBError("ODB++ steps directory is missing")
    steps = sorted(p.name for p in steps_dir.iterdir() if p.is_dir())
    matrix = job_dir / "matrix" / "matrix"
    layers: List[LayerInfo] = []
    if matrix.exists():
        for block in parse_kv_blocks(matrix, "LAYER"):
            name = block.get("NAME")
            if not name:
                continue
            layers.append(
                LayerInfo(
                    name=name,
                    layer_type=block.get("TYPE", "?"),
                    context=block.get("CONTEXT", "?"),
                    side=block.get("SIDE", "?"),
                    polarity=block.get("POLARITY", "?"),
                )
            )
    return JobInfo(name=job_dir.name, steps=steps, layers=layers)


def render_layer(job_dir: Path, step: str, layer: str, dpi: int) -> tuple[Image.Image, ODBRenderer]:
    renderer = ODBRenderer(job_dir, dpi)
    return renderer.render(step, layer), renderer


def profile_size_mm(job_dir: Path, step: str) -> tuple[float, float]:
    contours = parse_profile_contours(job_dir / "steps" / step.lower() / "profile")
    xmin, ymin, xmax, ymax = contours_bounds(contours)
    return (xmax - xmin) * 25.4, (ymax - ymin) * 25.4


class ODBCamApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("ODB++ CAM Image Renderer")
        self.geometry("1320x820")
        self.minsize(1050, 680)

        self.job_dir: Optional[Path] = None
        self.job_tmp = None
        self.job_info: Optional[JobInfo] = None
        self.source_file: Optional[Path] = None
        self.preview_photo: Optional[ImageTk.PhotoImage] = None
        self.preview_image: Optional[Image.Image] = None
        self.preview_zoom = 1.0
        self.preview_offset_x = 0.0
        self.preview_offset_y = 0.0
        self._pan_start: Optional[tuple[int, int]] = None
        self._preview_token = 0
        self._result_queue: queue.Queue = queue.Queue()

        self.step_var = tk.StringVar()
        self.layer_var = tk.StringVar()
        self.dpi_var = tk.IntVar(value=DEFAULT_RENDER_DPI)
        self.zoom_var = tk.StringVar(value="100%")
        self.status_var = tk.StringVar(value="ODB++ TGZ 파일을 열어주세요.")

        self._build_menu()
        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(100, self._poll_results)

    def _build_menu(self) -> None:
        menu = tk.Menu(self)
        file_menu = tk.Menu(menu, tearoff=False)
        file_menu.add_command(label="ODB++ 파일 열기...", accelerator="Ctrl+O", command=self.open_file)
        file_menu.add_separator()
        file_menu.add_command(label="종료", command=self._on_close)
        menu.add_cascade(label="파일", menu=file_menu)
        self.config(menu=menu)
        self.bind_all("<Control-o>", lambda _event: self.open_file())

    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=12)
        root.pack(fill="both", expand=True)

        top = ttk.Frame(root)
        top.pack(fill="x", pady=(0, 10))
        ttk.Button(top, text="파일 업로드", command=self.open_file).pack(side="left")
        self.file_label = ttk.Label(top, text="선택된 파일 없음")
        self.file_label.pack(side="left", padx=12)

        content = ttk.Panedwindow(root, orient="horizontal")
        content.pack(fill="both", expand=True)
        sidebar = ttk.Frame(content, padding=(4, 8))
        preview_frame = ttk.Frame(content, padding=(8, 8))
        content.add(sidebar, weight=0)
        content.add(preview_frame, weight=1)

        ttk.Label(sidebar, text="렌더링 설정", font=("TkDefaultFont", 12, "bold")).grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 12)
        )
        ttk.Label(sidebar, text="Step").grid(row=1, column=0, sticky="w", pady=5)
        self.step_combo = ttk.Combobox(sidebar, textvariable=self.step_var, state="disabled", width=24)
        self.step_combo.grid(row=1, column=1, sticky="ew", pady=5)
        self.step_combo.bind("<<ComboboxSelected>>", lambda _e: self._schedule_preview())

        ttk.Label(sidebar, text="Layer").grid(row=2, column=0, sticky="w", pady=5)
        self.layer_combo = ttk.Combobox(sidebar, textvariable=self.layer_var, state="disabled", width=24)
        self.layer_combo.grid(row=2, column=1, sticky="ew", pady=5)
        self.layer_combo.bind("<<ComboboxSelected>>", lambda _e: self._schedule_preview())

        ttk.Label(sidebar, text="출력 DPI").grid(row=3, column=0, sticky="w", pady=5)
        self.dpi_spin = ttk.Spinbox(sidebar, from_=72, to=10000, increment=50, textvariable=self.dpi_var, width=12)
        self.dpi_spin.grid(row=3, column=1, sticky="w", pady=5)

        self.render_button = ttk.Button(sidebar, text="렌더링 결과 저장", command=self.render_to_file, state="disabled")
        self.render_button.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(14, 12))

        ttk.Separator(sidebar).grid(row=5, column=0, columnspan=2, sticky="ew", pady=6)
        ttk.Label(sidebar, text="ODB 정보", font=("TkDefaultFont", 10, "bold")).grid(
            row=6, column=0, columnspan=2, sticky="w", pady=(4, 6)
        )
        self.info_text = tk.Text(sidebar, width=38, height=20, wrap="word", state="disabled", relief="flat")
        self.info_text.grid(row=7, column=0, columnspan=2, sticky="nsew")
        sidebar.rowconfigure(7, weight=1)
        sidebar.columnconfigure(1, weight=1)

        preview_header = ttk.Frame(preview_frame)
        preview_header.pack(fill="x", pady=(0, 8))
        ttk.Label(preview_header, text="CAM 미리보기", font=("TkDefaultFont", 12, "bold")).pack(side="left")

        zoom_tools = ttk.Frame(preview_header)
        zoom_tools.pack(side="right")
        ttk.Button(zoom_tools, text="−", width=3, command=lambda: self._change_zoom(1 / ZOOM_STEP)).pack(side="left", padx=(0, 2))
        ttk.Label(zoom_tools, textvariable=self.zoom_var, width=7, anchor="center").pack(side="left", padx=2)
        ttk.Button(zoom_tools, text="+", width=3, command=lambda: self._change_zoom(ZOOM_STEP)).pack(side="left", padx=2)
        ttk.Button(zoom_tools, text="맞춤", command=self._fit_preview).pack(side="left", padx=(6, 0))
        ttk.Button(zoom_tools, text="100%", command=self._actual_size_preview).pack(side="left", padx=(4, 0))

        canvas_frame = ttk.Frame(preview_frame)
        canvas_frame.pack(fill="both", expand=True)
        canvas_frame.rowconfigure(0, weight=1)
        canvas_frame.columnconfigure(0, weight=1)

        self.preview_canvas = tk.Canvas(canvas_frame, background="#202020", highlightthickness=0, cursor="fleur")
        self.preview_canvas.grid(row=0, column=0, sticky="nsew")
        self.h_scroll = ttk.Scrollbar(canvas_frame, orient="horizontal", command=self.preview_canvas.xview)
        self.v_scroll = ttk.Scrollbar(canvas_frame, orient="vertical", command=self.preview_canvas.yview)
        self.h_scroll.grid(row=1, column=0, sticky="ew")
        self.v_scroll.grid(row=0, column=1, sticky="ns")
        self.preview_canvas.configure(xscrollcommand=self.h_scroll.set, yscrollcommand=self.v_scroll.set)

        self.preview_canvas.bind("<Configure>", self._on_canvas_resize)
        self.preview_canvas.bind("<MouseWheel>", self._on_mouse_wheel)
        self.preview_canvas.bind("<Button-4>", self._on_mouse_wheel)
        self.preview_canvas.bind("<Button-5>", self._on_mouse_wheel)
        self.preview_canvas.bind("<ButtonPress-1>", self._start_pan)
        self.preview_canvas.bind("<B1-Motion>", self._pan_preview)
        self.preview_canvas.bind("<ButtonRelease-1>", self._end_pan)

        ttk.Label(root, textvariable=self.status_var, anchor="w", relief="sunken").pack(fill="x", pady=(10, 0))

    def open_file(self) -> None:
        path = filedialog.askopenfilename(
            title="ODB++ TGZ 파일 선택",
            filetypes=[("ODB++ TGZ", "*.tgz"), ("Tar GZip", "*.tar.gz"), ("모든 파일", "*.*")],
            defaultextension=".tgz",
        )
        if path:
            self._load_job(Path(path))

    def _load_job(self, path: Path) -> None:
        self.status_var.set("ODB++ 파일을 분석하는 중...")
        self.render_button.configure(state="disabled")
        self.step_combo.configure(state="disabled")
        self.layer_combo.configure(state="disabled")
        self._preview_token += 1
        token = self._preview_token

        def worker() -> None:
            try:
                job_dir, tmp = extract_input(path)
                self._result_queue.put(("loaded", token, path, job_dir, tmp, inspect_job(job_dir)))
            except Exception as exc:
                self._result_queue.put(("error", token, f"파일 로드 실패: {exc}"))

        threading.Thread(target=worker, daemon=True).start()

    def _apply_loaded_job(self, path: Path, job_dir: Path, tmp, info: JobInfo) -> None:
        if self.job_tmp is not None:
            self.job_tmp.cleanup()
        self.job_dir, self.job_tmp, self.job_info, self.source_file = job_dir, tmp, info, path
        self.file_label.configure(text=path.name)
        self.step_combo["values"] = info.steps
        self.step_var.set(self._preferred_step(info.steps))
        self.step_combo.configure(state="readonly")
        labels = [layer.label for layer in info.layers]
        self.layer_combo["values"] = labels
        if labels:
            self.layer_combo.current(self._preferred_layer_index(info.layers))
        self.layer_combo.configure(state="readonly" if labels else "disabled")
        self.render_button.configure(state="normal" if labels else "disabled")
        self.status_var.set(f"로드 완료: {info.name} / {len(info.steps)} steps / {len(info.layers)} layers")
        self._update_info()
        if labels:
            self._schedule_preview()

    @staticmethod
    def _preferred_step(steps: List[str]) -> str:
        for candidate in ("unit", "strip", "pnl"):
            if candidate in steps:
                return candidate
        return steps[0] if steps else ""

    @staticmethod
    def _preferred_layer_index(layers: List[LayerInfo]) -> int:
        for i, layer in enumerate(layers):
            if layer.name.lower() == "l1":
                return i
        for i, layer in enumerate(layers):
            if layer.layer_type.upper() == "SIGNAL":
                return i
        return 0

    def _selected_layer(self) -> Optional[LayerInfo]:
        if not self.job_info:
            return None
        return next((layer for layer in self.job_info.layers if layer.label == self.layer_var.get()), None)

    def _schedule_preview(self) -> None:
        if self.job_dir is None:
            return
        layer, step = self._selected_layer(), self.step_var.get()
        if layer is None or not step:
            return
        self._preview_token += 1
        token = self._preview_token
        self.status_var.set(f"미리보기 생성 중: {step}/{layer.name}")
        self.render_button.configure(state="disabled")
        job_dir = self.job_dir

        def worker() -> None:
            try:
                image, renderer = render_layer(job_dir, step, layer.name, PREVIEW_DPI)
                self._result_queue.put(("preview", token, image, renderer.stats))
            except Exception as exc:
                self._result_queue.put(("error", token, f"미리보기 생성 실패: {exc}"))

        threading.Thread(target=worker, daemon=True).start()
        self._update_info()

    def _poll_results(self) -> None:
        try:
            while True:
                result = self._result_queue.get_nowait()
                kind = result[0]
                if kind == "loaded":
                    _, token, path, job_dir, tmp, info = result
                    if token == self._preview_token:
                        self._apply_loaded_job(path, job_dir, tmp, info)
                    elif tmp is not None:
                        tmp.cleanup()
                elif kind == "preview":
                    _, token, image, stats = result
                    if token == self._preview_token:
                        self.preview_image = image
                        self.after_idle(self._fit_preview)
                        self.render_button.configure(state="normal")
                        self.status_var.set(
                            f"미리보기 완료 | {image.width}×{image.height}px @ {PREVIEW_DPI} DPI | "
                            f"pads={stats.pads}, lines={stats.lines}, surfaces={stats.surfaces}, warnings={stats.unsupported}"
                        )
                elif kind == "rendered":
                    _, image, output, stats = result
                    self.render_button.configure(state="normal")
                    self.status_var.set(f"렌더링 완료: {output}")
                    messagebox.showinfo(
                        "렌더링 완료",
                        f"결과 파일을 저장했습니다.\n\n{output}\n\n크기: {image.width} × {image.height} px\n"
                        f"Pads: {stats.pads}, Lines: {stats.lines}, Surfaces: {stats.surfaces}\nWarnings: {stats.unsupported}",
                    )
                elif kind == "error":
                    _, token, msg = result
                    if token != self._preview_token:
                        continue
                    self.render_button.configure(state="normal" if self.job_dir else "disabled")
                    self.status_var.set(msg)
                    messagebox.showerror("오류", msg)
        except queue.Empty:
            pass
        finally:
            self.after(100, self._poll_results)

    def _on_canvas_resize(self, _event=None) -> None:
        if self.preview_image is not None and self.preview_photo is None:
            self._fit_preview()
        else:
            self._draw_preview()

    def _fit_preview(self) -> None:
        if self.preview_image is None:
            self._draw_preview()
            return
        cw = max(1, self.preview_canvas.winfo_width() - 32)
        ch = max(1, self.preview_canvas.winfo_height() - 32)
        fit = min(cw / self.preview_image.width, ch / self.preview_image.height)
        self.preview_zoom = max(MIN_ZOOM, min(MAX_ZOOM, fit))
        self.preview_offset_x = 0.0
        self.preview_offset_y = 0.0
        self._draw_preview()

    def _actual_size_preview(self) -> None:
        if self.preview_image is None:
            return
        self.preview_zoom = 1.0
        self.preview_offset_x = 0.0
        self.preview_offset_y = 0.0
        self._draw_preview()

    def _change_zoom(self, factor: float, anchor_x: Optional[float] = None, anchor_y: Optional[float] = None) -> None:
        if self.preview_image is None:
            return
        old_zoom = self.preview_zoom
        new_zoom = max(MIN_ZOOM, min(MAX_ZOOM, old_zoom * factor))
        if abs(new_zoom - old_zoom) < 1e-9:
            return

        cw = max(1, self.preview_canvas.winfo_width())
        ch = max(1, self.preview_canvas.winfo_height())
        ax = cw / 2 if anchor_x is None else anchor_x
        ay = ch / 2 if anchor_y is None else anchor_y

        image_x = (ax - cw / 2 - self.preview_offset_x) / old_zoom
        image_y = (ay - ch / 2 - self.preview_offset_y) / old_zoom
        self.preview_zoom = new_zoom
        self.preview_offset_x = ax - cw / 2 - image_x * new_zoom
        self.preview_offset_y = ay - ch / 2 - image_y * new_zoom
        self._draw_preview()

    def _on_mouse_wheel(self, event) -> str:
        if self.preview_image is None:
            return "break"
        if getattr(event, "num", None) == 4 or getattr(event, "delta", 0) > 0:
            factor = ZOOM_STEP
        else:
            factor = 1 / ZOOM_STEP
        self._change_zoom(factor, event.x, event.y)
        return "break"

    def _start_pan(self, event) -> None:
        if self.preview_image is not None:
            self._pan_start = (event.x, event.y)
            self.preview_canvas.configure(cursor="hand2")

    def _pan_preview(self, event) -> None:
        if self._pan_start is None or self.preview_image is None:
            return
        x0, y0 = self._pan_start
        self.preview_offset_x += event.x - x0
        self.preview_offset_y += event.y - y0
        self._pan_start = (event.x, event.y)
        self._draw_preview()

    def _end_pan(self, _event) -> None:
        self._pan_start = None
        self.preview_canvas.configure(cursor="fleur")

    def _draw_preview(self) -> None:
        self.preview_canvas.delete("all")
        if self.preview_image is None:
            self.preview_photo = None
            self.zoom_var.set("100%")
            self.preview_canvas.create_text(
                max(10, self.preview_canvas.winfo_width() // 2),
                max(10, self.preview_canvas.winfo_height() // 2),
                text="ODB++ 파일을 열고 Layer를 선택하면 미리보기가 표시됩니다.",
                fill="white",
            )
            return

        width = max(1, int(round(self.preview_image.width * self.preview_zoom)))
        height = max(1, int(round(self.preview_image.height * self.preview_zoom)))
        resample = Image.Resampling.NEAREST if self.preview_zoom >= 1.0 else Image.Resampling.LANCZOS
        image = self.preview_image.resize((width, height), resample=resample)
        display = ImageOps.colorize(image, black="black", white="white")
        self.preview_photo = ImageTk.PhotoImage(display)

        cw = max(1, self.preview_canvas.winfo_width())
        ch = max(1, self.preview_canvas.winfo_height())
        cx = cw / 2 + self.preview_offset_x
        cy = ch / 2 + self.preview_offset_y
        self.preview_canvas.create_image(cx, cy, image=self.preview_photo, anchor="center", tags=("preview",))

        bbox = self.preview_canvas.bbox("preview")
        if bbox:
            margin = 100
            scrollregion = (
                min(0, bbox[0] - margin),
                min(0, bbox[1] - margin),
                max(cw, bbox[2] + margin),
                max(ch, bbox[3] + margin),
            )
            self.preview_canvas.configure(scrollregion=scrollregion)

        self.zoom_var.set(f"{self.preview_zoom * 100:.0f}%")

    def _update_info(self) -> None:
        if not self.job_dir or not self.job_info:
            return
        layer, step = self._selected_layer(), self.step_var.get()
        lines = [
            f"Job: {self.job_info.name}",
            f"Source: {self.source_file.name if self.source_file else '-'}",
            f"Step: {step or '-'}",
        ]
        if step:
            try:
                width_mm, height_mm = profile_size_mm(self.job_dir, step)
                lines.append(f"Profile: {width_mm:.3f} × {height_mm:.3f} mm")
            except Exception:
                pass
        if layer:
            lines.extend(
                [
                    "",
                    f"Layer: {layer.name}",
                    f"Type: {layer.layer_type}",
                    f"Context: {layer.context}",
                    f"Side: {layer.side}",
                    f"Polarity: {layer.polarity}",
                    "",
                    f"Preview DPI: {PREVIEW_DPI}",
                    f"Output DPI: {self.dpi_var.get()}",
                    "",
                    "Preview controls:",
                    "Mouse wheel: Zoom",
                    "Left drag: Pan",
                    "맞춤: Fit to window",
                    "100%: Actual preview pixels",
                ]
            )
        self.info_text.configure(state="normal")
        self.info_text.delete("1.0", "end")
        self.info_text.insert("1.0", "\n".join(lines))
        self.info_text.configure(state="disabled")

    def render_to_file(self) -> None:
        if self.job_dir is None:
            return
        layer, step = self._selected_layer(), self.step_var.get()
        if layer is None or not step:
            messagebox.showwarning("선택 필요", "렌더링할 Step과 Layer를 선택해주세요.")
            return
        try:
            dpi = int(self.dpi_var.get())
            if not 1 <= dpi <= 10000:
                raise ValueError
        except Exception:
            messagebox.showerror("DPI 오류", "DPI는 1~10000 사이의 정수여야 합니다.")
            return
        default_name = f"{self.job_info.name}_{step}_{layer.name}_{dpi}dpi.png" if self.job_info else "cam.png"
        output = filedialog.asksaveasfilename(
            title="CAM Image 저장",
            defaultextension=".png",
            initialfile=default_name,
            filetypes=[("PNG Image", "*.png")],
        )
        if not output:
            return
        output_path, job_dir = Path(output), self.job_dir
        self.render_button.configure(state="disabled")
        self.status_var.set(f"렌더링 중: {step}/{layer.name} @ {dpi} DPI")

        def worker() -> None:
            try:
                image, renderer = render_layer(job_dir, step, layer.name, dpi)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                image.save(output_path, optimize=True, dpi=(dpi, dpi))
                self._result_queue.put(("rendered", image, output_path, renderer.stats))
            except Exception as exc:
                self._result_queue.put(("error", self._preview_token, f"렌더링 실패: {exc}"))

        threading.Thread(target=worker, daemon=True).start()

    def _on_close(self) -> None:
        if self.job_tmp is not None:
            self.job_tmp.cleanup()
        self.destroy()


def main() -> int:
    app = ODBCamApp()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
