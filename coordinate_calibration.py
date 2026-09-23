#!/usr/bin/env python3
from __future__ import annotations

import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Optional

from PIL import Image, ImageOps, ImageTk

from app import inspect_job, render_layer
from coordinate_transform import (
    AffineTransform,
    CalibrationPoint,
    extract_xy_from_filename,
    fit_affine,
    load_calibration,
    save_calibration,
)
from odb_cam_renderer import extract_input

DEFAULT_DPI = 600
MIN_ZOOM = 0.05
MAX_ZOOM = 16.0
ZOOM_STEP = 1.25


class CoordinateCalibrationApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("AOI ↔ ODB Coordinate Calibration")
        self.geometry("1450x900")
        self.minsize(1180, 720)

        self.job_dir: Optional[Path] = None
        self.job_tmp = None
        self.job_info = None
        self.source_file: Optional[Path] = None

        self.preview_image: Optional[Image.Image] = None
        self.preview_photo: Optional[ImageTk.PhotoImage] = None
        self.preview_zoom = 1.0
        self.preview_offset_x = 0.0
        self.preview_offset_y = 0.0
        self._pan_start: Optional[tuple[int, int]] = None
        self._pick_mode = False
        self._marker: Optional[tuple[float, float]] = None

        self.crop_image: Optional[Image.Image] = None
        self.crop_photo: Optional[ImageTk.PhotoImage] = None
        self.crop_path: Optional[Path] = None

        self.points: list[CalibrationPoint] = []
        self.transform: Optional[AffineTransform] = None
        self._result_queue: queue.Queue = queue.Queue()
        self._token = 0

        self.step_var = tk.StringVar()
        self.layer_var = tk.StringVar()
        self.dpi_var = tk.IntVar(value=DEFAULT_DPI)
        self.aoi_x_var = tk.StringVar()
        self.aoi_y_var = tk.StringVar()
        self.odb_x_var = tk.StringVar()
        self.odb_y_var = tk.StringVar()
        self.test_x_var = tk.StringVar()
        self.test_y_var = tk.StringVar()
        self.result_var = tk.StringVar(value="3개 이상의 대응점을 등록하세요.")
        self.status_var = tk.StringVar(value="ODB++ 파일을 열어주세요.")
        self.pick_var = tk.StringVar(value="ODB 좌표 선택")

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(100, self._poll_results)

    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=10)
        root.pack(fill="both", expand=True)

        top = ttk.Frame(root)
        top.pack(fill="x", pady=(0, 8))
        ttk.Button(top, text="ODB++ 파일 열기", command=self.open_odb).pack(side="left")
        self.file_label = ttk.Label(top, text="선택된 ODB 없음")
        self.file_label.pack(side="left", padx=10)
        ttk.Label(top, text="Step").pack(side="left", padx=(20, 4))
        self.step_combo = ttk.Combobox(top, textvariable=self.step_var, state="disabled", width=18)
        self.step_combo.pack(side="left")
        self.step_combo.bind("<<ComboboxSelected>>", lambda _e: self._schedule_render())
        ttk.Label(top, text="Layer").pack(side="left", padx=(12, 4))
        self.layer_combo = ttk.Combobox(top, textvariable=self.layer_var, state="disabled", width=28)
        self.layer_combo.pack(side="left")
        self.layer_combo.bind("<<ComboboxSelected>>", lambda _e: self._schedule_render())
        ttk.Label(top, text="DPI").pack(side="left", padx=(12, 4))
        ttk.Spinbox(top, from_=72, to=10000, increment=50, textvariable=self.dpi_var, width=8).pack(side="left")
        ttk.Button(top, text="미리보기 갱신", command=self._schedule_render).pack(side="left", padx=(8, 0))

        pane = ttk.Panedwindow(root, orient="horizontal")
        pane.pack(fill="both", expand=True)
        left = ttk.Frame(pane, padding=(2, 4))
        right = ttk.Frame(pane, padding=(8, 4))
        pane.add(left, weight=0)
        pane.add(right, weight=1)

        self._build_left(left)
        self._build_preview(right)

        ttk.Label(root, textvariable=self.status_var, anchor="w", relief="sunken").pack(fill="x", pady=(8, 0))

    def _build_left(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)

        ttk.Label(parent, text="1. AOI Crop", font=("TkDefaultFont", 11, "bold")).grid(row=0, column=0, sticky="w")
        ttk.Button(parent, text="Crop 이미지 열기", command=self.open_crop).grid(row=1, column=0, sticky="ew", pady=(6, 4))
        self.crop_label = ttk.Label(parent, text="Crop 이미지 없음", anchor="center", width=42)
        self.crop_label.grid(row=2, column=0, sticky="ew", pady=(0, 8))

        coord = ttk.LabelFrame(parent, text="AOI 좌표", padding=8)
        coord.grid(row=3, column=0, sticky="ew", pady=4)
        ttk.Label(coord, text="X").grid(row=0, column=0, padx=(0, 4))
        ttk.Entry(coord, textvariable=self.aoi_x_var, width=14).grid(row=0, column=1, padx=(0, 10))
        ttk.Label(coord, text="Y").grid(row=0, column=2, padx=(0, 4))
        ttk.Entry(coord, textvariable=self.aoi_y_var, width=14).grid(row=0, column=3)

        ttk.Label(parent, text="2. ODB 대응점", font=("TkDefaultFont", 11, "bold")).grid(row=4, column=0, sticky="w", pady=(12, 2))
        ttk.Button(parent, textvariable=self.pick_var, command=self.arm_pick).grid(row=5, column=0, sticky="ew", pady=4)

        odb = ttk.Frame(parent)
        odb.grid(row=6, column=0, sticky="ew")
        ttk.Label(odb, text="ODB X").grid(row=0, column=0)
        ttk.Entry(odb, textvariable=self.odb_x_var, width=13, state="readonly").grid(row=0, column=1, padx=(4, 10))
        ttk.Label(odb, text="ODB Y").grid(row=0, column=2)
        ttk.Entry(odb, textvariable=self.odb_y_var, width=13, state="readonly").grid(row=0, column=3, padx=(4, 0))
        ttk.Button(parent, text="대응점 추가", command=self.add_point).grid(row=7, column=0, sticky="ew", pady=(6, 6))

        self.tree = ttk.Treeview(parent, columns=("aoi_x", "aoi_y", "odb_x", "odb_y"), show="headings", height=7)
        for key, title, width in (
            ("aoi_x", "AOI X", 78),
            ("aoi_y", "AOI Y", 78),
            ("odb_x", "ODB px X", 82),
            ("odb_y", "ODB px Y", 82),
        ):
            self.tree.heading(key, text=title)
            self.tree.column(key, width=width, anchor="e")
        self.tree.grid(row=8, column=0, sticky="ew")
        btns = ttk.Frame(parent)
        btns.grid(row=9, column=0, sticky="ew", pady=(4, 8))
        ttk.Button(btns, text="선택 삭제", command=self.delete_selected).pack(side="left")
        ttk.Button(btns, text="전체 삭제", command=self.clear_points).pack(side="left", padx=4)

        ttk.Label(parent, text="3. 변환 결과", font=("TkDefaultFont", 11, "bold")).grid(row=10, column=0, sticky="w", pady=(6, 2))
        ttk.Label(parent, textvariable=self.result_var, wraplength=350, justify="left").grid(row=11, column=0, sticky="ew", pady=(2, 8))

        test = ttk.LabelFrame(parent, text="AOI 좌표 예측", padding=8)
        test.grid(row=12, column=0, sticky="ew", pady=4)
        ttk.Label(test, text="X").grid(row=0, column=0)
        ttk.Entry(test, textvariable=self.test_x_var, width=12).grid(row=0, column=1, padx=(4, 8))
        ttk.Label(test, text="Y").grid(row=0, column=2)
        ttk.Entry(test, textvariable=self.test_y_var, width=12).grid(row=0, column=3, padx=(4, 8))
        ttk.Button(test, text="ODB 위치 표시", command=self.show_prediction).grid(row=1, column=0, columnspan=4, sticky="ew", pady=(6, 0))

        io = ttk.Frame(parent)
        io.grid(row=13, column=0, sticky="ew", pady=(10, 0))
        ttk.Button(io, text="Calibration 저장", command=self.save_calibration_file).pack(side="left", fill="x", expand=True)
        ttk.Button(io, text="불러오기", command=self.load_calibration_file).pack(side="left", fill="x", expand=True, padx=(4, 0))

    def _build_preview(self, parent: ttk.Frame) -> None:
        header = ttk.Frame(parent)
        header.pack(fill="x", pady=(0, 6))
        ttk.Label(header, text="ODB CAM 미리보기", font=("TkDefaultFont", 11, "bold")).pack(side="left")
        ttk.Label(header, text="좌클릭 드래그: 이동 / 휠: 확대축소").pack(side="right")

        self.canvas = tk.Canvas(parent, background="#202020", highlightthickness=0, cursor="fleur")
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda _e: self._draw_preview())
        self.canvas.bind("<MouseWheel>", self._on_wheel)
        self.canvas.bind("<Button-4>", self._on_wheel)
        self.canvas.bind("<Button-5>", self._on_wheel)
        self.canvas.bind("<ButtonPress-1>", self._left_press)
        self.canvas.bind("<B1-Motion>", self._pan)
        self.canvas.bind("<ButtonRelease-1>", self._left_release)

    def open_odb(self) -> None:
        path = filedialog.askopenfilename(
            title="ODB++ TGZ 파일 선택",
            filetypes=[("ODB++ TGZ", "*.tgz"), ("Tar GZip", "*.tar.gz"), ("모든 파일", "*.*")],
        )
        if not path:
            return
        self.status_var.set("ODB++ 파일 분석 중...")
        self._token += 1
        token = self._token

        def worker() -> None:
            try:
                job_dir, tmp = extract_input(Path(path))
                info = inspect_job(job_dir)
                self._result_queue.put(("loaded", token, Path(path), job_dir, tmp, info))
            except Exception as exc:
                self._result_queue.put(("error", token, f"ODB 로드 실패: {exc}"))

        threading.Thread(target=worker, daemon=True).start()

    def _apply_loaded(self, path: Path, job_dir: Path, tmp, info) -> None:
        if self.job_tmp is not None:
            self.job_tmp.cleanup()
        self.source_file, self.job_dir, self.job_tmp, self.job_info = path, job_dir, tmp, info
        self.file_label.configure(text=path.name)

        self.step_combo["values"] = info.steps
        preferred = next((s for s in ("unit", "strip", "pnl") if s in info.steps), info.steps[0] if info.steps else "")
        self.step_var.set(preferred)
        self.step_combo.configure(state="readonly" if info.steps else "disabled")

        labels = [layer.label for layer in info.layers]
        self.layer_combo["values"] = labels
        if labels:
            self.layer_combo.current(0)
        self.layer_combo.configure(state="readonly" if labels else "disabled")
        self.status_var.set(f"로드 완료: {info.name}")
        if preferred and labels:
            self._schedule_render()

    def _selected_layer_name(self) -> str | None:
        if not self.job_info:
            return None
        current = self.layer_var.get()
        for layer in self.job_info.layers:
            if layer.label == current:
                return layer.name
        return None

    def _schedule_render(self) -> None:
        if self.job_dir is None:
            return
        step = self.step_var.get()
        layer = self._selected_layer_name()
        if not step or not layer:
            return
        try:
            dpi = int(self.dpi_var.get())
        except Exception:
            messagebox.showerror("DPI 오류", "DPI는 정수여야 합니다.")
            return

        self._token += 1
        token = self._token
        self.status_var.set(f"ODB 미리보기 생성 중: {step}/{layer} @ {dpi} DPI")
        job_dir = self.job_dir

        def worker() -> None:
            try:
                image, _renderer = render_layer(job_dir, step, layer, dpi)
                self._result_queue.put(("preview", token, image))
            except Exception as exc:
                self._result_queue.put(("error", token, f"미리보기 생성 실패: {exc}"))

        threading.Thread(target=worker, daemon=True).start()

    def _poll_results(self) -> None:
        try:
            while True:
                result = self._result_queue.get_nowait()
                kind = result[0]
                if kind == "loaded":
                    _, token, path, job_dir, tmp, info = result
                    if token == self._token:
                        self._apply_loaded(path, job_dir, tmp, info)
                    else:
                        tmp.cleanup()
                elif kind == "preview":
                    _, token, image = result
                    if token == self._token:
                        self.preview_image = image
                        self._marker = None
                        self.after_idle(self.fit_preview)
                        self.status_var.set(f"미리보기 완료: {image.width}×{image.height}px")
                elif kind == "error":
                    _, token, msg = result
                    if token == self._token:
                        self.status_var.set(msg)
                        messagebox.showerror("오류", msg)
        except queue.Empty:
            pass
        finally:
            self.after(100, self._poll_results)

    def open_crop(self) -> None:
        path = filedialog.askopenfilename(
            title="AOI Crop 이미지 선택",
            filetypes=[("Image", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("모든 파일", "*.*")],
        )
        if not path:
            return
        try:
            image = Image.open(path)
            image.load()
            image = ImageOps.grayscale(image)
        except Exception as exc:
            messagebox.showerror("이미지 오류", str(exc))
            return

        self.crop_path = Path(path)
        self.crop_image = image
        thumb = image.copy()
        thumb.thumbnail((350, 200), Image.Resampling.LANCZOS)
        self.crop_photo = ImageTk.PhotoImage(thumb)
        self.crop_label.configure(image=self.crop_photo, text="")

        coords = extract_xy_from_filename(self.crop_path.name)
        if coords:
            self.aoi_x_var.set(f"{coords[0]:g}")
            self.aoi_y_var.set(f"{coords[1]:g}")
            self.test_x_var.set(f"{coords[0]:g}")
            self.test_y_var.set(f"{coords[1]:g}")
            self.status_var.set(f"파일명에서 AOI 좌표 인식: X={coords[0]:g}, Y={coords[1]:g}")
        else:
            self.status_var.set("Crop 로드 완료. AOI X/Y 좌표를 직접 입력하세요.")

    def arm_pick(self) -> None:
        if self.preview_image is None:
            messagebox.showwarning("ODB 필요", "먼저 ODB 미리보기를 생성하세요.")
            return
        self._pick_mode = True
        self.pick_var.set("ODB 화면에서 위치를 클릭하세요")
        self.canvas.configure(cursor="crosshair")

    def _canvas_to_image(self, x: float, y: float) -> tuple[float, float] | None:
        if self.preview_image is None:
            return None
        cw, ch = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
        cx = cw / 2 + self.preview_offset_x
        cy = ch / 2 + self.preview_offset_y
        ix = (x - cx) / self.preview_zoom + self.preview_image.width / 2
        iy = (y - cy) / self.preview_zoom + self.preview_image.height / 2
        if not (0 <= ix < self.preview_image.width and 0 <= iy < self.preview_image.height):
            return None
        return ix, iy

    def _left_press(self, event) -> None:
        if self._pick_mode:
            point = self._canvas_to_image(event.x, event.y)
            if point is None:
                self.status_var.set("이미지 영역 안을 클릭하세요.")
                return
            self.odb_x_var.set(f"{point[0]:.3f}")
            self.odb_y_var.set(f"{point[1]:.3f}")
            self._marker = point
            self._pick_mode = False
            self.pick_var.set("ODB 좌표 선택")
            self.canvas.configure(cursor="fleur")
            self._draw_preview()
            self.status_var.set(f"ODB 대응점 선택: ({point[0]:.2f}, {point[1]:.2f}) px")
            return
        self._pan_start = (event.x, event.y)

    def _pan(self, event) -> None:
        if self._pick_mode or self._pan_start is None:
            return
        x0, y0 = self._pan_start
        self.preview_offset_x += event.x - x0
        self.preview_offset_y += event.y - y0
        self._pan_start = (event.x, event.y)
        self._draw_preview()

    def _left_release(self, _event) -> None:
        self._pan_start = None

    def _on_wheel(self, event) -> str:
        if self.preview_image is None:
            return "break"
        factor = ZOOM_STEP if getattr(event, "num", None) == 4 or getattr(event, "delta", 0) > 0 else 1 / ZOOM_STEP
        old_zoom = self.preview_zoom
        new_zoom = max(MIN_ZOOM, min(MAX_ZOOM, old_zoom * factor))
        if abs(new_zoom - old_zoom) < 1e-12:
            return "break"

        cw, ch = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
        ax, ay = event.x, event.y
        image_x = (ax - cw / 2 - self.preview_offset_x) / old_zoom
        image_y = (ay - ch / 2 - self.preview_offset_y) / old_zoom
        self.preview_zoom = new_zoom
        self.preview_offset_x = ax - cw / 2 - image_x * new_zoom
        self.preview_offset_y = ay - ch / 2 - image_y * new_zoom
        self._draw_preview()
        return "break"

    def fit_preview(self) -> None:
        if self.preview_image is None:
            return
        cw = max(1, self.canvas.winfo_width() - 30)
        ch = max(1, self.canvas.winfo_height() - 30)
        self.preview_zoom = max(MIN_ZOOM, min(MAX_ZOOM, min(cw / self.preview_image.width, ch / self.preview_image.height)))
        self.preview_offset_x = 0.0
        self.preview_offset_y = 0.0
        self._draw_preview()

    def _draw_preview(self) -> None:
        self.canvas.delete("all")
        if self.preview_image is None:
            self.preview_photo = None
            self.canvas.create_text(
                max(10, self.canvas.winfo_width() // 2),
                max(10, self.canvas.winfo_height() // 2),
                text="ODB++ 파일을 열면 CAM 이미지가 표시됩니다.",
                fill="white",
            )
            return

        width = max(1, int(round(self.preview_image.width * self.preview_zoom)))
        height = max(1, int(round(self.preview_image.height * self.preview_zoom)))
        resample = Image.Resampling.NEAREST if self.preview_zoom >= 1.0 else Image.Resampling.LANCZOS
        image = self.preview_image.resize((width, height), resample=resample)
        self.preview_photo = ImageTk.PhotoImage(ImageOps.colorize(image, black="black", white="white"))

        cw, ch = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
        cx = cw / 2 + self.preview_offset_x
        cy = ch / 2 + self.preview_offset_y
        self.canvas.create_image(cx, cy, image=self.preview_photo, anchor="center")

        if self._marker is not None:
            mx = cx + (self._marker[0] - self.preview_image.width / 2) * self.preview_zoom
            my = cy + (self._marker[1] - self.preview_image.height / 2) * self.preview_zoom
            size = 12
            self.canvas.create_line(mx - size, my, mx + size, my, fill="#ff4040", width=2)
            self.canvas.create_line(mx, my - size, mx, my + size, fill="#ff4040", width=2)
            self.canvas.create_oval(mx - 5, my - 5, mx + 5, my + 5, outline="#ff4040", width=2)

    def add_point(self) -> None:
        try:
            p = CalibrationPoint(
                float(self.aoi_x_var.get()),
                float(self.aoi_y_var.get()),
                float(self.odb_x_var.get()),
                float(self.odb_y_var.get()),
            )
        except ValueError:
            messagebox.showerror("좌표 오류", "AOI X/Y와 ODB 좌표를 모두 지정하세요.")
            return

        self.points.append(p)
        self._refresh_points()
        self.odb_x_var.set("")
        self.odb_y_var.set("")
        self._fit_transform()

    def _refresh_points(self) -> None:
        for item in self.tree.get_children():
            self.tree.delete(item)
        for p in self.points:
            self.tree.insert("", "end", values=(f"{p.aoi_x:g}", f"{p.aoi_y:g}", f"{p.odb_x:.3f}", f"{p.odb_y:.3f}"))

    def delete_selected(self) -> None:
        selected = self.tree.selection()
        if not selected:
            return
        indices = sorted((self.tree.index(item) for item in selected), reverse=True)
        for index in indices:
            del self.points[index]
        self._refresh_points()
        self._fit_transform()

    def clear_points(self) -> None:
        self.points.clear()
        self.transform = None
        self._refresh_points()
        self.result_var.set("3개 이상의 대응점을 등록하세요.")
        self._marker = None
        self._draw_preview()

    def _fit_transform(self) -> None:
        if len(self.points) < 3:
            self.transform = None
            self.result_var.set(f"현재 {len(self.points)}개 대응점. 최소 3개가 필요합니다.")
            return
        try:
            self.transform = fit_affine(self.points)
        except ValueError as exc:
            self.transform = None
            self.result_var.set(str(exc))
            return

        t = self.transform
        self.result_var.set(
            "ODB_X = "
            f"{t.a:.8g}·AOI_X + {t.b:.8g}·AOI_Y + {t.c:.8g}\n"
            "ODB_Y = "
            f"{t.d:.8g}·AOI_X + {t.e:.8g}·AOI_Y + {t.f:.8g}\n"
            f"RMSE = {t.rmse:.3f} px / 대응점 {t.point_count}개"
        )

    def show_prediction(self) -> None:
        if self.transform is None:
            messagebox.showwarning("Calibration 필요", "먼저 3개 이상의 대응점으로 Calibration을 계산하세요.")
            return
        try:
            x = float(self.test_x_var.get())
            y = float(self.test_y_var.get())
        except ValueError:
            messagebox.showerror("좌표 오류", "예측할 AOI X/Y를 입력하세요.")
            return
        odb_x, odb_y = self.transform.apply(x, y)
        self._marker = (odb_x, odb_y)
        self._draw_preview()
        self.status_var.set(f"예측 ODB 위치: ({odb_x:.2f}, {odb_y:.2f}) px")

    def save_calibration_file(self) -> None:
        if self.transform is None:
            messagebox.showwarning("Calibration 필요", "저장할 Calibration 결과가 없습니다.")
            return
        path = filedialog.asksaveasfilename(
            title="Calibration 저장",
            defaultextension=".json",
            filetypes=[("JSON", "*.json")],
            initialfile="aoi_odb_calibration.json",
        )
        if not path:
            return
        metadata = {
            "odb_source": self.source_file.name if self.source_file else None,
            "step": self.step_var.get(),
            "layer": self._selected_layer_name(),
            "dpi": int(self.dpi_var.get()),
            "image_width": self.preview_image.width if self.preview_image else None,
            "image_height": self.preview_image.height if self.preview_image else None,
        }
        save_calibration(path, self.transform, self.points, metadata)
        self.status_var.set(f"Calibration 저장: {path}")

    def load_calibration_file(self) -> None:
        path = filedialog.askopenfilename(title="Calibration 불러오기", filetypes=[("JSON", "*.json")])
        if not path:
            return
        try:
            transform, points, metadata = load_calibration(path)
        except Exception as exc:
            messagebox.showerror("불러오기 오류", str(exc))
            return
        self.transform = transform
        self.points = points
        self._refresh_points()
        self._fit_transform()
        expected_dpi = metadata.get("dpi")
        current_dpi = int(self.dpi_var.get())
        if expected_dpi and expected_dpi != current_dpi:
            messagebox.showwarning(
                "DPI 불일치",
                f"이 Calibration은 {expected_dpi} DPI 기준입니다.\n현재 미리보기는 {current_dpi} DPI입니다.",
            )
        self.status_var.set(f"Calibration 불러오기: {path}")

    def _on_close(self) -> None:
        if self.job_tmp is not None:
            self.job_tmp.cleanup()
        self.destroy()


def main() -> int:
    app = CoordinateCalibrationApp()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
