#!/usr/bin/env python3
from __future__ import annotations

from tkinter import messagebox, ttk

from coordinate_calibration import CoordinateCalibrationApp


class CoordinateCalibrationFixedApp(CoordinateCalibrationApp):
    """Calibration UI with explicit calculate action and visible point count.

    This keeps the original rendering / coordinate picking implementation intact,
    but makes calibration state explicit so the user can immediately verify that
    point registration and affine fitting are actually happening.
    """

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

        ttk.Button(parent, text="대응점 추가", command=self.add_point).grid(row=7, column=0, sticky="ew", pady=(6, 4))
        self.point_count_label = ttk.Label(parent, text="등록된 대응점: 0개", anchor="w")
        self.point_count_label.grid(row=8, column=0, sticky="ew", pady=(0, 4))

        self.tree = ttk.Treeview(parent, columns=("aoi_x", "aoi_y", "odb_x", "odb_y"), show="headings", height=7)
        for key, title, width in (
            ("aoi_x", "AOI X", 78),
            ("aoi_y", "AOI Y", 78),
            ("odb_x", "ODB px X", 82),
            ("odb_y", "ODB px Y", 82),
        ):
            self.tree.heading(key, text=title)
            self.tree.column(key, width=width, anchor="e")
        self.tree.grid(row=9, column=0, sticky="ew")

        btns = ttk.Frame(parent)
        btns.grid(row=10, column=0, sticky="ew", pady=(4, 8))
        ttk.Button(btns, text="선택 삭제", command=self.delete_selected).pack(side="left")
        ttk.Button(btns, text="전체 삭제", command=self.clear_points).pack(side="left", padx=4)

        ttk.Label(parent, text="3. Calibration", font=("TkDefaultFont", 11, "bold")).grid(row=11, column=0, sticky="w", pady=(6, 2))
        ttk.Button(parent, text="Calibration 계산", command=self.calculate_calibration).grid(row=12, column=0, sticky="ew", pady=(4, 6))
        ttk.Label(parent, textvariable=self.result_var, wraplength=350, justify="left").grid(row=13, column=0, sticky="ew", pady=(2, 8))

        test = ttk.LabelFrame(parent, text="AOI 좌표 예측", padding=8)
        test.grid(row=14, column=0, sticky="ew", pady=4)
        ttk.Label(test, text="X").grid(row=0, column=0)
        ttk.Entry(test, textvariable=self.test_x_var, width=12).grid(row=0, column=1, padx=(4, 8))
        ttk.Label(test, text="Y").grid(row=0, column=2)
        ttk.Entry(test, textvariable=self.test_y_var, width=12).grid(row=0, column=3, padx=(4, 8))
        ttk.Button(test, text="ODB 위치 표시", command=self.show_prediction).grid(row=1, column=0, columnspan=4, sticky="ew", pady=(6, 0))

        io = ttk.Frame(parent)
        io.grid(row=15, column=0, sticky="ew", pady=(10, 0))
        ttk.Button(io, text="Calibration 저장", command=self.save_calibration_file).pack(side="left", fill="x", expand=True)
        ttk.Button(io, text="불러오기", command=self.load_calibration_file).pack(side="left", fill="x", expand=True, padx=(4, 0))

    def _refresh_points(self) -> None:
        super()._refresh_points()
        if hasattr(self, "point_count_label"):
            self.point_count_label.configure(text=f"등록된 대응점: {len(self.points)}개")

    def add_point(self) -> None:
        before = len(self.points)
        super().add_point()
        after = len(self.points)
        if after > before:
            self.status_var.set(f"대응점 {after}개 등록 완료. 3개 이상이면 'Calibration 계산'을 눌러 확인하세요.")

    def calculate_calibration(self) -> None:
        if len(self.points) < 3:
            self.transform = None
            self.result_var.set(f"현재 {len(self.points)}개 대응점. 최소 3개가 필요합니다.")
            self.status_var.set("Calibration 계산 실패: 대응점이 부족합니다.")
            messagebox.showwarning("대응점 부족", f"현재 {len(self.points)}개입니다. 서로 떨어진 대응점을 최소 3개 등록하세요.")
            return

        self._fit_transform()
        if self.transform is None:
            self.status_var.set("Calibration 계산 실패: 대응점 배치를 확인하세요.")
            messagebox.showerror(
                "Calibration 실패",
                "Affine Transform을 계산할 수 없습니다.\n\n세 AOI 좌표가 한 직선상에 있지 않은지 확인하고, Unit 전체에 퍼진 대응점을 사용하세요.",
            )
            return

        t = self.transform
        self.status_var.set(f"Calibration 완료: {t.point_count}개 대응점 / RMSE {t.rmse:.3f}px")


def main() -> int:
    app = CoordinateCalibrationFixedApp()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
