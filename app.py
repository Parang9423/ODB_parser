#!/usr/bin/env python3
from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

from app_core import App as _CoreApp


class App(_CoreApp):
    """UI compatibility wrapper that separates Composite add/edit actions."""

    def _ui(self) -> None:
        super()._ui()

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

        # 수정 버튼을 독립된 열에 배치하고 Composite 목록 폭만 확장한다.
        self.comp_tree.grid_configure(columnspan=5)
        ttk.Button(
            self.comp_frame,
            text="선택 레이어 수정",
            command=self.update_composite_row,
        ).grid(row=2, column=4, padx=(4, 0), sticky="ew")

    def add_composite_row(self) -> None:
        """Always append a new Composite row regardless of current selection."""
        selected = self.comp_tree.selection()
        if selected:
            self.comp_tree.selection_remove(*selected)
        _CoreApp.upsert_composite_row(self)

    def update_composite_row(self) -> None:
        """Update only the explicitly selected Composite row."""
        if not self.comp_tree.selection():
            messagebox.showwarning("Composite", "수정할 Composite Layer를 목록에서 선택하세요.")
            return
        _CoreApp.upsert_composite_row(self)


if __name__ == "__main__":
    App().mainloop()
