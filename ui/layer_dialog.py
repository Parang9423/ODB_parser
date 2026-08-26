"""Lazy metadata-only layer selection dialog."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk
from typing import List, Optional

from app_core import LayerInfo


class LayerDialog(tk.Toplevel):
    """Select Matrix layers without touching their feature files."""

    def __init__(self, parent: tk.Misc, layers: List[LayerInfo], selected: set[str]):
        super().__init__(parent)
        self.withdraw()
        self.title("렌더링 레이어 선택")
        self.geometry("780x620")
        self.minsize(680, 460)
        self.transient(parent)

        self.layers = layers
        self.result: Optional[List[str]] = None
        self.vars = {
            layer.name: tk.BooleanVar(value=layer.name in selected)
            for layer in layers
        }
        self.type_var = tk.StringVar(value="전체")
        self.search_var = tk.StringVar()

        root = ttk.Frame(self, padding=12)
        root.pack(fill="both", expand=True)
        ttk.Label(
            root,
            text="Matrix 메타데이터만 조회합니다. 체크한 Layer의 feature만 실제 렌더링 시 로드/캐시됩니다.",
        ).pack(anchor="w", pady=(0, 10))

        bar = ttk.Frame(root)
        bar.pack(fill="x", pady=(0, 8))
        ttk.Label(bar, text="Type").pack(side="left")
        types = ["전체"] + sorted({layer.layer_type for layer in layers})
        type_combo = ttk.Combobox(
            bar,
            textvariable=self.type_var,
            values=types,
            state="readonly",
            width=18,
        )
        type_combo.pack(side="left", padx=(6, 14))
        type_combo.bind("<<ComboboxSelected>>", lambda _event: self.refresh())

        ttk.Label(bar, text="검색").pack(side="left")
        ttk.Entry(bar, textvariable=self.search_var).pack(
            side="left", fill="x", expand=True, padx=(6, 0)
        )
        self.search_var.trace_add("write", lambda *_args: self.refresh())

        tools = ttk.Frame(root)
        tools.pack(fill="x", pady=(0, 6))
        ttk.Button(
            tools,
            text="표시 목록 전체 선택",
            command=lambda: self.set_visible(True),
        ).pack(side="left")
        ttk.Button(
            tools,
            text="표시 목록 전체 해제",
            command=lambda: self.set_visible(False),
        ).pack(side="left", padx=6)

        head = ttk.Frame(root)
        head.pack(fill="x")
        headings = [
            ("사용", 6),
            ("Layer", 22),
            ("Type", 18),
            ("Context", 12),
            ("Side", 10),
            ("Polarity", 10),
        ]
        for column, (title, width) in enumerate(headings):
            ttk.Label(head, text=title, width=width, anchor="w").grid(
                row=0, column=column, sticky="w"
            )

        body = ttk.Frame(root)
        body.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(body, highlightthickness=0)
        scroll = ttk.Scrollbar(body, orient="vertical", command=self.canvas.yview)
        self.rows = ttk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.rows, anchor="nw", tags="rows")
        self.rows.bind(
            "<Configure>",
            lambda _event: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )
        self.canvas.bind(
            "<Configure>",
            lambda event: self.canvas.itemconfigure("rows", width=event.width),
        )
        self.canvas.configure(yscrollcommand=scroll.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

        buttons = ttk.Frame(root)
        buttons.pack(fill="x", pady=(10, 0))
        ttk.Button(buttons, text="취소", command=self.destroy).pack(side="right")
        ttk.Button(buttons, text="선택 레이어 로드", command=self.apply).pack(
            side="right", padx=6
        )

        self.refresh()
        self.protocol("WM_DELETE_WINDOW", self.destroy)

        # X11/xrdp can lose a modal window when grab_set happens before mapping.
        self.update_idletasks()
        self.deiconify()
        self.lift()
        try:
            self.wait_visibility()
            self.focus_force()
            self.grab_set()
        except tk.TclError:
            pass

    def filtered(self) -> List[LayerInfo]:
        rows = self.layers
        if self.type_var.get() != "전체":
            rows = [layer for layer in rows if layer.layer_type == self.type_var.get()]
        query = self.search_var.get().strip().lower()
        if not query:
            return rows
        return [
            layer
            for layer in rows
            if query
            in " ".join(
                (
                    layer.name,
                    layer.layer_type,
                    layer.context,
                    layer.side,
                    layer.polarity,
                )
            ).lower()
        ]

    def refresh(self) -> None:
        for child in self.rows.winfo_children():
            child.destroy()
        for row_index, layer in enumerate(self.filtered()):
            ttk.Checkbutton(self.rows, variable=self.vars[layer.name]).grid(
                row=row_index, column=0, sticky="w", pady=2
            )
            values = [
                (layer.name, 22),
                (layer.layer_type, 18),
                (layer.context, 12),
                (layer.side, 10),
                (layer.polarity, 10),
            ]
            for column, (value, width) in enumerate(values, start=1):
                ttk.Label(self.rows, text=value, width=width).grid(
                    row=row_index, column=column, sticky="w"
                )

    def set_visible(self, value: bool) -> None:
        for layer in self.filtered():
            self.vars[layer.name].set(value)

    def apply(self) -> None:
        selected = [
            layer.name for layer in self.layers if self.vars[layer.name].get()
        ]
        if not selected:
            messagebox.showwarning(
                "레이어 선택",
                "하나 이상의 Layer를 선택하세요.",
                parent=self,
            )
            return
        self.result = selected
        self.destroy()
