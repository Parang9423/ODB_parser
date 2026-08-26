"""Platform-specific UI compatibility helpers."""

import platform
import tkinter as tk
import tkinter.font as tkfont


def configure_linux_fonts(root: tk.Misc) -> None:
    if platform.system() != "Linux":
        return
    families = set(tkfont.families(root))
    candidates = (
        "Noto Sans CJK KR",
        "Noto Sans KR",
        "NanumGothic",
        "UnDotum",
        "DejaVu Sans",
    )
    family = next((name for name in candidates if name in families), None)
    if not family:
        return
    for name in (
        "TkDefaultFont",
        "TkTextFont",
        "TkMenuFont",
        "TkHeadingFont",
        "TkCaptionFont",
        "TkSmallCaptionFont",
        "TkIconFont",
        "TkTooltipFont",
    ):
        try:
            tkfont.nametofont(name).configure(family=family)
        except tk.TclError:
            pass
