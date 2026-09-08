from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


def find_reference(g_image: Path) -> Path | None:
    """Find the matching C_ reference beside a G_ AOI image."""
    name = g_image.name
    if not name.startswith("G_"):
        return None
    direct = g_image.with_name("C_" + name[2:])
    if direct.exists():
        return direct
    matches = sorted(g_image.parent.glob("C_" + g_image.stem[2:] + ".*"))
    return matches[0] if matches else None


def fmt_seconds(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(seconds, 60.0)
    if minutes < 60.0:
        return f"{int(minutes)}m {sec:.1f}s"
    hours, minutes = divmod(minutes, 60.0)
    return f"{int(hours)}h {int(minutes)}m {sec:.1f}s"


def score_crop(reference: Image.Image, candidate: Image.Image) -> dict[str, float]:
    """Return simple grayscale occupancy/edge similarity scores for diagnostics."""
    ref = np.asarray(ImageOps.grayscale(reference), dtype=np.float32) / 255.0
    cand_img = ImageOps.grayscale(candidate)
    if cand_img.size != reference.size:
        cand_img = cand_img.resize(reference.size, Image.Resampling.NEAREST)
    cand = np.asarray(cand_img, dtype=np.float32) / 255.0

    def edge_map(a: np.ndarray) -> np.ndarray:
        gx = np.abs(np.diff(a, axis=1, prepend=a[:, :1]))
        gy = np.abs(np.diff(a, axis=0, prepend=a[:1, :]))
        return np.maximum(gx, gy)

    def similarity(a: np.ndarray, b: np.ndarray) -> float:
        return float(max(0.0, min(1.0, 1.0 - np.mean(np.abs(a - b)))))

    occupancy = similarity(ref, cand)
    edge = similarity(edge_map(ref), edge_map(cand))
    score = math.sqrt(max(0.0, occupancy * edge))
    return {"score": score, "edge_score": edge, "occupancy_score": occupancy}


# Backward-compatible private names for existing validators.
_find_reference = find_reference
_fmt_seconds = fmt_seconds
_score_crop = score_crop
