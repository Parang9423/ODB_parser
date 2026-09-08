from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageChops, ImageFilter, ImageOps


def find_reference(g_image: Path) -> Path | None:
    """Find the matching C_ reference beside a G_ AOI image."""
    name = g_image.name
    if not name.upper().startswith("G_"):
        return None
    suffix = name[2:]
    wanted_stem = "C_" + Path(suffix).stem
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
        candidate = g_image.with_name(wanted_stem + ext)
        if candidate.is_file():
            return candidate
    return None


def fmt_seconds(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    if seconds < 3600.0:
        return f"{seconds / 60.0:.1f}m"
    return f"{seconds / 3600.0:.2f}h"


def _binary(image: Image.Image, invert: bool = False) -> Image.Image:
    gray = ImageOps.autocontrast(ImageOps.grayscale(image))
    if invert:
        gray = ImageOps.invert(gray)
    return gray.point(lambda v: 255 if v >= 128 else 0, mode="L")


def _edge_mask(binary: Image.Image) -> Image.Image:
    return binary.filter(ImageFilter.FIND_EDGES).point(lambda v: 255 if v >= 32 else 0, mode="L")


def _count_on(binary: Image.Image) -> int:
    return int(binary.histogram()[255])


def _occupancy(binary: Image.Image) -> float:
    return _count_on(binary) / float(max(1, binary.width * binary.height))


def _tolerant_edge_dice(a: Image.Image, b: Image.Image) -> float:
    na, nb = _count_on(a), _count_on(b)
    if na + nb == 0:
        return 0.0
    da = a.filter(ImageFilter.MaxFilter(3))
    db = b.filter(ImageFilter.MaxFilter(3))
    overlap_a = _count_on(ImageChops.multiply(a, db))
    overlap_b = _count_on(ImageChops.multiply(b, da))
    return min(1.0, (overlap_a + overlap_b) / float(na + nb))


def score_crop_pair(cam_crop: Image.Image, reference: Image.Image) -> tuple[float, dict]:
    """Legacy-compatible CAM/reference score using Pillow only."""
    if cam_crop.size != reference.size:
        cam_crop = cam_crop.resize(reference.size, Image.Resampling.BILINEAR)

    cam_bin = _binary(cam_crop)
    cam_edge = _edge_mask(cam_bin)
    cam_occ = _occupancy(cam_bin)
    best: tuple[float, dict] = (-1.0, {})

    for mode, invert in (("normal", False), ("inverted", True)):
        ref_bin = _binary(reference, invert)
        ref_edge = _edge_mask(ref_bin)
        ref_occ = _occupancy(ref_bin)
        occupancy_score = max(0.0, 1.0 - abs(cam_occ - ref_occ))
        edge_score = _tolerant_edge_dice(cam_edge, ref_edge)
        score = 0.35 * occupancy_score + 0.65 * edge_score

        cam_solid = cam_occ <= 0.005 or cam_occ >= 0.995
        ref_solid = ref_occ <= 0.005 or ref_occ >= 0.995
        rejected = bool(cam_solid and not ref_solid)
        if rejected:
            score *= 0.02

        detail = {
            "reference_mode": mode,
            "occupancy_score": round(occupancy_score, 6),
            "edge_score": round(edge_score, 6),
            "cam_occupancy": round(cam_occ, 6),
            "reference_occupancy": round(ref_occ, 6),
            "cam_solid_rejected": rejected,
        }
        if score > best[0]:
            best = (float(score), detail)

    return best


def score_crop(reference: Image.Image, candidate: Image.Image) -> dict[str, float | str | bool]:
    """Dictionary wrapper used by the final fixed-coordinate validator."""
    score, detail = score_crop_pair(candidate, reference)
    return {"score": float(score), **detail}


_find_reference = find_reference
_fmt_seconds = fmt_seconds
_score_crop = score_crop_pair
