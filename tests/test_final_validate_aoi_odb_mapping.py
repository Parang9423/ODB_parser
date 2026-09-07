from pathlib import Path

from aoi.coordinate_validation import ImageContext
from final_validate_aoi_odb_mapping import _ert_yx_frame, _translation, map_aoi_to_odb, spatial_pick


def _ctx(name: str, x: float, y: float) -> ImageContext:
    return ImageContext(
        image_path=Path(name),
        item_revision="ITEM-001",
        layer="L1",
        lot="LOT1",
        panel="1",
        x_mm=x,
        y_mm=y,
    )


def test_ert_yx_frame_swaps_region_axes():
    assert _ert_yx_frame([0.0, 0.0, 392.785, 485.338]) == [0.0, 0.0, 485.338, 392.785]


def test_translation_for_fixed_swap_and_center_alignment():
    frame = [0.0, 0.0, 485.338, 392.785]
    target = [-196.35, -242.625, 196.35, 242.625]
    tx, ty = _translation(frame, target)
    assert abs(tx - (-196.3925)) < 1e-9
    assert abs(ty - 242.669) < 1e-9


def test_map_aoi_to_odb_x_plus_y_minus():
    x, y = map_aoi_to_odb(45.138, 101.663, (-196.3925, 242.669))
    assert abs(x - (-94.7295)) < 1e-9
    assert abs(y - 197.531) < 1e-9


def test_spatial_pick_returns_requested_unique_contexts():
    contexts = [
        _ctx("a.jpg", 0.0, 0.0),
        _ctx("b.jpg", 100.0, 0.0),
        _ctx("c.jpg", 0.0, 100.0),
        _ctx("d.jpg", 100.0, 100.0),
        _ctx("e.jpg", 50.0, 50.0),
    ]
    picked = spatial_pick(contexts, 4)
    assert len(picked) == 4
    assert len({p.image_path for p in picked}) == 4
