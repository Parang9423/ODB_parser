from pathlib import Path

import pytest

from render.roi import _bbox_intersects, roi_bounds_in, select_roi_layers


def _write_matrix(job: Path, body: str) -> None:
    (job / "steps" / "pnl").mkdir(parents=True)
    (job / "matrix").mkdir(parents=True)
    (job / "matrix" / "matrix").write_text(body, encoding="utf-8")


def test_roi_bounds_are_centered_and_match_resolution():
    bounds = roi_bounds_in(100.0, 200.0, 5.0, 100, 100)
    xmin, ymin, xmax, ymax = bounds
    assert abs(((xmin + xmax) / 2.0) * 25.4 - 100.0) < 1e-9
    assert abs(((ymin + ymax) / 2.0) * 25.4 - 200.0) < 1e-9
    assert abs((xmax - xmin) * 25.4 - 0.5) < 1e-9
    assert abs((ymax - ymin) * 25.4 - 0.5) < 1e-9


def test_select_roi_layers_maps_recipe_l1_and_drill(tmp_path: Path):
    job = tmp_path / "job"
    _write_matrix(job, """
LAYER {
NAME=L1
TYPE=SIGNAL
CONTEXT=BOARD
SIDE=TOP
POLARITY=POSITIVE
}
LAYER {
NAME=L2
TYPE=SIGNAL
CONTEXT=BOARD
SIDE=INTERNAL
POLARITY=POSITIVE
}
LAYER {
NAME=DLD_1-2
TYPE=DRILL
CONTEXT=BOARD
SIDE=NONE
POLARITY=POSITIVE
}
LAYER {
NAME=TH_1-4
TYPE=DRILL
CONTEXT=BOARD
SIDE=NONE
POLARITY=POSITIVE
}
LAYER {
NAME=DLD_2-3
TYPE=DRILL
CONTEXT=BOARD
SIDE=NONE
POLARITY=POSITIVE
}
""")
    selected = select_roi_layers(job, "L1-TU-11-T-025")
    assert selected.physical_signal_layer == 1
    assert selected.signal_layer == "L1"
    assert selected.drill_layers == ("DLD_1-2", "TH_1-4")
    assert selected.excluded_drill_layers == ("DLD_2-3",)


def test_select_roi_layers_never_falls_back_to_other_physical_layer(tmp_path: Path):
    job = tmp_path / "job"
    _write_matrix(job, """
LAYER {
NAME=L3_TD
TYPE=SIGNAL
CONTEXT=BOARD
SIDE=TOP
POLARITY=POSITIVE
}
LAYER {
NAME=L2_TU
TYPE=SIGNAL
CONTEXT=BOARD
SIDE=TOP
POLARITY=POSITIVE
}
""")
    with pytest.raises(ValueError, match="Refusing to fall back"):
        select_roi_layers(job, "L1-TU-11-T-025")


def test_select_roi_layers_uses_orientation_only_within_same_layer_number(tmp_path: Path):
    job = tmp_path / "job"
    _write_matrix(job, """
LAYER {
NAME=L1_TD
TYPE=SIGNAL
CONTEXT=BOARD
SIDE=TOP
POLARITY=POSITIVE
}
LAYER {
NAME=L1_TU
TYPE=SIGNAL
CONTEXT=BOARD
SIDE=TOP
POLARITY=POSITIVE
}
LAYER {
NAME=L2_TU
TYPE=SIGNAL
CONTEXT=BOARD
SIDE=TOP
POLARITY=POSITIVE
}
""")
    selected = select_roi_layers(job, "L1-TU-11-T-025")
    assert selected.signal_layer == "L1_TU"


def test_bbox_intersection_used_by_feature_diagnostics():
    roi = (0.0, 0.0, 1.0, 1.0)
    assert _bbox_intersects((0.9, 0.9, 1.2, 1.2), roi)
    assert not _bbox_intersects((1.1, 1.1, 2.0, 2.0), roi)
