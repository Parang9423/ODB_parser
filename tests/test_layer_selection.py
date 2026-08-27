from pathlib import Path

import pytest

import render.roi as roi
from app_core import JobInfo, LayerInfo


def test_physical_layer_number_variants():
    assert roi._physical_layer_number("L1") == 1
    assert roi._physical_layer_number("L1-TU") == 1
    assert roi._physical_layer_number("L3_TD") == 3
    assert roi._physical_layer_number("L4BU") == 4


def test_drill_layer_span_variants():
    assert roi._drill_layer_span("DLD_1-2") == (1, 2)
    assert roi._drill_layer_span("TH_1-4") == (1, 4)
    assert roi._drill_layer_span("DRILL_2_3") == (2, 3)
    assert roi._drill_layer_span("THRU_4-2") == (2, 4)
    assert roi._drill_layer_span("GDRILL") is None


def test_select_layers_uses_signal_number_and_matching_drill_spans(monkeypatch):
    info = JobInfo(
        "job",
        ["pnl", "strip", "unit"],
        [
            LayerInfo("L1_TU", "SIGNAL"),
            LayerInfo("L2_TD", "SIGNAL"),
            LayerInfo("DLD_1-2", "DRILL"),
            LayerInfo("TH_1-4", "DRILL"),
            LayerInfo("DLD_2-3", "DRILL"),
            LayerInfo("TH_3-4", "DRILL"),
        ],
    )
    monkeypatch.setattr(roi, "inspect_job", lambda _job: info)
    selected = roi.select_roi_layers(Path("dummy"), "L1-TU-11-T-025")
    assert selected.physical_signal_layer == 1
    assert selected.signal_layer == "L1_TU"
    assert selected.drill_layers == ("DLD_1-2", "TH_1-4")
    assert selected.excluded_drill_layers == ("DLD_2-3", "TH_3-4")


def test_select_layers_never_falls_back_to_different_signal_number(monkeypatch):
    info = JobInfo(
        "job",
        ["pnl"],
        [LayerInfo("L2_TU", "SIGNAL"), LayerInfo("TH_1-4", "DRILL")],
    )
    monkeypatch.setattr(roi, "inspect_job", lambda _job: info)
    with pytest.raises(ValueError, match="L1"):
        roi.select_roi_layers(Path("dummy"), "L1-TU-11-T-025")
