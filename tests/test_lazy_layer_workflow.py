from app import default_spec
from app_core import LayerInfo


def test_signal_layer_default_is_white_replace():
    spec = default_spec(LayerInfo("L1", layer_type="SIGNAL"))
    assert (spec.operation, spec.gv) == ("REPLACE", 255)


def test_drill_layer_default_is_gray_replace():
    spec = default_spec(LayerInfo("DRILL_1-4", layer_type="DRILL"))
    assert (spec.operation, spec.gv) == ("REPLACE", 96)


def test_mixed_layer_default_is_additive_gray():
    spec = default_spec(LayerInfo("MIXED_A", layer_type="MIXED"))
    assert (spec.operation, spec.gv) == ("ADD", 220)


def test_solder_mask_default_is_additive_gray():
    spec = default_spec(LayerInfo("SM-L1", layer_type="SOLDER_MASK"))
    assert (spec.operation, spec.gv) == ("ADD", 160)
