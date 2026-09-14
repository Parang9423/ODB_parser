import pytest

pytest.importorskip("shapely")
from shapely.geometry import box

from odb.effective_geometry import EffectiveLayerGeometry
from odb.feature_query import ODBVectorFeature


def feature(feature_id, geometry, polarity="P"):
    return ODBVectorFeature(
        feature_id=feature_id,
        layer="l1_tu",
        step="unit",
        depth=2,
        primitive_type="S",
        symbol=None,
        polarity=polarity,
        geometry=geometry,
    )


def test_positive_then_negative_subtracts_material():
    defect = box(0, 0, 10, 10)
    features = [
        feature("positive", box(0, 0, 10, 10), "P"),
        feature("negative", box(0, 0, 5, 10), "N"),
    ]
    result = EffectiveLayerGeometry(features).intersect(defect)
    assert result.area_mm2 == pytest.approx(50.0)
    assert result.overlap_pct == pytest.approx(50.0)
    assert result.residual_pct == pytest.approx(50.0)
    assert result.positive_hit_count == 1
    assert result.negative_hit_count == 1


def test_later_positive_can_restore_previous_negative_region():
    defect = box(0, 0, 10, 10)
    features = [
        feature("positive-base", box(0, 0, 10, 10), "P"),
        feature("negative-hole", box(2, 2, 8, 8), "N"),
        feature("positive-restore", box(2, 2, 8, 8), "P"),
    ]
    result = EffectiveLayerGeometry(features).intersect(defect)
    assert result.area_mm2 == pytest.approx(100.0)
    assert result.overlap_pct == pytest.approx(100.0)
    assert result.applied_feature_ids == (
        "positive-base",
        "negative-hole",
        "positive-restore",
    )


def test_negative_before_positive_does_not_remove_future_material():
    defect = box(0, 0, 10, 10)
    features = [
        feature("negative-first", box(0, 0, 5, 10), "N"),
        feature("positive-later", box(0, 0, 10, 10), "P"),
    ]
    result = EffectiveLayerGeometry(features).intersect(defect)
    assert result.overlap_pct == pytest.approx(100.0)


def test_non_intersecting_features_are_not_applied():
    defect = box(0, 0, 10, 10)
    features = [feature("outside", box(20, 20, 30, 30), "P")]
    result = EffectiveLayerGeometry(features).intersect(defect)
    assert result.area_mm2 == pytest.approx(0.0)
    assert result.residual_pct == pytest.approx(100.0)
    assert result.applied_feature_ids == ()


def test_invalid_defect_is_rejected():
    from shapely.geometry import Point
    with pytest.raises(ValueError):
        EffectiveLayerGeometry([]).intersect(Point(0, 0))
