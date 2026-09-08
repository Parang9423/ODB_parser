import pytest

shapely = pytest.importorskip("shapely")
from shapely.geometry import LineString, Point, box

from odb.feature_query import DefectContourQuery, ODBVectorFeature


def feature(feature_id, geometry, primitive="P", layer="l1_tu"):
    return ODBVectorFeature(
        feature_id=feature_id,
        layer=layer,
        step="unit",
        depth=2,
        primitive_type=primitive,
        symbol="r100" if primitive == "P" else None,
        polarity="P",
        geometry=geometry,
    )


def test_exact_query_returns_only_positive_area_intersections():
    defect = box(0, 0, 10, 10)
    inside = feature("inside", box(2, 2, 4, 4))
    partial = feature("partial", box(8, 8, 12, 12))
    outside = feature("outside", box(20, 20, 21, 21))
    touching = feature("touching", box(10, 2, 12, 4))

    hits = DefectContourQuery([inside, partial, outside, touching]).query(defect)
    assert [h.feature.feature_id for h in hits] == ["inside", "partial"]
    assert hits[0].intersection_area == pytest.approx(4.0)
    assert hits[0].defect_overlap_pct == pytest.approx(4.0)
    assert hits[1].intersection_area == pytest.approx(4.0)


def test_multiple_features_can_intersect_one_defect():
    defect = box(0, 0, 10, 10)
    pad = feature("pad", Point(3, 5).buffer(2), "P")
    circuit = feature("line", LineString([(0, 7), (10, 7)]).buffer(0.5), "L")
    hits = DefectContourQuery([pad, circuit]).query(defect)
    assert {h.feature.primitive_type for h in hits} == {"P", "L"}
    assert all(h.intersection_area > 0 for h in hits)


def test_bbox_candidate_that_does_not_exactly_intersect_is_removed():
    # The two triangles' bounding boxes overlap, but their actual polygons do not.
    from shapely.geometry import Polygon
    defect = Polygon([(0, 0), (4, 0), (0, 4)])
    candidate = feature("bbox-only", Polygon([(3, 3), (4, 3), (3, 4)]))
    assert defect.bounds == (0.0, 0.0, 4.0, 4.0)
    assert candidate.bounds == (3.0, 3.0, 4.0, 4.0)
    assert DefectContourQuery([candidate]).query(defect) == []


def test_overlap_is_relative_to_defect_area():
    defect = box(0, 0, 4, 4)  # area 16
    hit = feature("half", box(0, 0, 2, 4))  # intersection area 8
    result = DefectContourQuery([hit]).query(defect)[0]
    assert result.intersection_area == pytest.approx(8.0)
    assert result.defect_overlap_pct == pytest.approx(50.0)


def test_invalid_defect_polygon_rejected():
    with pytest.raises(ValueError):
        DefectContourQuery([]).query(Point(0, 0))
