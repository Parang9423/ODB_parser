import pytest

from area_classifier import AREA_UNK, AreaRule, classify_area_from_intersections


RULES = [
    AreaRule("PAD", 10.0, 1),
    AreaRule("CIRCUIT", 20.0, 2),
    AreaRule("SPACE", 30.0, 3),
]


def test_priority_wins_after_threshold_pass():
    result = classify_area_from_intersections(100.0, {"PAD": 30.0, "CIRCUIT": 70.0}, RULES)
    assert result.final_area == "PAD"


def test_high_priority_area_below_threshold_is_rejected():
    result = classify_area_from_intersections(100.0, {"PAD": 8.0, "CIRCUIT": 92.0}, RULES)
    assert result.final_area == "CIRCUIT"
    pad = next(c for c in result.candidates if c.area == "PAD")
    assert pad.eligible is False


def test_no_threshold_pass_returns_area_unk():
    result = classify_area_from_intersections(100.0, {"PAD": 9.0, "CIRCUIT": 19.0, "SPACE": 29.0}, RULES)
    assert result.final_area == AREA_UNK
    assert result.reason == "no_area_passed_threshold"


def test_threshold_is_inclusive():
    result = classify_area_from_intersections(100.0, {"PAD": 10.0, "CIRCUIT": 90.0}, RULES)
    assert result.final_area == "PAD"


def test_same_priority_uses_larger_overlap_as_tie_breaker():
    rules = [AreaRule("PAD", 10.0, 1), AreaRule("CIRCUIT", 10.0, 1)]
    result = classify_area_from_intersections(100.0, {"PAD": 30.0, "CIRCUIT": 70.0}, rules)
    assert result.final_area == "CIRCUIT"


def test_zero_defect_area_returns_area_unk():
    result = classify_area_from_intersections(0.0, {"PAD": 1.0}, RULES)
    assert result.final_area == AREA_UNK
    assert result.reason == "invalid_defect_area"


def test_duplicate_rules_are_rejected():
    with pytest.raises(ValueError):
        classify_area_from_intersections(100.0, {"PAD": 50.0}, [AreaRule("PAD", 10, 1), AreaRule("PAD", 20, 2)])
