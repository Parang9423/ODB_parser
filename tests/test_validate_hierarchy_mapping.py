from validate_hierarchy_mapping import LEVELS, _mean


def test_hierarchy_levels_are_explicit():
    assert LEVELS["PNL_ONLY"] == ("pnl",)
    assert LEVELS["STRIP_ONLY"] == ("strip",)
    assert LEVELS["UNIT_ONLY"] == ("unit",)
    assert LEVELS["ALL"] == ("pnl", "strip", "unit")


def test_mean_handles_empty_and_values():
    assert _mean([]) == 0.0
    assert _mean([1.0, 0.0, 0.5]) == 0.5
