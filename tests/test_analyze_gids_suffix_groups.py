from pathlib import Path
from analyze_gids_suffix_groups import parse_name, _stats


def test_parse_gids_name_uses_y_x_suffix_order():
    got=parse_name(Path("G_392.540_9.095_20.jpg"))
    assert got == {"y_mm":392.54,"x_mm":9.095,"suffix":"20"}


def test_parse_rejects_non_gids_name():
    assert parse_name(Path("foo.jpg")) is None


def test_stats():
    assert _stats([1.0,3.0]) == {"min":1.0,"max":3.0,"mean":2.0,"span":2.0}
