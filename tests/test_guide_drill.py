from pathlib import Path

from guide_drill import collect_guide_drill_candidates, nearest_candidate
from odb_cam_renderer import Transform


class FakeRenderer:
    def _step_dir(self, step):
        return Path("/") / step

    def _feature_data(self, path):
        if str(path).endswith("/pnl/layers/gdrill/features"):
            return {1: "r100"}, (
                "$1 r100",
                "P 1.0 2.0 1 P 0",
                "P 3.0 4.0 1 N 0",
            )
        return {}, ()

    def _repeats(self, step):
        return ()

    def _child_transform(self, *args):
        return Transform()


def test_collect_positive_round_guide_drill():
    renderer = FakeRenderer()
    rows = collect_guide_drill_candidates(renderer, "pnl", ["gdrill"], {"pnl"})
    assert len(rows) == 1
    row = rows[0]
    assert row.step == "pnl"
    assert row.layer == "gdrill"
    assert row.x_in == 1.0
    assert row.y_in == 2.0
    assert row.diameter_in > 0
    assert nearest_candidate(rows, 1.01, 2.01) == row
