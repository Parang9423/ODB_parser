from PIL import Image

from odb_cam_renderer import CompositeLayer
from step_composite_renderer import render_selected_steps_composite


class FakeRenderer:
    def __init__(self):
        self.calls = []

    def render_hierarchy(self, root_step, layer, visible_steps, margin_px=0):
        step = next(iter(visible_steps))
        self.calls.append((root_step, layer, step))
        image = Image.new("L", (4, 1), 0)
        values = {
            ("pnl", "l1"): [255, 0, 0, 0],
            ("strip", "l1"): [0, 255, 0, 0],
            ("unit", "l1"): [0, 0, 255, 0],
        }.get((step, layer), [0, 0, 0, 0])
        image.putdata(values)
        return image


def test_checked_steps_are_all_merged():
    renderer = FakeRenderer()
    image = render_selected_steps_composite(
        renderer,
        "pnl",
        [CompositeLayer("l1", "REPLACE", 200)],
        {"pnl", "strip", "unit"},
    )
    assert list(image.getdata()) == [200, 200, 200, 0]
    assert {call[2] for call in renderer.calls} == {"pnl", "strip", "unit"}


def test_unchecked_step_is_not_saved():
    renderer = FakeRenderer()
    image = render_selected_steps_composite(
        renderer,
        "pnl",
        [CompositeLayer("l1", "REPLACE", 200)],
        {"pnl", "unit"},
    )
    assert list(image.getdata()) == [200, 0, 200, 0]
    assert {call[2] for call in renderer.calls} == {"pnl", "unit"}
