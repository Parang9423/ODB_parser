from pathlib import Path

from PIL import Image

from odb_cam_renderer import CompositeLayer
from step_composite_renderer import build_render_diagnostics, render_selected_steps_composite


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


def _feature(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


class DiagnosticRenderer:
    def __init__(self, job: Path, dpi: float):
        self.job = job
        self.dpi_x = self.dpi_y = dpi

    def collect_instances(self, _root_step):
        class Item:
            def __init__(self, step): self.step = step
        return [Item("pnl"), Item("strip"), Item("unit"), Item("unit")]

    def render_hierarchy(self, _root_step, layer, visible_steps, margin_px=0):
        step = next(iter(visible_steps))
        image = Image.new("L", (4, 1), 0)
        if (self.job / "steps" / step / "layers" / layer.lower() / "features").exists():
            image.putpixel((0, 0), 255)
        return image


def test_render_diagnostics_reports_step_layer_presence(tmp_path: Path):
    job = tmp_path / "job"
    _feature(job / "steps" / "pnl" / "layers" / "l1" / "features", "P 0 0 0 P 0 0\n")
    _feature(job / "steps" / "strip" / "layers" / "l1" / "features", "L 0 0 1 1 0 P 0 0\n")
    renderer = DiagnosticRenderer(job, 1200.0)
    report = build_render_diagnostics(
        renderer,
        "pnl",
        [CompositeLayer("l1", "REPLACE", 255)],
        {"pnl", "strip", "unit"},
    )
    assert "PNL   : 1" in report
    assert "STRIP : 1" in report
    assert "UNIT  : 2" in report
    assert "STEP=PNL" in report and "file=YES" in report
    assert "STEP=UNIT" in report and "file=NO" in report
    assert "nonzero@72dpi=1" in report
