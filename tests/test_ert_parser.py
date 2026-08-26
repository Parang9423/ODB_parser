from pathlib import Path

from ert_parser import parse_ert


def test_parse_ert_footer(tmp_path: Path):
    path = tmp_path / "001.ERT"
    path.write_text(
        "0, 23.465, 471.321, 6, data\n"
        "0.000,0.000,392.860,485.465,56.430,493.705,\n"
        "MCP19013C00-005,L1-13-T-050,MS2640159-08-00,1,5.000,0.000\n"
        "2026,5,14,6,53,15,\n"
    )
    meta = parse_ert(path)
    assert meta.region_values == (0.0, 0.0, 392.86, 485.465)
    assert meta.guide_reference_candidate == (56.43, 493.705)
    assert meta.resolution_um_per_px == 5.0
    assert meta.item_code == "MCP19013C00-005"
    assert meta.recipe_layer == "L1-13-T-050"
    assert meta.timestamp == (2026, 5, 14, 6, 53, 15)
    assert meta.roi_size_mm_for_pixels(100, 100) == (0.5, 0.5)
