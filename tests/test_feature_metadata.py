from odb.feature_metadata import find_top_level_record, iter_top_level_records, scan_feature_file_metadata


def test_top_level_record_count_matches_surface_consumption():
    records = [
        "$1 r100",
        "@1 .example_attribute",
        "P 1.0 2.0 1 P 0 ;7=ABC",
        "S P ;8=PLANE",
        "OB 0 0 I",
        "OS 1 0",
        "OS 1 1",
        "OE",
        "SE",
        "L 0 0 1 1 1 P ;9=NET",
    ]
    rows = list(iter_top_level_records(records))
    assert [row.command for row in rows] == ["@1", "P", "S", "L"]
    assert [row.record_index for row in rows] == [1, 2, 3, 4]
    assert find_top_level_record(records, 3).raw_record == "S P ;8=PLANE"


def test_inline_metadata_is_preserved_without_semantic_guessing():
    records = [
        "$1 r100",
        "P 1.0 2.0 1 P 0 ;7=ABC ;8=PAD?",
    ]
    row = find_top_level_record(records, 1)
    assert row is not None
    assert row.raw_record == "P 1.0 2.0 1 P 0 ;7=ABC ;8=PAD?"
    assert row.tail_tokens == (";7=ABC", ";8=PAD?")
    assert row.semicolon_suffix == ("7=ABC", "8=PAD?")


def test_metadata_inventory_reports_prefixes_unknowns_and_geometry_tails():
    records = [
        "$1 r100",
        "@1 .net",
        "&1 GND",
        "P 1.0 2.0 1 P 0 ;1=42",
        "X_VENDOR opaque value",
    ]
    payload = scan_feature_file_metadata(records)
    assert payload["metadata_prefix_counts"] == {"&": 1, "@": 1}
    assert payload["geometry_records_with_extra_fields"] == 1
    assert payload["geometry_extra_field_samples"][0]["semicolon_suffix"] == ["1=42"]
    assert payload["unknown_top_level_command_counts"]["X_VENDOR"] == 1
