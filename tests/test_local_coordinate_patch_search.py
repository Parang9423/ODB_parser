from PIL import Image, ImageDraw

from search_local_coordinate_match import (
    _candidate_crop,
    _score_crop,
    _stage_resolution,
)


def test_candidate_crop_maps_positive_y_up_to_raster_up():
    patch = Image.new("L", (100, 100), 0)
    draw = ImageDraw.Draw(patch)
    draw.rectangle((45, 15, 54, 24), fill=255)

    crop = _candidate_crop(
        patch,
        patch_center_x_mm=0.0,
        patch_center_y_mm=0.0,
        candidate_x_mm=0.0,
        candidate_y_mm=3.0,
        resolution_um=100.0,
        crop_size=(10, 10),
    )
    assert crop is not None
    assert crop.getbbox() == (0, 0, 10, 10)


def test_solid_cam_is_penalized_against_structured_reference():
    reference = Image.new("L", (40, 40), 0)
    ImageDraw.Draw(reference).rectangle((10, 10, 29, 29), fill=255)
    matching = reference.copy()
    solid = Image.new("L", (40, 40), 255)

    matching_score, matching_detail = _score_crop(matching, reference)
    solid_score, solid_detail = _score_crop(solid, reference)

    assert matching_score > solid_score
    assert matching_detail["cam_solid_rejected"] is False
    assert solid_detail["cam_solid_rejected"] is True


def test_stage_resolution_gets_finer_near_native_scale():
    native = 2.5
    assert _stage_resolution(native, 2.0) == 20.0
    assert _stage_resolution(native, 0.25) == 5.0
    assert _stage_resolution(native, 0.02) == native
