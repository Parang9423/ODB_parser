import json

import pytest

from aoi.contour_mapping import AoiOdbTransform
from aoi.seg_result import (
    detection_to_odb_polygon,
    parse_aoi_image_filename,
    seg_result_from_mapping,
)


SAMPLE = {
    "image": "AS2640050-01-00_18_L5-BD-11-B-025_61.993_361.640_54_SYSTEM.JPG",
    "width": 100,
    "height": 100,
    "detections": [
        {
            "class_id": 2,
            "class_name": "Dimple",
            "confidence": 0.6813163161277771,
            "bbox": [44.46424102783203, 43.69666290283203, 56.07617950439453, 55.76841354370117],
            "mask_contour": [
                [47, 44], [45, 46], [45, 47], [44, 48], [44, 51], [45, 52],
                [45, 53], [47, 55], [52, 55], [55, 52], [55, 46], [53, 44],
            ],
        }
    ],
}


def test_parse_uploaded_seg_result_shape():
    result = seg_result_from_mapping(SAMPLE)
    assert result.image.endswith("_54_SYSTEM.JPG")
    assert (result.width, result.height) == (100, 100)
    assert len(result.detections) == 1
    detection = result.detections[0]
    assert detection.class_id == 2
    assert detection.class_name == "Dimple"
    assert detection.confidence == pytest.approx(0.6813163161277771)
    assert len(detection.mask_contour) == 12
    assert detection.mask_contour[0] == (47.0, 44.0)


def test_parse_known_filename_y_then_x_and_separate_aoi_class():
    info = parse_aoi_image_filename(SAMPLE["image"])
    assert info.aoi_y_mm == pytest.approx(61.993)
    assert info.aoi_x_mm == pytest.approx(361.640)
    assert info.aoi_defect_class_id == 54


def test_crop_geometry_comes_from_json_size_and_external_resolution():
    result = seg_result_from_mapping(SAMPLE)
    geometry = result.crop_geometry(2.5)
    assert geometry.width_px == 100
    assert geometry.height_px == 100
    assert geometry.resolution_um_per_px_x == pytest.approx(2.5)
    assert geometry.center_px == pytest.approx((49.5, 49.5))


def test_real_mask_contour_converts_through_existing_odb_mapping():
    pytest.importorskip("shapely")
    result = seg_result_from_mapping(SAMPLE)
    info = parse_aoi_image_filename(result.image)
    transform = AoiOdbTransform(tx_mm=-196.3925, ty_mm=242.6690)
    polygon = detection_to_odb_polygon(
        result,
        result.detections[0],
        image_center_aoi_mm=(info.aoi_x_mm, info.aoi_y_mm),
        resolution_um_per_px=2.5,
        transform=transform,
    )
    assert polygon.area > 0
    # The contour lies around the 100x100 crop center, so its centroid should be
    # very close to the mapped image-center coordinate.
    center_odb_x = info.aoi_y_mm + transform.tx_mm
    center_odb_y = -info.aoi_x_mm + transform.ty_mm
    assert polygon.centroid.x == pytest.approx(center_odb_x, abs=0.01)
    assert polygon.centroid.y == pytest.approx(center_odb_y, abs=0.01)


def test_multiple_detections_are_preserved_as_independent_instances():
    data = dict(SAMPLE)
    second = dict(SAMPLE["detections"][0])
    second["class_id"] = 7
    second["class_name"] = "Scratch"
    data["detections"] = [SAMPLE["detections"][0], second]
    result = seg_result_from_mapping(data)
    assert [d.class_name for d in result.detections] == ["Dimple", "Scratch"]


def test_invalid_contour_is_rejected():
    data = dict(SAMPLE)
    bad = dict(SAMPLE["detections"][0])
    bad["mask_contour"] = [[1, 1], [2, 2]]
    data["detections"] = [bad]
    with pytest.raises(ValueError, match="mask_contour"):
        seg_result_from_mapping(data)


def test_filename_without_known_coordinate_suffix_is_rejected():
    with pytest.raises(ValueError):
        parse_aoi_image_filename("plain_image.jpg")
