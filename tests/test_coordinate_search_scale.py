from PIL import Image

from search_coordinate_mapping import _center_crop_or_pad
from render.surface_validation import _point_in_polygon, _polygon_rect_intersects


def test_center_crop_keeps_native_pixel_scale():
    image = Image.new("L", (200, 200), 0)
    image.putpixel((100, 100), 255)
    crop = _center_crop_or_pad(image, 100, 100)
    assert crop.size == (100, 100)
    # 200x200 -> 100x100 is a crop, not a resize. The centre pixel remains one pixel.
    assert crop.getpixel((50, 50)) == 255
    assert sum(1 for value in crop.getdata() if value) == 1


def test_surface_geometry_helpers_distinguish_polygon_from_bbox():
    triangle = [(0.0, 0.0), (10.0, 0.0), (0.0, 10.0)]
    assert _point_in_polygon((1.0, 1.0), triangle)
    assert not _point_in_polygon((9.0, 9.0), triangle)
    # This rectangle overlaps the triangle's bounding box but not the triangle itself.
    assert not _polygon_rect_intersects(triangle, (8.0, 8.0, 9.0, 9.0))
    assert _polygon_rect_intersects(triangle, (0.5, 0.5, 1.5, 1.5))
