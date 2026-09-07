import pytest

from validate_local_context import context_pixels


def test_context_pixels_at_2_5_um():
    assert context_pixels(5.0, 2.5) == 2000
    assert context_pixels(10.0, 2.5) == 4000


def test_context_pixels_rejects_nonpositive_values():
    with pytest.raises(ValueError):
        context_pixels(0.0, 2.5)
    with pytest.raises(ValueError):
        context_pixels(5.0, 0.0)
