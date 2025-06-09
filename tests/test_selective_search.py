import pytest

# Skip if numpy is not installed since lucid depends on it
np = pytest.importorskip("numpy")

import lucid
from lucid.models.conv.rcnn import _SelectiveSearch


def test_selective_search_runs():
    # Construct a simple 4x4 image with two halves of different colors
    img = lucid.zeros((3, 4, 4), dtype=lucid.Int32)
    img[:, 2:, :] = 255

    ss = _SelectiveSearch(scales=(50,), min_size=1, max_boxes=10, connectivity=4)
    boxes = ss(img)

    assert boxes.ndim == 2
    assert boxes.shape[1] == 4
    assert boxes.dtype == lucid.Int32
    assert (boxes[:, 0] >= 0).all() and (boxes[:, 1] >= 0).all()
    assert (boxes[:, 2] < 4).all() and (boxes[:, 3] < 4).all()
