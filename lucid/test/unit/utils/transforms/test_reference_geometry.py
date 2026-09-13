"""Resize and centre-crop geometry agree with the reference pixel for pixel.

The evaluation preset every classifier ships with resizes the shorter side
and then centre-crops.  Both steps were one pixel off the reference: the
longer side was rounded where the reference truncates, and the crop offset
floored where the reference rounds half to even.  Either one moves the whole
crop by a column and still passes every shape check — on real photos it
changed ResNet-18's top-1 on one of six images, with the model itself exact.
"""

import pytest

import lucid
import lucid.utils.transforms as T
import lucid.utils.transforms.functional as F


@pytest.mark.parametrize(
    ("h", "w", "size", "expected"),
    [
        # 1300 * 256 / 876 = 379.9: the reference keeps 379.
        (876, 1300, 256, (256, 379)),
        (1300, 876, 256, (379, 256)),
        (480, 640, 256, (256, 341)),
        (256, 256, 224, (224, 224)),
    ],
)
def test_the_longer_side_is_truncated_like_the_reference(
    h: int, w: int, size: int, expected: tuple[int, int]
) -> None:
    assert F.resize_target(h, w, size) == expected


# (h, w, crop, top, left) — the reference's offsets, measured against it.
_CROPS = [
    (256, 379, 224, 16, 78),  # 155 / 2 = 77.5 rounds to 78, not 77
    (288, 343, 256, 16, 44),  # 87 / 2 = 43.5 rounds to 44
    (256, 341, 224, 16, 58),  # 117 / 2 = 58.5 rounds half to even: 58
    (257, 342, 224, 16, 59),
]


@pytest.mark.parametrize(("h", "w", "crop", "top", "left"), _CROPS)
def test_the_centre_crop_lands_where_the_reference_does(
    h: int, w: int, crop: int, top: int, left: int
) -> None:
    grid = lucid.arange(h * w).reshape(1, h, w)
    first = int(F.center_crop(grid, crop)[0, 0, 0].item())
    assert divmod(first, w) == (top, left)


@pytest.mark.parametrize(("h", "w", "crop", "top", "left"), _CROPS)
def test_boxes_are_cropped_at_the_same_offset_as_the_image(
    h: int, w: int, crop: int, top: int, left: int
) -> None:
    # The box and keypoint paths compute their own offsets; they must
    # move with the image or every box lands a column off.
    assert T.CenterCrop(crop, crop)._offsets((h, w)) == (top, left)


@pytest.mark.parametrize(
    "preset",
    [
        T.ImageClassification(crop_size=224, resize_size=256),
        T.Segmentation(crop_size=224, resize_size=256),
    ],
    ids=["classification", "segmentation"],
)
def test_the_evaluation_presets_resize_the_reference_way(
    preset: T.ImageClassification | T.Segmentation,
) -> None:
    # SmallestMaxSize keeps the Albumentations rule, which rounds: these
    # presets used it and resized this photo to 380 columns, not 379.
    img = lucid.rand(3, 876, 1300)
    got = preset(T.Image(img)).data
    mean = lucid.tensor([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = lucid.tensor([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    want = (F.center_crop(F.resize(img, 256), 224) - mean) / std
    assert float((got - want).abs().max().item()) < 1e-5
