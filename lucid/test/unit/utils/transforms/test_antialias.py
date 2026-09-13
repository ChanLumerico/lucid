"""Antialiased shrinking in :func:`lucid.utils.transforms.functional.resize`.

The reference low-pass filters before it shrinks an image.  Without that,
every classifier preset fed a real photo a different image from the one
its checkpoint was evaluated on, and ResNet-18 changed its top-1 on one
photo of six.  These pin the behaviour without the reference; the parity
tier compares the numbers against it.
"""

import pytest

import lucid
import lucid.utils.transforms as T
import lucid.utils.transforms.functional as F


def _checkerboard(h: int, w: int) -> lucid.Tensor:
    rows = [[float((i + j) % 2) for j in range(w)] for i in range(h)]
    return lucid.tensor(rows).reshape(1, h, w)


@pytest.mark.parametrize("mode", ["bilinear", "bicubic"])
def test_shrinking_a_fine_pattern_averages_it_instead_of_sampling_it(
    mode: str,
) -> None:
    # A one-pixel checkerboard has nothing a smaller image can show:
    # filtered it is flat grey, sampled it is whatever pixels the grid hit.
    # 64 -> 20 rather than 64 -> 16, because at an exact 4x every sample
    # lands midway between two pixels and sampling also reads 0.5.
    img = _checkerboard(64, 64)
    smooth = F.resize(img, (20, 20), interpolation=mode)
    sampled = F.resize(img, (20, 20), interpolation=mode, antialias=False)
    assert float((smooth - 0.5).abs().max().item()) < 0.05
    assert float((sampled - 0.5).abs().max().item()) > 0.2


@pytest.mark.parametrize("mode", ["bilinear", "bicubic"])
def test_a_flat_image_stays_flat(mode: str) -> None:
    # Each output pixel is a weighted average whose weights sum to one.
    img = lucid.ones(3, 50, 70) * 0.3
    out = F.resize(img, (13, 29), interpolation=mode)
    assert float((out - 0.3).abs().max().item()) < 1e-5


def test_growing_an_image_is_unchanged_by_the_flag() -> None:
    img = lucid.rand(3, 20, 30)
    filtered = F.resize(img, (40, 60))
    sampled = F.resize(img, (40, 60), antialias=False)
    assert float((filtered - sampled).abs().max().item()) == 0.0


def test_the_stretching_segmentation_preset_filters_too() -> None:
    # stretch=True reproduces processors that resize through PIL, which
    # always filters; the Albumentations Resize it used to reuse does not.
    img = lucid.rand(3, 200, 300)
    got = T.Segmentation(resize_size=64, stretch=True)(T.Image(img)).data
    mean = lucid.tensor([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = lucid.tensor([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    want = (F.resize(img, (64, 64)) - mean) / std
    assert float((got - want).abs().max().item()) < 1e-5


def test_the_albumentations_resizers_still_sample_without_filtering() -> None:
    # SmallestMaxSize documents the Albumentations (OpenCV) behaviour, which
    # does not low-pass filter; only the reference presets do.
    img = lucid.rand(3, 120, 160)
    got = T.SmallestMaxSize(40)(T.Image(img)).data
    want = F.resize(img, (40, 53), antialias=False)
    assert float((got - want).abs().max().item()) == 0.0
