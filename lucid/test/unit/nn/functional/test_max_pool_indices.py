"""``return_indices`` on max pooling, checked without the reference.

Every max pool used to raise ``NotImplementedError`` for it, which left
``MaxUnpool`` — built to consume exactly those indices — fed only by hand.
The parity tier compares the numbers against the reference.
"""

from typing import Any

import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F


def test_each_index_points_at_the_value_it_reports() -> None:
    x = lucid.randn(2, 3, 9, 11)
    out, idx = F.max_pool2d(x, 3, 2, 1, return_indices=True)
    picked = lucid.gather(x.reshape(2, 3, -1), idx.reshape(2, 3, -1), dim=2)
    assert float((picked.reshape(*out.shape) - out).abs().max().item()) == 0.0


@pytest.mark.parametrize(
    ("cls", "shape", "kwargs"),
    [
        (nn.MaxPool1d, (2, 3, 16), {"kernel_size": 2}),
        (nn.MaxPool2d, (2, 3, 8, 8), {"kernel_size": 2}),
        (nn.MaxPool3d, (1, 2, 4, 4, 4), {"kernel_size": 2}),
        (nn.AdaptiveMaxPool1d, (2, 3, 16), {"output_size": 4}),
        (nn.AdaptiveMaxPool2d, (2, 3, 8, 8), {"output_size": (3, 3)}),
        (nn.AdaptiveMaxPool3d, (1, 2, 4, 6, 8), {"output_size": (2, 3, 4)}),
    ],
    ids=lambda v: v.__name__ if isinstance(v, type) else None,
)
def test_the_modules_return_values_and_indices(
    cls: Any, shape: tuple[int, ...], kwargs: dict[str, Any]
) -> None:
    result = cls(return_indices=True, **kwargs)(lucid.randn(*shape))
    assert isinstance(result, tuple)
    out, idx = result
    assert tuple(idx.shape) == tuple(out.shape)
    assert idx.dtype == lucid.int64


def test_max_pool3d_honours_ceil_mode() -> None:
    # The module used to drop ceil_mode (and dilation) on the way to the
    # functional op, so this pooled 5 -> 2 where it asked for 5 -> 3.
    x = lucid.randn(1, 1, 5, 5, 5)
    assert tuple(nn.MaxPool3d(2, ceil_mode=True)(x).shape) == (1, 1, 3, 3, 3)
    assert tuple(nn.MaxPool3d(2)(x).shape) == (1, 1, 2, 2, 2)


def test_without_the_flag_a_single_tensor_comes_back() -> None:
    assert isinstance(F.max_pool2d(lucid.randn(1, 1, 4, 4), 2), lucid.Tensor)


def test_a_maximum_two_windows_share_is_unpooled_once() -> None:
    # Overlapping windows can pick the same element: here 9 is the maximum
    # of both [1, 9] and [9, 2], so its index appears twice.  Unpooling
    # used to add the copies and put 18 where the input had 9.
    x = lucid.tensor([[[1.0, 9.0, 2.0, 3.0]]])
    out, idx = F.max_pool1d(x, 2, 1, return_indices=True)
    assert idx.numpy().tolist() == [[[1, 1, 3]]]
    back = F.max_unpool1d(out, idx, 2, 1, output_size=(4,))
    assert back.numpy().tolist() == [[[0.0, 9.0, 0.0, 3.0]]]


@pytest.mark.parametrize("n", [1, 2, 3])
def test_unpooling_defaults_to_the_size_pooling_started_from(n: int) -> None:
    # output_size used to be required, with a message blaming the missing
    # indices; the default follows from kernel, stride and padding alone.
    shape = (1, 2) + (8,) * n
    out, idx = getattr(F, f"max_pool{n}d")(lucid.randn(*shape), 2, return_indices=True)
    assert tuple(getattr(F, f"max_unpool{n}d")(out, idx, 2).shape) == shape
