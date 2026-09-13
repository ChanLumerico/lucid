"""Parity: ``lucid.utils.transforms.functional.resize`` vs the reference.

Shrinking antialiases on both sides by default: the same separable filter
(a triangle for bilinear, the a = -0.5 cubic for bicubic) stretched by the
downscale factor.  Covers exact and fractional factors and a resize that
shrinks one side while growing the other.
"""

from typing import Any

import numpy as np
import pytest

import lucid
import lucid.utils.transforms.functional as TF

_SIZES = [(256, 341), (240, 320), (100, 100), (400, 533), (256, 800)]


@pytest.mark.parity
@pytest.mark.parametrize("mode", ["bilinear", "bicubic"])
@pytest.mark.parametrize("size", _SIZES, ids=[f"{h}x{w}" for h, w in _SIZES])
def test_resize_matches_the_reference(
    ref: Any, mode: str, size: tuple[int, int]
) -> None:
    x = np.random.default_rng(0).random((1, 3, 480, 640), dtype=np.float32)
    got = TF.resize(lucid.from_numpy(x.copy()), size, interpolation=mode).numpy()
    want = ref.nn.functional.interpolate(
        ref.from_numpy(x.copy()),
        size=size,
        mode=mode,
        align_corners=False,
        antialias=True,
    ).numpy()
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-4)
