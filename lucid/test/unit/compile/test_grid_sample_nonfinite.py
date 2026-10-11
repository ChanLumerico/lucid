"""Compiled ``grid_sample`` reads no pixel eager would not (LCD-294).

The compiled sampler clamps each coordinate and casts it to an int32 index
for a gather.  A NaN coordinate made that cast undefined — the gather could
read anywhere — and MPSGraph's clamp turned a NaN border-padded coordinate
into a finite one.  Eager answers fixed values there: Apple silicon converts
NaN to index 0 on both devices, so a NaN nearest sample is pixel 0 and a NaN
bilinear sample is NaN through its weights.  Zero padding also replaces an
outside corner by 0 rather than multiplying it by 0, which kept an infinite
pixel at the clamped position from turning the sample into NaN.

Each case compares the compiled forward with eager Metal, NaN for NaN.
"""

import math
import warnings

import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid.compile._core import fallback
from lucid.test.unit.compile._helpers import COMPILE_DEVICE

NAN, INF = float("nan"), float("inf")

#: (x, y) sample points: NaN on either or both axes, infinities, far outside.
POINTS = {
    "nan_x": (NAN, 0.0),
    "nan_y": (0.0, NAN),
    "nan_both": (NAN, NAN),
    "plus_inf": (INF, 0.0),
    "minus_inf": (-INF, 0.5),
    "far_outside": (5.0, -7.0),
    "edge": (1.0, -1.0),
    "inside": (0.2, 0.3),
}


class _Sample(nn.Module):
    def __init__(self, mode: str, padding: str, align: bool) -> None:
        super().__init__()
        self.mode, self.padding, self.align = mode, padding, align

    def forward(self, x: lucid.Tensor, grid: lucid.Tensor) -> lucid.Tensor:
        return F.grid_sample(
            x, grid, mode=self.mode, padding_mode=self.padding, align_corners=self.align
        )


def _same(a: list[float], b: list[float]) -> bool:
    return all(
        (math.isnan(p) and math.isnan(q)) or p == pytest.approx(q, rel=1e-6, abs=1e-6)
        for p, q in zip(a, b, strict=True)
    )


@pytest.mark.parametrize("align", [False, True])
@pytest.mark.parametrize("padding", ["zeros", "border"])
@pytest.mark.parametrize("mode", ["bilinear", "nearest"])
@pytest.mark.parametrize("image", ["finite", "infinite_corner"])
def test_compiled_grid_sample_matches_eager_off_the_grid(
    image: str, mode: str, padding: str, align: bool
) -> None:
    x = lucid.arange(16, dtype=lucid.float32).reshape(1, 1, 4, 4) + 1.0
    if image == "infinite_corner":
        # The pixel every clamped outside coordinate lands on first.
        x[0, 0, 0, 0] = INF
    grid = lucid.tensor([[list(POINTS.values())]])
    x, grid = x.to(COMPILE_DEVICE), grid.to(COMPILE_DEVICE)
    model = _Sample(mode, padding, align)
    compiled = lucid.compile(model)
    with warnings.catch_warnings():
        warnings.simplefilter("error", fallback.CompileFallbackWarning)
        compiled(x, grid)
        got = compiled(x, grid).reshape(-1).tolist()
    assert not compiled.cache_info()["eager_only"]
    want = model(x, grid).reshape(-1).tolist()
    assert _same(got, want), dict(zip(POINTS, zip(got, want, strict=True), strict=True))
