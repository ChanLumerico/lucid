"""``interpolate`` modes, refusals and coordinate mapping, without the reference.

The values are matched against the reference in
``lucid/test/parity/nn/test_interpolate_parity.py``; this tier pins what
can be stated by hand: where a fractional scale factor puts the samples,
that ``nearest-exact`` is not ``nearest``, which arguments are refused and
with what, and when the size-only engine kernels may stand in for the
gathers.
"""

from typing import Any

import numpy as np
import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid.autograd import grad
from lucid.nn.functional import _interpolate


def _ramp(length: int) -> lucid.Tensor:
    return lucid.arange(length, dtype=lucid.float32).reshape(1, 1, length)


class TestCoordinateMapping:
    def test_a_fractional_scale_steps_by_its_reciprocal(self) -> None:
        # 5 * 1.7 -> 8 samples.  Stepping by 1/1.7 (not 5/8) puts sample 1
        # at (1.5 / 1.7) - 0.5 = 0.38235...
        out = F.interpolate(_ramp(5), scale_factor=1.7, mode="linear")
        assert out.shape == (1, 1, 8)
        assert float(out[0, 0, 1].item()) == pytest.approx(1.5 / 1.7 - 0.5, abs=1e-6)

    def test_recompute_steps_by_the_size_ratio(self) -> None:
        out = F.interpolate(
            _ramp(5), scale_factor=1.7, mode="linear", recompute_scale_factor=True
        )
        assert float(out[0, 0, 1].item()) == pytest.approx(1.5 * 5 / 8 - 0.5, abs=1e-6)

    def test_nearest_exact_is_not_nearest(self) -> None:
        near = F.interpolate(_ramp(5), size=9, mode="nearest").reshape(-1).tolist()
        exact = (
            F.interpolate(_ramp(5), size=9, mode="nearest-exact").reshape(-1).tolist()
        )
        assert near == [0, 0, 1, 1, 2, 2, 3, 3, 4]
        assert exact == [0, 0, 1, 1, 2, 3, 3, 4, 4]

    def test_area_is_adaptive_average_pooling(self) -> None:
        out = F.interpolate(_ramp(5), scale_factor=0.6, mode="area")
        assert out.reshape(-1).tolist() == pytest.approx([0.5, 2.0, 3.5])

    def test_nearest_keeps_an_integer_dtype(self) -> None:
        x = lucid.arange(6, dtype=lucid.int64).reshape(1, 1, 2, 3)
        out = F.interpolate(x, scale_factor=1.5, mode="nearest")
        assert out.dtype == lucid.int64
        assert out.shape == (1, 1, 3, 4)

    def test_same_size_returns_a_new_tensor(self) -> None:
        x = _ramp(5)
        out = F.interpolate(x, scale_factor=1.1, mode="linear")
        assert out is not x
        assert out.reshape(-1).tolist() == x.reshape(-1).tolist()


class TestEngineStandIn:
    """The size-only kernels step by ``in / out``: allowed only when that is
    where the reference puts every sample."""

    @pytest.mark.parametrize(
        ("mode", "spatial", "sizes", "scales", "corners", "expected"),
        [
            ("bilinear", [5, 7], [8, 11], [1.7, 1.7], False, False),
            ("bilinear", [5, 7], [8, 11], [1.7, 1.7], True, True),
            ("bilinear", [5, 7], [8, 11], [None, None], False, True),
            ("bilinear", [4, 6], [8, 12], [2.0, 2.0], False, True),
            ("nearest", [5, 7], [8, 11], [1.7, 1.7], False, False),
            ("nearest", [4, 6], [8, 12], [2.0, 2.0], False, True),
            ("nearest", [5], [8], [None], False, True),
        ],
    )
    def test_when_the_kernel_may_run(
        self,
        mode: str,
        spatial: list[int],
        sizes: list[int],
        scales: list[float | None],
        corners: bool,
        expected: bool,
    ) -> None:
        assert (
            _interpolate._engine_maps_like(mode, spatial, sizes, scales, corners)
            is expected
        )


class TestRefusals:
    @pytest.mark.parametrize(
        ("shape", "kw", "error", "match"),
        [
            ((1, 1, 4, 4), {"size": 2, "mode": "lanczos3"}, ValueError, "unknown mode"),
            ((1, 1, 4, 4), {"size": 2, "mode": "linear"}, ValueError, "spatial axes"),
            ((1, 1, 4), {"size": 2, "mode": "bilinear"}, ValueError, "spatial axes"),
            ((1, 1, 4, 4), {"size": 2, "scale_factor": 2.0}, ValueError, "not both"),
            ((1, 1, 4, 4), {}, ValueError, "either size"),
            ((1, 1, 4, 4), {"size": (2, 3, 4)}, ValueError, "entries"),
            ((1, 1, 4, 4), {"scale_factor": (2.0,)}, ValueError, "entries"),
            ((1, 1, 4, 4), {"scale_factor": 0.1}, ValueError, "positive"),
            (
                (1, 1, 4, 4),
                {"size": 2, "recompute_scale_factor": True},
                ValueError,
                "recompute_scale_factor",
            ),
            (
                (1, 1, 4, 4),
                {"size": 2, "mode": "nearest", "align_corners": False},
                ValueError,
                "align_corners",
            ),
            (
                (1, 1, 4, 4),
                {"size": 2, "mode": "area", "align_corners": True},
                ValueError,
                "align_corners",
            ),
            (
                (1, 1, 4, 4, 4),
                {"size": 2, "mode": "trilinear", "antialias": True},
                ValueError,
                "antialias",
            ),
            (
                (1, 1, 4, 4),
                {"size": 2, "mode": "nearest", "antialias": True},
                ValueError,
                "antialias",
            ),
        ],
    )
    def test_refused_with(
        self,
        shape: tuple[int, ...],
        kw: dict[str, Any],
        error: type[Exception],
        match: str,
    ) -> None:
        with pytest.raises(error, match=match):
            F.interpolate(lucid.zeros(*shape), **kw)

    def test_interpolating_an_integer_input_is_a_type_error(self) -> None:
        x = lucid.arange(16, dtype=lucid.int64).reshape(1, 1, 4, 4)
        with pytest.raises(TypeError, match="floating-point"):
            F.interpolate(x, size=2, mode="bilinear")


@pytest.mark.parametrize("mode", ["bilinear", "bicubic", "nearest-exact"])
def test_the_gathered_path_differentiates_twice(mode: str) -> None:
    # A fractional scale runs the gathers rather than a kernel; a gradient
    # penalty through it must still see a second derivative.
    rng = np.random.default_rng(0)
    x0 = rng.standard_normal((1, 2, 5, 7))
    r0 = lucid.tensor(rng.standard_normal((1, 2, 8, 11)))
    u = rng.standard_normal(x0.shape)
    e = rng.standard_normal(x0.shape)

    def loss(x: lucid.Tensor) -> lucid.Tensor:
        y = F.interpolate(x, scale_factor=1.7, mode=mode)
        return (r0 * y * y).sum()

    def projected_gradient(values: np.ndarray) -> float:
        x = lucid.tensor(values, requires_grad=True)
        (g,) = grad(loss(x), [x])
        return float((np.asarray(g.numpy()) * u).sum())

    x = lucid.tensor(x0, requires_grad=True)
    (g,) = grad(loss(x), [x], create_graph=True)
    (hvp,) = grad((g * lucid.tensor(u)).sum(), [x])
    h = 1e-6
    numeric = (projected_gradient(x0 + h * e) - projected_gradient(x0 - h * e)) / (
        2 * h
    )
    assert float((np.asarray(hvp.numpy()) * e).sum()) == pytest.approx(
        numeric, rel=1e-6
    )


class TestUpsampleForwardsEveryOption:
    def test_recompute_scale_factor_reaches_interpolate(self) -> None:
        x = lucid.randn(1, 2, 5, 7)
        up = nn.Upsample(scale_factor=1.7, mode="bilinear", recompute_scale_factor=True)
        expected = F.interpolate(
            x, scale_factor=1.7, mode="bilinear", recompute_scale_factor=True
        )
        assert float((up(x) - expected).abs().max().item()) == 0.0
        assert (
            float(
                (up(x) - F.interpolate(x, scale_factor=1.7, mode="bilinear"))
                .abs()
                .max()
                .item()
            )
            > 1e-3
        )
