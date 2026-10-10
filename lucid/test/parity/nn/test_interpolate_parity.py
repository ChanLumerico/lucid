"""Every ``interpolate`` setting against the reference, values and gradients.

A scale factor that does not divide the input evenly used to be rounded
to an output size, and the sample positions then derived from the size,
so every mode read the wrong input samples — ``1.7`` on a 5-pixel axis
steps by ``1/1.7``, not ``5/8``.  ``nearest-exact`` computed ``nearest``,
and ``linear``, 1-D nearest and ``area`` were refused.

The matrix is the product the reference accepts: every mode at every
spatial rank it takes, sized by a whole, a fractional up and a
fractional down scale factor, a per-axis tuple and an explicit size,
with ``recompute_scale_factor`` unset and ``True``, both corner
conventions where they apply, and ``antialias`` at several downsampling
ratios — on every device, with the input gradient.
"""

from typing import Any

import numpy as np
import pytest

import lucid
import lucid.nn.functional as F
from lucid.test._helpers.compare import assert_close

_SHAPES = {1: (2, 3, 7), 2: (2, 3, 5, 7), 3: (1, 2, 4, 5, 6)}
_RANKS = {
    "nearest": (1, 2, 3),
    "nearest-exact": (1, 2, 3),
    "area": (1, 2, 3),
    "linear": (1,),
    "bilinear": (2,),
    "bicubic": (2,),
    "trilinear": (3,),
}
_INTERPOLATING = {"linear", "bilinear", "bicubic", "trilinear"}


def _sizings(rank: int) -> list[tuple[str, dict[str, Any]]]:
    per_axis = (1.7, 0.6, 2.5)[:rank]
    size = (9, 4, 3)[:rank]
    return [
        ("x2", {"scale_factor": 2.0}),
        ("x1.7", {"scale_factor": 1.7}),
        ("x0.6", {"scale_factor": 0.6}),
        ("tuple", {"scale_factor": per_axis}),
        ("x1.7-recompute", {"scale_factor": 1.7, "recompute_scale_factor": True}),
        ("x0.6-recompute", {"scale_factor": 0.6, "recompute_scale_factor": True}),
        ("size", {"size": size}),
    ]


def _cases() -> list[Any]:
    cases = []
    for mode, ranks in _RANKS.items():
        corner_options = (True, False) if mode in _INTERPOLATING else (None,)
        for rank in ranks:
            for tag, kw in _sizings(rank):
                for corners in corner_options:
                    ctag = {True: "corners", False: "edges", None: "-"}[corners]
                    cases.append(
                        pytest.param(
                            mode,
                            rank,
                            {**kw, "align_corners": corners},
                            id=f"{mode}-{rank}d-{tag}-{ctag}",
                        )
                    )
    return cases


_ANTIALIAS = [
    pytest.param(
        mode, kw, corners, id=f"{mode}-{tag}-{'corners' if corners else 'edges'}"
    )
    for mode in ("bilinear", "bicubic")
    for tag, kw in (
        ("x0.5", {"scale_factor": 0.5}),
        ("x0.3", {"scale_factor": 0.3}),
        ("x0.37-recompute", {"scale_factor": 0.37, "recompute_scale_factor": True}),
        ("tuple", {"scale_factor": (0.75, 0.21)}),
        ("size", {"size": (4, 3)}),
        ("x1.1-same-size", {"scale_factor": (1.1, 1.0)}),
        ("x1.7-up", {"scale_factor": 1.7}),
    )
    for corners in (True, False)
]


def _compare(
    ref: Any, device: str, shape: tuple[int, ...], kw: dict[str, Any], seed: int
) -> None:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(shape).astype(np.float32)
    lx = lucid.tensor(x, device=device, requires_grad=True)
    rx = ref.tensor(x, requires_grad=True)
    l_out = F.interpolate(lx, **kw)
    r_out = ref.nn.functional.interpolate(rx, **kw)
    assert tuple(l_out.shape) == tuple(r_out.shape)
    assert_close(l_out, r_out.detach(), atol=1e-5, msg="value")

    # A random cotangent, so no gradient term cancels against another.
    weight = rng.standard_normal(tuple(r_out.shape)).astype(np.float32)
    (l_out * lucid.tensor(weight, device=device)).sum().backward()
    (r_out * ref.tensor(weight)).sum().backward()
    assert lx.grad is not None
    assert_close(lx.grad, rx.grad, atol=1e-5, msg="input grad")


@pytest.mark.parity
class TestInterpolateMatrix:
    @pytest.mark.parametrize(("mode", "rank", "kw"), _cases())
    def test_values_and_gradient(
        self, ref: Any, device: str, mode: str, rank: int, kw: dict[str, Any]
    ) -> None:
        _compare(ref, device, _SHAPES[rank], {**kw, "mode": mode}, seed=rank)


@pytest.mark.parity
class TestInterpolateAntialias:
    @pytest.mark.parametrize(("mode", "kw", "align_corners"), _ANTIALIAS)
    def test_values_and_gradient(
        self,
        ref: Any,
        device: str,
        mode: str,
        kw: dict[str, Any],
        align_corners: bool,
    ) -> None:
        kw = {**kw, "mode": mode, "align_corners": align_corners, "antialias": True}
        _compare(ref, device, (2, 3, 11, 14), kw, seed=7)


@pytest.mark.parity
class TestUpsampleModules:
    @pytest.mark.parametrize(
        ("module", "kw"),
        [
            ("Upsample", {"scale_factor": 0.6, "mode": "bilinear"}),
            ("Upsample", {"scale_factor": 1.7, "mode": "nearest"}),
            (
                "Upsample",
                {
                    "scale_factor": 1.7,
                    "mode": "bicubic",
                    "recompute_scale_factor": True,
                },
            ),
            ("UpsamplingNearest2d", {"scale_factor": 1.7}),
            ("UpsamplingBilinear2d", {"scale_factor": 1.7}),
        ],
        ids=[
            "upsample-bilinear",
            "upsample-nearest",
            "upsample-recompute",
            "nearest2d",
            "bilinear2d",
        ],
    )
    def test_modules_delegate(
        self, ref: Any, device: str, module: str, kw: dict[str, Any]
    ) -> None:
        x = np.random.default_rng(0).standard_normal((2, 3, 5, 7)).astype(np.float32)
        l_out = getattr(lucid.nn, module)(**kw)(lucid.tensor(x, device=device))
        r_out = getattr(ref.nn, module)(**kw)(ref.tensor(x))
        assert_close(l_out, r_out, atol=1e-5)
