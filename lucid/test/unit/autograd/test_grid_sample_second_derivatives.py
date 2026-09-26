"""grid_sample differentiates twice, in the input and the grid.

A gradient penalty or a meta-learning step through a spatial transformer,
``remap`` or ``warp_affine`` stopped at the sampler: its backward had no
graph-mode form.  It is the kernel's adjoint written in ops now — a
weighted scatter-add for the input, the gradient's projection onto the
weights' slope for the grid — with the kernel's own coordinate arithmetic,
so every point picks the corner the forward picked.

Checked against central differences of the first gradient in float64, for
both arguments, with grid points kept off the integer lines where the
sampler has a kink.
"""

import numpy as np
import pytest

import lucid
import lucid.nn.functional as F
from lucid.autograd import grad
from lucid.test._fixtures.devices import metal_available

CASES = [
    ("bilinear zeros", "bilinear", "zeros", False, 1.2),
    ("bilinear zeros corners", "bilinear", "zeros", True, 1.2),
    ("bilinear border", "bilinear", "border", False, 1.4),
    ("bilinear border corners", "bilinear", "border", True, 1.4),
    ("nearest zeros", "nearest", "zeros", False, 0.9),
]


def _sample(
    x: lucid.Tensor, g: lucid.Tensor, mode: str, pad: str, align: bool
) -> lucid.Tensor:
    return F.grid_sample(x, g, mode=mode, padding_mode=pad, align_corners=align)


@pytest.mark.parametrize("first", [0, 1])
@pytest.mark.parametrize("name,mode,pad,align,reach", CASES, ids=[c[0] for c in CASES])
def test_the_second_derivative_matches_central_differences(
    first: int, name: str, mode: str, pad: str, align: bool, reach: float
) -> None:
    if mode == "nearest" and first == 1:
        pytest.skip("nearest sampling has no grid gradient")
    rng = np.random.default_rng(len(name) + first)
    x0 = rng.standard_normal((2, 3, 5, 6))
    g0 = rng.uniform(-reach, reach, (2, 4, 3, 2))
    r0 = rng.standard_normal((2, 3, 4, 3))
    base = [x0, g0]
    u = [rng.standard_normal(b.shape) for b in base]

    def loss(x: lucid.Tensor, g: lucid.Tensor) -> lucid.Tensor:
        y = _sample(x, g, mode, pad, align)
        return (lucid.tensor(r0) * y * y).sum()

    def projected_gradient(values: list[np.ndarray]) -> float:
        args = [lucid.tensor(v, requires_grad=True) for v in values]
        (d,) = grad(loss(*args), [args[first]])
        return float((np.asarray(d.numpy()) * u[first]).sum())

    args = [lucid.tensor(b, requires_grad=True) for b in base]
    (d,) = grad(loss(*args), [args[first]], create_graph=True)
    assert float((np.asarray(d.numpy()) * u[first]).sum()) == pytest.approx(
        projected_gradient(base), rel=1e-10, abs=1e-12
    )
    second = grad((d * lucid.tensor(u[first])).sum(), args, allow_unused=True)
    h = 1e-6
    for k, analytic in enumerate(second):
        if mode == "nearest" and k == 1:
            continue  # piecewise constant in the grid
        e = rng.standard_normal(base[k].shape)
        plus = [v + h * e if i == k else v for i, v in enumerate(base)]
        minus = [v - h * e if i == k else v for i, v in enumerate(base)]
        numeric = (projected_gradient(plus) - projected_gradient(minus)) / (2 * h)
        exact = (
            0.0 if analytic is None else float((np.asarray(analytic.numpy()) * e).sum())
        )
        assert exact == pytest.approx(numeric, rel=1e-5, abs=1e-6), (first, k)


@pytest.mark.skipif(not metal_available(), reason="metal unavailable")
def test_metal_differentiates_twice_as_the_cpu_does() -> None:
    rng = np.random.default_rng(7)
    x0 = rng.standard_normal((2, 3, 5, 6)).astype(np.float32)
    g0 = rng.uniform(-1.1, 1.1, (2, 4, 3, 2)).astype(np.float32)
    answers = {}
    for device in ("cpu", "metal"):
        x = lucid.tensor(x0, device=device, requires_grad=True)
        g = lucid.tensor(g0, device=device, requires_grad=True)
        y = _sample(x, g, "bilinear", "zeros", False)
        (dx,) = grad((y * y).sum(), [x], create_graph=True)
        by_x, by_g = grad((dx * dx).sum(), [x, g])
        answers[device] = (by_x.numpy(), by_g.numpy())
    for got, want in zip(answers["metal"], answers["cpu"]):
        np.testing.assert_allclose(got, want, rtol=1e-3, atol=1e-3)
