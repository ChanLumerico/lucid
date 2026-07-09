"""Pooling backward regression — the family had no numerical backward test, which
let a silent metal bug hide: the non-overlapping max_pool2d fast path
reconstructed dx at (Oh*Kh, Ow*Kw) and mis-placed gradients whenever the input
was not an exact multiple of the kernel (odd feature maps).  Fixed 2026-07-10.

Oracle = the CPU (Accelerate) backend, which uses a straightforward scatter and
is verified correct.  (Finite-difference gradcheck is unreliable for max: the
gradient is piecewise-constant, so an FD step that crosses an argmax boundary
disagrees with the true subgradient — it "fails" on the correct CPU path too.)
"""

import numpy as np
import pytest

import lucid
import lucid.nn.functional as F
from lucid.test._fixtures.devices import metal_available


def _grad(fn, shape, device, seed):
    a = np.random.default_rng(seed).standard_normal(shape).astype(np.float32)
    x = lucid.tensor(a, dtype=lucid.float32, device=device)
    x.requires_grad = True
    fn(x).sum().backward()
    return x.grad.numpy()


# Non-divisible sizes are the regression: (5,5)/(7,7)/(15,15) with kernel 2 were
# silently wrong on metal; even sizes happened to be fine.
@pytest.mark.parametrize("hw", [(5, 5), (7, 7), (8, 8), (15, 15), (16, 16), (6, 9), (9, 6)])
def test_max_pool2d_backward_metal_matches_cpu(hw: tuple) -> None:
    if not metal_available():
        pytest.skip("metal backend unavailable")
    fn = lambda t: F.max_pool2d(t, kernel_size=2, stride=2)
    seed = hw[0] * 31 + hw[1]
    np.testing.assert_allclose(
        _grad(fn, (2, 3, *hw), "metal", seed),
        _grad(fn, (2, 3, *hw), "cpu", seed),
        atol=1e-5,
    )


@pytest.mark.parametrize(
    "k,s,hw", [(3, 2, (9, 9)), (3, 3, (7, 7)), (2, 2, (10, 7)), (3, 1, (6, 6))]
)
def test_max_pool2d_variants_metal_matches_cpu(k: int, s: int, hw: tuple) -> None:
    """Overlapping (stride != kernel, scatter path) + more non-divisible cases."""
    if not metal_available():
        pytest.skip("metal backend unavailable")
    fn = lambda t: F.max_pool2d(t, kernel_size=k, stride=s)
    np.testing.assert_allclose(
        _grad(fn, (2, 3, *hw), "metal", 7),
        _grad(fn, (2, 3, *hw), "cpu", 7),
        atol=1e-5,
    )
