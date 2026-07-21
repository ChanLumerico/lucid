"""``linalg.norm`` — device parity + the keepdim heap-overflow regression.

Two bugs, both found by the metal-vs-cpu backward sweep (2026-07-13) and both
invisible to the existing tests, which only covered 1-D vectors:

1. **Matrix semantics diverged by device.**  ``lucid.linalg.norm`` documents the
   *entrywise* definition (default = Frobenius for a matrix; the wrapper maps
   ``ord="fro"`` to 2.0 and has no matrix-norm dispatch — ``matrix_norm``
   implements spectral / nuclear itself).  The CPU stream did that, but the GPU
   stream forwarded to ``mlx::linalg::norm`` with no axis, which applies *matrix*
   semantics to a 2-D input: ``ord=2`` became the largest singular value and
   ``ord=1`` the max column sum.  The same call returned different numbers on
   different devices (3.892 vs 2.812 on a 4x5).

2. **CPU keepdim wrote out of bounds.**  ``norm_elementwise_loop`` mapped a
   surviving input axis to the *packed* output-axis counter even under keepdims,
   where the reduced axes are retained as size-1 and the ranks match.  For
   ``(2,3,4)`` reduced over dim 1 with ``keepdim=True`` that indexes up to 16 into
   an 8-element accumulator — a heap overflow that returned partly-zeroed data
   and then **segfaulted** (exit 139).  A 2-D input survived by luck (the
   mis-picked stride equals 1 there), which is why nothing caught it.
"""

import numpy as np
import pytest

import lucid
import lucid.linalg as LA
from lucid.test._fixtures.devices import metal_available

_SHAPES = [(6,), (4, 5), (2, 3, 4), (2, 3, 4, 5)]


def _entrywise(x: np.ndarray, ord_: float, axis: object, keepdims: bool) -> np.ndarray:
    """Reference: the vector p-norm applied over ``axis`` (never a matrix norm)."""
    if ord_ == 1:
        return np.abs(x).sum(axis=axis, keepdims=keepdims)
    return np.sqrt((x**2).sum(axis=axis, keepdims=keepdims))


@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("keepdim", [False, True])
def test_norm_full_reduction_is_entrywise(shape: tuple, keepdim: bool) -> None:
    """Default / ord=2 / 'fro' must all be the entrywise Frobenius value."""
    x = np.random.default_rng(len(shape)).standard_normal(shape).astype(np.float32)
    ref = _entrywise(x, 2.0, None, keepdim)
    got = LA.norm(lucid.tensor(x), keepdim=keepdim).numpy()
    np.testing.assert_allclose(got, ref, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(
        LA.norm(lucid.tensor(x), ord="fro", keepdim=keepdim).numpy(),
        ref,
        atol=1e-4,
        rtol=1e-4,
    )


@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("dim", [0, 1, [0, 1]])
@pytest.mark.parametrize("keepdim", [False, True])
@pytest.mark.parametrize("ord_", [1, 2])
def test_norm_cpu_matches_entrywise_reference(
    shape: tuple, dim: object, keepdim: bool, ord_: int
) -> None:
    """Guards the keepdim indexing (bug 2) across ranks — this segfaulted at 3-D."""
    dims = [dim] if isinstance(dim, int) else dim
    if max(dims) >= len(shape):
        pytest.skip("dim out of range for this rank")
    x = np.random.default_rng(7).standard_normal(shape).astype(np.float32)
    axis = dim if isinstance(dim, int) else tuple(dim)
    ref = _entrywise(x, float(ord_), axis, keepdim)
    got = LA.norm(lucid.tensor(x), ord=ord_, dim=dim, keepdim=keepdim).numpy()
    assert got.shape == ref.shape
    np.testing.assert_allclose(got, ref, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not metal_available(), reason="metal unavailable")
@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("kwargs", [{}, {"ord": 1}, {"ord": 2}, {"ord": "fro"}])
def test_norm_metal_matches_cpu(shape: tuple, kwargs: dict) -> None:
    """The same call must return the same number on both streams (bug 1)."""
    x = np.random.default_rng(3).standard_normal(shape).astype(np.float32)
    cpu = LA.norm(lucid.tensor(x, device="cpu"), **kwargs).numpy()
    mtl = LA.norm(lucid.tensor(x, device="metal"), **kwargs).numpy()
    np.testing.assert_allclose(mtl, cpu, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not metal_available(), reason="metal unavailable")
@pytest.mark.parametrize("dim", [0, 1, [0, 1]])
@pytest.mark.parametrize("keepdim", [False, True])
def test_norm_metal_matches_cpu_reduced(dim: object, keepdim: bool) -> None:
    """Axis-reduced norms (incl. the multi-axis path) must agree device-wise."""
    x = np.random.default_rng(5).standard_normal((2, 3, 4)).astype(np.float32)
    cpu = LA.norm(lucid.tensor(x, device="cpu"), dim=dim, keepdim=keepdim).numpy()
    mtl = LA.norm(lucid.tensor(x, device="metal"), dim=dim, keepdim=keepdim).numpy()
    assert cpu.shape == mtl.shape
    np.testing.assert_allclose(mtl, cpu, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not metal_available(), reason="metal unavailable")
def test_norm_backward_metal_matches_cpu() -> None:
    """The wrong forward also poisoned the gradient (dA = A/|A| uses the value)."""
    x = np.random.default_rng(0).standard_normal((4, 5)).astype(np.float32)

    def grad(device: str) -> np.ndarray:
        t = lucid.tensor(x, dtype=lucid.float32, device=device)
        t.requires_grad = True
        LA.norm(t).backward()
        return t.grad.numpy()

    np.testing.assert_allclose(grad("metal"), grad("cpu"), atol=1e-5, rtol=1e-4)
