"""``lp_pool{1,2,3}d(norm_type=inf)`` is max pooling of ``|x|``.

The power average ``(sum x ** p) ** (1 / p)`` cannot take ``p = inf``:
``x ** inf`` is ``inf`` or 0 and ``inf ** 0`` is 1, so a refactor that
dropped the ``inf`` branch made every window 1.0 and only the opt-in parity
suite noticed.  These cases need no reference framework, so they run in the
default suite: each window's ``max |x|`` is computed here, and the gradient
is ``sign(x)`` at that window's arg-max.
"""

import itertools

import numpy as np
import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F

_KERNEL = 2
_SIZE = 5


def _signed(*shape: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(shape).astype(np.float32)


def _windows(rank: int, ceil_mode: bool) -> list[tuple[slice, ...]]:
    """The spatial slices of each output cell, stride equal to the kernel."""
    count = -(-_SIZE // _KERNEL) if ceil_mode else _SIZE // _KERNEL
    axis = [slice(i * _KERNEL, min(i * _KERNEL + _KERNEL, _SIZE)) for i in range(count)]
    return list(itertools.product(axis, repeat=rank))


def _expected(x: np.ndarray, rank: int, ceil_mode: bool) -> tuple[np.ndarray, ...]:
    """Each window's ``max |x|``, and the gradient of the output's plain sum."""
    lead = x.shape[:-rank]
    count = -(-_SIZE // _KERNEL) if ceil_mode else _SIZE // _KERNEL
    flat = x.reshape(-1, *x.shape[-rank:])
    out = np.zeros((flat.shape[0], len(_windows(rank, ceil_mode))), dtype=x.dtype)
    grad = np.zeros_like(flat)
    for b in range(flat.shape[0]):
        for w, window in enumerate(_windows(rank, ceil_mode)):
            cell = np.abs(flat[b][window])
            out[b, w] = cell.max()
            at = np.unravel_index(int(cell.argmax()), cell.shape)
            src = tuple(s.start + i for s, i in zip(window, at, strict=True))
            grad[b][src] += np.sign(flat[b][src])
    return out.reshape(*lead, *(count,) * rank), grad.reshape(x.shape)


@pytest.mark.parametrize("rank", [1, 2, 3])
@pytest.mark.parametrize("batched", [True, False], ids=["batched", "unbatched"])
@pytest.mark.parametrize("ceil_mode", [False, True], ids=["floor", "ceil"])
def test_infinite_norm_is_max_pool_of_the_magnitude(
    device: str, rank: int, batched: bool, ceil_mode: bool
) -> None:
    x = _signed(*((2, 3) if batched else (3,)), *(_SIZE,) * rank)
    want, want_grad = _expected(x, rank, ceil_mode)
    xs = lucid.tensor(x, device=device, requires_grad=True)
    got = getattr(F, f"lp_pool{rank}d")(xs, float("inf"), _KERNEL, ceil_mode=ceil_mode)
    np.testing.assert_allclose(got.numpy(), want, rtol=1e-6, atol=0)
    got.sum().backward()
    assert xs.grad is not None
    np.testing.assert_allclose(xs.grad.numpy(), want_grad, rtol=1e-6, atol=0)
    module = getattr(nn, f"LPPool{rank}d")(float("inf"), _KERNEL, ceil_mode=ceil_mode)
    np.testing.assert_allclose(
        module(lucid.tensor(x, device=device)).numpy(), want, rtol=1e-6, atol=0
    )
