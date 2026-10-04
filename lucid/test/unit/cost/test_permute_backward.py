"""Backpropagating through a CPU permute costs one strided copy, and is exact.

On the CPU every permute / transpose / swapaxes / mT is a view, so the copy
happens only in the backward, where the gradient is laid back into the
input's axis order (``backend/cpu/Shape.cpp``).  That copy rebuilt each
element's N-D coordinate with a division per axis — 4-6 ns an element on
one core — and splitting a small transformer's attention heads,
``(b, t, 3, h, d) -> (3, b, h, t, d)``, took 2.5 ms to backpropagate: half
of the whole CPU backward pass of the long-run GPT recipe, which made a
training step 1.3x the reference framework's instead of 0.75x.  It is now
the same merged-axis, run-at-a-time pack ``contiguous()`` uses.

The values are held bitwise to NumPy's transpose on every path the pack
takes (contiguous runs, a tiled 2-D transpose, a strided inner axis, size-1
and empty axes).  The cost is held relative to ``contiguous()`` of the same
view in the same process — the same bytes moved — so the bound does not
depend on how fast or busy the machine is.
"""

import time
from collections.abc import Callable

import numpy as np
import pytest

import lucid

CASES = [
    ((32, 64, 3, 4, 16), (2, 0, 3, 1, 4)),  # attention head split: 16-float runs
    ((32, 4, 64, 16), (0, 2, 1, 3)),  # head merge
    ((37, 45), (1, 0)),  # tiled 2-D transpose, ragged tiles
    ((5, 6, 7), (2, 0, 1)),  # last axis moves: strided inner loop
    ((3, 1, 4, 1, 2), (4, 3, 2, 1, 0)),  # size-1 axes dropped by merging
    ((2, 3, 4), (0, 1, 2)),  # identity: one memcpy
    ((4, 0, 3), (2, 1, 0)),  # empty
    ((7,), (0,)),
    ((), ()),
]


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.float16])
@pytest.mark.parametrize(("shape", "perm"), CASES, ids=[str(p) for _, p in CASES])
def test_the_gradient_is_the_inverse_permutation_bitwise(
    shape: tuple[int, ...], perm: tuple[int, ...], dtype: type
) -> None:
    rng = np.random.default_rng(0)
    a = np.asarray(rng.standard_normal(shape)).astype(dtype)
    out_shape = tuple(shape[p] for p in perm)
    g = np.asarray(rng.standard_normal(out_shape) * 100).astype(dtype)
    x = lucid.tensor(a)
    x.requires_grad_(True)
    # ``* g`` seeds the permute's backward with exactly ``g`` (``g * 1``).
    (x.permute(*perm) * lucid.tensor(g)).sum().backward()
    assert x.grad is not None
    want = np.transpose(g, np.argsort(perm)) if perm else g
    np.testing.assert_array_equal(x.grad.numpy(), want)


def _ms(fn: Callable[[], object]) -> float:
    start = time.perf_counter()
    fn()
    return (time.perf_counter() - start) * 1e3


def test_backward_through_a_permute_costs_about_one_copy() -> None:
    shape, perm = CASES[0]
    x = lucid.randn(*shape, requires_grad=True)

    def backward() -> None:
        x.grad = None
        x.permute(*perm).sum().backward()

    def copy() -> None:
        x.detach().permute(*perm).contiguous()

    for _ in range(3):
        backward()
        copy()
    # Interleaved, best of each, so load drifts onto both.  The backward is
    # a ones-fill, the permuting copy and the gradient hand-off: about
    # 1.5-2.5 copies.  The element walk it replaced was 15-20.
    back = once = float("inf")
    for _ in range(15):
        back = min(back, _ms(backward))
        once = min(once, _ms(copy))
    assert back < 6 * once, f"backward {back:.3f} ms vs one copy {once:.3f} ms"
