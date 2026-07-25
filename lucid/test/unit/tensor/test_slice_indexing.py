"""Regression tests for basic slice indexing — negative steps and empty ranges.

Two bugs found on 2026-07-26 while running an end-to-end CNN training smoke
test; the horizontal flip in the augmentation step (``x[..., ::-1]``) was the
trigger.

1. **Every negative step raised.**  ``_select_slice`` re-derived the element
   count with a hand-rolled ceil-division that dropped the ``step`` term from
   the numerator.  The canonical ``x[::-1]`` over-counted by one, asked
   ``arange`` for n+1 indices and raised ``ShapeMismatch``.  Positive steps
   with ``|step| > 1`` were wrong too whenever the range did not divide
   evenly (they under-counted).  ``step == 1`` was unaffected — it takes an
   earlier fast path.

2. **Backwards ranges segfaulted.**  ``x[5:2]`` (start > stop, step 1) reached
   ``split_at(impl, [5, 2])`` with *descending* split points and killed the
   process with SIGSEGV.  ``x[2:2]`` survived only because the points were
   equal.  This is a plain empty slice, valid Python, and a memory-safety bug.

The sweep below is exhaustive over small lengths rather than hand-picked:
both bugs were sensitive to whether the range divided evenly, so spot checks
would have missed cases.
"""

import numpy as np
import pytest

import lucid

DEVICES = ["cpu", "metal"]


def _lucid_slice(arr, sl, device):
    return lucid.tensor(arr, device=device)[sl].numpy()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("length", [1, 2, 3, 5, 7, 8])
def test_slice_matches_numpy_exhaustively(device, length):
    """Every (start, stop, step) combination must match NumPy exactly."""
    rng = np.random.default_rng(0)
    arr = rng.standard_normal((length,)).astype(np.float32)
    steps = [-4, -3, -2, -1, 1, 2, 3, 4]
    bounds = [None, 0, 1, -1, -2, length - 1, length]
    for step in steps:
        for start in bounds:
            for stop in bounds:
                sl = slice(start, stop, step)
                ref = arr[sl]
                got = _lucid_slice(arr, sl, device)
                assert got.shape == ref.shape, f"{sl}: {got.shape} != {ref.shape}"
                if ref.size:
                    assert np.abs(got - ref).max() == 0.0, f"{sl}"


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("shape", [(5,), (4, 6), (2, 3, 8), (2, 3, 4, 4)])
def test_reverse_last_axis(device, shape):
    """``x[..., ::-1]`` — the horizontal-flip idiom.  Used to raise."""
    rng = np.random.default_rng(1)
    arr = rng.standard_normal(shape).astype(np.float32)
    got = lucid.tensor(arr, device=device)[..., ::-1].numpy()
    assert np.abs(got - arr[..., ::-1]).max() == 0.0


@pytest.mark.parametrize("device", DEVICES)
def test_reverse_each_axis_of_an_image_batch(device):
    """Flip along every axis of an (N, C, H, W) batch."""
    rng = np.random.default_rng(2)
    arr = rng.standard_normal((2, 3, 5, 7)).astype(np.float32)
    t = lucid.tensor(arr, device=device)
    assert np.abs(t[::-1].numpy() - arr[::-1]).max() == 0.0
    assert np.abs(t[:, ::-1].numpy() - arr[:, ::-1]).max() == 0.0
    assert np.abs(t[:, :, ::-1].numpy() - arr[:, :, ::-1]).max() == 0.0
    assert np.abs(t[:, :, :, ::-1].numpy() - arr[:, :, :, ::-1]).max() == 0.0


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "sl",
    [
        slice(5, 2, None),  # backwards range, step 1 — used to SIGSEGV
        slice(1, 0, None),
        slice(3, 1, None),
        slice(2, 2, None),  # equal bounds — survived before, must keep working
        slice(0, 0, None),
        slice(2, 5, -1),  # backwards range with a negative step
        slice(0, 4, -2),
    ],
)
def test_empty_slices_are_empty_not_a_crash(device, sl):
    rng = np.random.default_rng(3)
    arr = rng.standard_normal((6,)).astype(np.float32)
    got = _lucid_slice(arr, sl, device)
    assert got.shape == arr[sl].shape
    assert got.size == 0


@pytest.mark.parametrize("device", DEVICES)
def test_empty_slice_keeps_other_axes(device):
    rng = np.random.default_rng(4)
    arr = rng.standard_normal((3, 6, 4)).astype(np.float32)
    got = lucid.tensor(arr, device=device)[:, 5:2, :].numpy()
    assert got.shape == arr[:, 5:2, :].shape == (3, 0, 4)


@pytest.mark.parametrize("device", DEVICES)
def test_reversed_slice_is_differentiable(device):
    """A flip in an augmentation pipeline sits inside the graph."""
    rng = np.random.default_rng(5)
    arr = rng.standard_normal((4, 5)).astype(np.float32)
    x = lucid.tensor(arr, device=device)
    x.requires_grad = True
    w = rng.standard_normal((4, 5)).astype(np.float32)
    out = x[:, ::-1]
    (out * lucid.tensor(w, device=device)).sum().backward()
    assert x.grad is not None
    # d/dx of a reversal is the reversed upstream gradient.
    assert np.abs(x.grad.numpy() - w[:, ::-1]).max() == 0.0
