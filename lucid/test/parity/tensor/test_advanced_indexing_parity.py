"""Advanced indexing where the array indices are split, against the reference.

NumPy is the oracle for every expression in
``lucid/test/unit/tensor/test_advanced_indexing.py``; the split cases here
are the ones where the placement of the result is the whole question, so
they are checked against the reference itself, which Lucid follows, and
live in the parity tier because they need it installed.
"""

import numpy as np
import pytest

import lucid

pytestmark = pytest.mark.parity

A = np.arange(2 * 3 * 4 * 5, dtype=np.float64).reshape(2, 3, 4, 5)
B = np.arange(2 * 3 * 4 * 5 * 6, dtype=np.float64).reshape(2, 3, 4, 5, 6)

I1 = np.array([0, 2, 1])


def _t(a):
    return lucid.tensor(np.asarray(a).copy())


def _i(a):
    return lucid.tensor(np.asarray(a, dtype=np.int32), dtype=lucid.int32)


def _v(x):
    return np.asarray(x.numpy())


def _same(got, want):
    assert got.shape == want.shape, f"shape {got.shape} != {want.shape}"
    assert np.array_equal(got, want)


def test_the_split_cases_match_the_reference_framework(ref):
    cases = [
        (lambda t, i: t[:, i(I1), :, i([1, 2, 3])], A),
        (lambda t, i: t[1:, i(I1), :, i([1, 2, 3])], A),
        (lambda t, i: t[:, i(I1), None, :, i([1, 2, 3])], A),
        (lambda t, i: t[None, :, i(I1), :, i([1, 2, 3])], A),
        (lambda t, i: t[:, i(I1), :, i([1, 2, 3]), :], B),
        (lambda t, i: t[0, :, i(I1), 1], A),
    ]
    for k, (take, array) in enumerate(cases):
        want = take(ref.tensor(array), lambda a: ref.tensor(np.asarray(a, np.int64)))
        _same(_v(take(_t(array), _i)), want.numpy())
