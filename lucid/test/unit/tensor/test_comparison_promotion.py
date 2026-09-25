"""The comparison operators promote their operands, as arithmetic does.

The engine compares only equal dtypes, and ``==``, ``<`` and the rest handed
it the operands as they came: ``bool_tensor > 0`` and ``int_tensor > 0.5``
raised ``DtypeMismatch`` while ``bool_tensor + 1`` and the functional
``lucid.greater`` promoted and answered.
"""

import operator

import numpy as np
import pytest

import lucid

OPS = [operator.eq, operator.ne, operator.lt, operator.le, operator.gt, operator.ge]


@pytest.mark.parametrize("device", ["cpu", "metal"])
@pytest.mark.parametrize("op", OPS, ids=lambda f: f.__name__)
@pytest.mark.parametrize(
    "left,right",
    [
        (np.array([True, False]), 0),
        (np.array([True, False]), np.array([0, 1])),
        (np.array([0, 1, 2]), 0.5),
        (np.array([0, 1, 2]), np.array([0.5, 1.0, 2.5], dtype=np.float32)),
    ],
    ids=["bool-int", "bool-inttensor", "int-float", "int-floattensor"],
)
def test_mixed_dtype_comparisons_answer_like_numpy(device, op, left, right):
    lhs = lucid.tensor(left, device=device)
    rhs = right if np.isscalar(right) else lucid.tensor(right, device=device)
    got = op(lhs, rhs)
    assert got.dtype == lucid.bool
    np.testing.assert_array_equal(got.numpy(), op(left, right))


def test_a_reflected_scalar_comparison_promotes_too():
    i = lucid.tensor([0, 1, 2])
    np.testing.assert_array_equal((0.5 < i).numpy(), [False, True, True])


def test_where_takes_a_bool_mask_compared_with_an_int():
    b = lucid.tensor([True, False, True])
    got = lucid.where(b > 0, lucid.tensor([1.0, 2.0, 3.0]), lucid.tensor(0.0))
    np.testing.assert_array_equal(got.numpy(), [1.0, 0.0, 3.0])
