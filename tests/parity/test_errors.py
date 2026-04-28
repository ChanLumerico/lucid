"""Error-path tests — verify the engine surfaces the right error type and
message at each guard. Complements the value-parity sweep in test_parity.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from lucid._C import engine as E


def _t(a, device=E.Device.CPU, requires_grad=False):
    return E.TensorImpl(a, device, requires_grad)


# --------------------------------------------------------------------------- #
# Dtype mismatch
# --------------------------------------------------------------------------- #

def test_add_dtype_mismatch():
    a = _t(np.zeros((4,), dtype=np.float32))
    b = _t(np.zeros((4,), dtype=np.float64))
    with pytest.raises(Exception, match="(?i)dtype"):
        E.add(a, b)


def test_matmul_dtype_mismatch():
    a = _t(np.zeros((3, 4), dtype=np.float32))
    b = _t(np.zeros((4, 5), dtype=np.float64))
    with pytest.raises(Exception, match="(?i)dtype"):
        E.matmul(a, b)


# --------------------------------------------------------------------------- #
# Device mismatch
# --------------------------------------------------------------------------- #

def test_add_device_mismatch():
    a = _t(np.zeros((4,), dtype=np.float32), device=E.Device.CPU)
    b = _t(np.zeros((4,), dtype=np.float32), device=E.Device.GPU)
    with pytest.raises(Exception, match="(?i)device"):
        E.add(a, b)


# --------------------------------------------------------------------------- #
# Shape mismatch
# --------------------------------------------------------------------------- #

def test_matmul_shape_mismatch():
    a = _t(np.zeros((3, 4), dtype=np.float32))
    b = _t(np.zeros((5, 6), dtype=np.float32))
    with pytest.raises(Exception, match="(?i)shape|dim"):
        E.matmul(a, b)


def test_inner_shape_mismatch():
    a = _t(np.zeros((4,), dtype=np.float32))
    b = _t(np.zeros((5,), dtype=np.float32))
    with pytest.raises(Exception, match="(?i)shape|inner"):
        E.inner(a, b)


# --------------------------------------------------------------------------- #
# Linalg square / float requirements
# --------------------------------------------------------------------------- #

def test_inv_requires_square():
    a = _t(np.zeros((4, 5), dtype=np.float32))
    with pytest.raises(Exception, match="(?i)square|equal"):
        E.linalg.inv(a)


def test_inv_rejects_int():
    a = _t(np.zeros((4, 4), dtype=np.int32))
    with pytest.raises(Exception, match="(?i)f32|float|dtype"):
        E.linalg.inv(a)


def test_cholesky_rejects_non_pd():
    # A negative-definite matrix → spotrf info > 0 → engine raises.
    A = -np.eye(4, dtype=np.float32) - 0.1 * np.ones((4, 4), dtype=np.float32)
    a = _t(A)
    with pytest.raises(Exception, match="(?i)cholesky|info|numerical"):
        E.linalg.cholesky(a, False)


# --------------------------------------------------------------------------- #
# Reduction axis bounds
# --------------------------------------------------------------------------- #

def test_sum_invalid_axis():
    a = _t(np.zeros((4, 5), dtype=np.float32))
    with pytest.raises(Exception):
        E.sum(a, [10], False)


# --------------------------------------------------------------------------- #
# einops parsing
# --------------------------------------------------------------------------- #

def test_einops_missing_arrow():
    a = _t(np.zeros((4, 5), dtype=np.float32))
    with pytest.raises(Exception, match="(?i)->|arrow|pattern"):
        E.einops.rearrange(a, "a b", {})


def test_einsum_operand_count_mismatch():
    a = _t(np.zeros((3, 4), dtype=np.float32))
    b = _t(np.zeros((4, 5), dtype=np.float32))
    with pytest.raises(Exception, match="(?i)operand|count|3"):
        E.einops.einsum("ij,jk,kl->il", [a, b])  # 3 specs, 2 operands
