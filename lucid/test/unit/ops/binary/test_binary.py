"""Element-wise binary ops + broadcasting."""

from collections.abc import Callable

import numpy as np
import pytest

import lucid
from lucid.test._fixtures.devices import skip_if_unsupported
from lucid.test._helpers.compare import assert_close


# (name, lucid_fn, numpy_fn, low, high, atol)
_BINARY_OPS: list[tuple[str, Callable, Callable, float, float, float]] = [
    ("add",       lambda a, b: a + b,         np.add,       -2.0, 2.0, 1e-6),
    ("sub",       lambda a, b: a - b,         np.subtract,  -2.0, 2.0, 1e-6),
    ("mul",       lambda a, b: a * b,         np.multiply,  -2.0, 2.0, 1e-6),
    ("div",       lambda a, b: a / b,         np.divide,    0.5, 2.0,  1e-5),
    ("pow",       lambda a, b: a ** b,        np.power,     0.5, 2.0,  1e-4),
    ("maximum",   lambda a, b: lucid.maximum(a, b), np.maximum, -2.0, 2.0, 0.0),
    ("minimum",   lambda a, b: lucid.minimum(a, b), np.minimum, -2.0, 2.0, 0.0),
    ("atan2",     lambda a, b: lucid.atan2(a, b),   np.arctan2, -2.0, 2.0, 1e-5),
    ("hypot",     lambda a, b: lucid.hypot(a, b),   np.hypot,    0.1, 2.0, 1e-4),
    ("fmod",      lambda a, b: lucid.fmod(a, b),    np.fmod,     0.1, 2.0, 1e-5),
    ("remainder", lambda a, b: lucid.remainder(a, b), np.remainder, 0.1, 2.0, 1e-5),
    ("fmax",      lambda a, b: lucid.fmax(a, b),    np.fmax,     -2.0, 2.0, 0.0),
    ("fmin",      lambda a, b: lucid.fmin(a, b),    np.fmin,     -2.0, 2.0, 0.0),
    ("logaddexp", lambda a, b: lucid.logaddexp(a, b), np.logaddexp, -2.0, 2.0, 1e-5),
]


@pytest.mark.parametrize("name,lucid_fn,np_fn,low,high,atol", _BINARY_OPS,
                         ids=[op[0] for op in _BINARY_OPS])
def test_binary_value_match(
    name: str,
    lucid_fn: Callable,
    np_fn: Callable,
    low: float,
    high: float,
    atol: float,
    device: str,
    float_dtype: lucid.dtype,
) -> None:
    skip_if_unsupported(device, float_dtype)
    np.random.seed(0)
    shape = (4, 5)
    a = np.random.uniform(low, high, size=shape).astype(np.float32)
    b = np.random.uniform(low, high, size=shape).astype(np.float32)
    la = lucid.tensor(a.copy(), dtype=float_dtype, device=device)
    lb = lucid.tensor(b.copy(), dtype=float_dtype, device=device)
    out = lucid_fn(la, lb)
    expected = np_fn(a, b)
    assert_close(out, expected, atol=atol)


# ── broadcasting ─────────────────────────────────────────────────────────


_BCAST_PAIRS = [
    ((3,), (3,)),
    ((1,), (4,)),
    ((4, 1), (1, 5)),
    ((2, 3, 1), (1, 3, 5)),
    ((1, 3, 8, 8), (3, 1, 1)),
]


@pytest.mark.parametrize("shape_a,shape_b", _BCAST_PAIRS)
def test_add_broadcasts(
    shape_a: tuple[int, ...],
    shape_b: tuple[int, ...],
    device: str,
) -> None:
    np.random.seed(0)
    a = np.random.uniform(-1.0, 1.0, size=shape_a).astype(np.float32)
    b = np.random.uniform(-1.0, 1.0, size=shape_b).astype(np.float32)
    la = lucid.tensor(a.copy(), device=device)
    lb = lucid.tensor(b.copy(), device=device)
    out = la + lb
    assert_close(out, a + b, atol=1e-6)


# ── scalar mixing ────────────────────────────────────────────────────────


class TestScalarOps:
    def test_add_scalar(self, device: str) -> None:
        t = lucid.tensor([1.0, 2.0, 3.0], device=device)
        out = (t + 5.0).numpy()
        np.testing.assert_array_equal(out, [6.0, 7.0, 8.0])

    def test_radd_scalar(self, device: str) -> None:
        t = lucid.tensor([1.0, 2.0, 3.0], device=device)
        out = (10.0 + t).numpy()
        np.testing.assert_array_equal(out, [11.0, 12.0, 13.0])

    def test_mul_scalar(self, device: str) -> None:
        t = lucid.tensor([1.0, 2.0, 3.0], device=device)
        np.testing.assert_array_equal((t * 3.0).numpy(), [3.0, 6.0, 9.0])

    def test_div_scalar(self, device: str) -> None:
        t = lucid.tensor([2.0, 4.0, 8.0], device=device)
        np.testing.assert_array_equal((t / 2.0).numpy(), [1.0, 2.0, 4.0])


# ── matmul / mm / bmm ────────────────────────────────────────────────────


class TestMatmul:
    def test_2d_2d(self, device: str) -> None:
        a = lucid.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
        b = lucid.tensor([[5.0, 6.0], [7.0, 8.0]], device=device)
        out = (a @ b).numpy()
        np.testing.assert_array_equal(out, [[19.0, 22.0], [43.0, 50.0]])

    def test_mm_alias(self, device: str) -> None:
        a = lucid.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
        b = lucid.tensor([[5.0, 6.0], [7.0, 8.0]], device=device)
        out_mm = lucid.mm(a, b).numpy()
        out_at = (a @ b).numpy()
        np.testing.assert_array_equal(out_mm, out_at)

    def test_bmm(self, device: str) -> None:
        a = lucid.tensor([[[1.0, 2.0], [3.0, 4.0]]] * 2, device=device)
        b = lucid.tensor([[[5.0, 6.0], [7.0, 8.0]]] * 2, device=device)
        out = lucid.bmm(a, b).numpy()
        assert out.shape == (2, 2, 2)


# ── backward sanity ──────────────────────────────────────────────────────


class TestBinaryBackward:
    def test_add_backward(self, device: str) -> None:
        a = lucid.tensor([1.0, 2.0], device=device, requires_grad=True)
        b = lucid.tensor([3.0, 4.0], device=device, requires_grad=True)
        (a + b).sum().backward()
        np.testing.assert_array_equal(a.grad.numpy(), [1.0, 1.0])
        np.testing.assert_array_equal(b.grad.numpy(), [1.0, 1.0])

    def test_mul_backward(self, device: str) -> None:
        a = lucid.tensor([2.0, 3.0], device=device, requires_grad=True)
        b = lucid.tensor([5.0, 7.0], device=device, requires_grad=True)
        (a * b).sum().backward()
        np.testing.assert_array_equal(a.grad.numpy(), [5.0, 7.0])
        np.testing.assert_array_equal(b.grad.numpy(), [2.0, 3.0])
