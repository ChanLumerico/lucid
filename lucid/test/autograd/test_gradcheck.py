"""Tests using gradcheck to verify finite-difference gradient correctness."""

import pytest
import lucid
from lucid.autograd import gradcheck
from lucid.test.helpers.numerics import make_tensor

# ── Helper: run gradcheck with small perturbation ─────────────────────────────


def _gc(fn, inputs, eps=1e-3, atol=1e-3):
    """Convenience wrapper around lucid.autograd.gradcheck."""
    return gradcheck(fn, inputs, eps=eps, atol=atol)


class TestGradcheckUnary:
    def test_exp(self):
        x = make_tensor(
            (3,), low=-1.0, high=1.0, dtype=lucid.float64, requires_grad=True, seed=1
        )
        assert _gc(lambda t: lucid.sum(lucid.exp(t)), (x,))

    def test_log(self):
        x = make_tensor(
            (3,), low=0.5, high=2.0, dtype=lucid.float64, requires_grad=True, seed=2
        )
        assert _gc(lambda t: lucid.sum(lucid.log(t)), (x,))

    def test_sin(self):
        x = make_tensor(
            (3,), low=-1.0, high=1.0, dtype=lucid.float64, requires_grad=True, seed=3
        )
        assert _gc(lambda t: lucid.sum(lucid.sin(t)), (x,))

    def test_relu(self):
        x = make_tensor(
            (3,), low=0.2, high=2.0, dtype=lucid.float64, requires_grad=True, seed=4
        )
        assert _gc(lambda t: lucid.sum(lucid.relu(t)), (x,))

    def test_sigmoid(self):
        x = make_tensor(
            (3,), low=-1.0, high=1.0, dtype=lucid.float64, requires_grad=True, seed=5
        )
        assert _gc(lambda t: lucid.sum(lucid.sigmoid(t)), (x,))


class TestGradcheckBinary:
    def test_add(self):
        a = make_tensor((3,), dtype=lucid.float64, requires_grad=True, seed=1)
        b = make_tensor((3,), dtype=lucid.float64, requires_grad=True, seed=2)
        assert _gc(lambda x, y: lucid.sum(x + y), (a, b))

    def test_mul(self):
        a = make_tensor((3,), dtype=lucid.float64, requires_grad=True, seed=1)
        b = make_tensor((3,), dtype=lucid.float64, requires_grad=True, seed=2)
        assert _gc(lambda x, y: lucid.sum(x * y), (a, b))

    def test_matmul(self):
        a = make_tensor((3, 4), dtype=lucid.float64, requires_grad=True, seed=1)
        b = make_tensor((4, 2), dtype=lucid.float64, requires_grad=True, seed=2)
        assert _gc(lambda x, y: lucid.sum(lucid.matmul(x, y)), (a, b))


class TestGradcheckNNOps:
    def test_linear(self):
        import lucid.nn.functional as F

        x = make_tensor((3, 4), dtype=lucid.float64, requires_grad=True, seed=1)
        w = make_tensor((5, 4), dtype=lucid.float64, requires_grad=True, seed=2)
        b = make_tensor((5,), dtype=lucid.float64, requires_grad=True, seed=3)
        assert _gc(lambda x_, w_, b_: lucid.sum(F.linear(x_, w_, b_)), (x, w, b))

    def test_softmax(self):
        import lucid.nn.functional as F

        x = make_tensor((3, 5), dtype=lucid.float64, requires_grad=True, seed=1)
        assert _gc(lambda t: lucid.sum(F.softmax(t, dim=-1) * t), (x,))
