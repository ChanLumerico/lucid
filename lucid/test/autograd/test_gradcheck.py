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
        assert _gc(lambda t: lucid.sum(lucid.nn.functional.relu(t)), (x,))

    def test_sigmoid(self):
        x = make_tensor(
            (3,), low=-1.0, high=1.0, dtype=lucid.float64, requires_grad=True, seed=5
        )
        assert _gc(lambda t: lucid.sum(lucid.nn.functional.sigmoid(t)), (x,))


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


class TestGradgradcheck:
    """Second-order finite-difference checks."""

    def test_cubic(self):
        from lucid.autograd import gradgradcheck

        x = make_tensor(
            (3,), low=-1.0, high=1.0, dtype=lucid.float64, requires_grad=True, seed=1
        )
        assert gradgradcheck(lambda t: lucid.sum(t * t * t), (x,))

    def test_multi_input(self):
        from lucid.autograd import gradgradcheck

        a = make_tensor((2,), dtype=lucid.float64, requires_grad=True, seed=1)
        b = make_tensor((2,), dtype=lucid.float64, requires_grad=True, seed=2)
        assert gradgradcheck(lambda x, y: lucid.sum(x * x * y), (a, b))

    def test_grad_outputs_kwarg_accepted(self):
        from lucid.autograd import gradgradcheck

        # The kwarg is accepted for source-compat with the reference framework;
        # passing it should not crash.
        x = make_tensor((2,), dtype=lucid.float64, requires_grad=True, seed=1)
        assert gradgradcheck(lambda t: lucid.sum(t * t), (x,), grad_outputs=None)


class TestFunctionApplyBase:
    def test_base_apply_raises(self):
        from lucid.autograd import Function

        with pytest.raises(NotImplementedError):
            Function.apply(lucid.tensor([1.0]))

    def test_subclass_apply_works(self):
        from lucid.autograd import Function

        class Square(Function):
            @staticmethod
            def forward(ctx, x):
                ctx.save_for_backward(x)
                return x * x

            @staticmethod
            def backward(ctx, gy):
                (x,) = ctx.saved_tensors
                return gy * 2 * x

        x = lucid.tensor([3.0], requires_grad=True)
        y = Square.apply(x)
        assert hasattr(Square, "apply")
        y.sum().backward()
        assert float(x.grad.item()) == 6.0
