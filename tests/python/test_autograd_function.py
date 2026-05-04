"""
Tests for lucid.autograd.Function — custom differentiable operations.

Covers:
  - forward-only pass (no grad tracking)
  - backward through save_for_backward / ctx.saved_tensors
  - ctx attribute storage
  - needs_input_grad flag
  - multi-input functions with mixed requires_grad
  - gradient accumulation across multiple backward calls
  - chained custom functions
  - gradcheck verification for several custom ops
  - Function as a module component
"""

import numpy as np
import pytest
import lucid
import lucid.autograd as autograd
from lucid.autograd.function import Function, FunctionCtx
from lucid.autograd.gradcheck import gradcheck


# ── Helper custom functions ───────────────────────────────────────────────────

class Square(Function):
    @staticmethod
    def forward(ctx: FunctionCtx, x):
        ctx.save_for_backward(x)
        return x * x

    @staticmethod
    def backward(ctx: FunctionCtx, grad):
        (x,) = ctx.saved_tensors
        return 2.0 * x * grad


class Negate(Function):
    @staticmethod
    def forward(ctx: FunctionCtx, x):
        return lucid.tensor(-x.numpy())

    @staticmethod
    def backward(ctx: FunctionCtx, grad):
        return lucid.tensor(-grad.numpy())


class ScaleByCtx(Function):
    """Uses ctx attribute to store a scalar constant."""

    @staticmethod
    def forward(ctx: FunctionCtx, x, alpha: float = 2.0):
        ctx.alpha = alpha
        ctx.save_for_backward(x)
        return x * alpha

    @staticmethod
    def backward(ctx: FunctionCtx, grad):
        return grad * ctx.alpha


class Add(Function):
    """Two-input addition — both inputs differentiable."""

    @staticmethod
    def forward(ctx: FunctionCtx, x, y):
        return x + y

    @staticmethod
    def backward(ctx: FunctionCtx, grad):
        return grad, grad


class LinearFunc(Function):
    """y = x @ w + b  (matrix multiply + bias)."""

    @staticmethod
    def forward(ctx: FunctionCtx, x, w, b):
        ctx.save_for_backward(x, w)
        return x @ w + b

    @staticmethod
    def backward(ctx: FunctionCtx, grad):
        x, w = ctx.saved_tensors
        grad_x = grad @ w.T if grad is not None else None
        grad_w = x.T @ grad if grad is not None else None
        grad_b = grad.sum([0]) if grad is not None else None
        return grad_x, grad_w, grad_b


class MixedGrad(Function):
    """One input requires grad, the other doesn't."""

    @staticmethod
    def forward(ctx: FunctionCtx, x, y):
        ctx.save_for_backward(x, y)
        return x * y

    @staticmethod
    def backward(ctx: FunctionCtx, grad):
        x, y = ctx.saved_tensors
        gx = grad * y if ctx.needs_input_grad[0] else None
        gy = grad * x if ctx.needs_input_grad[1] else None
        return gx, gy


# ── Forward-pass behaviour ────────────────────────────────────────────────────

class TestForward:
    def test_square_values(self):
        x = lucid.tensor([[1.0, -2.0, 3.0]])
        y = Square.apply(x)
        np.testing.assert_allclose(y.numpy(), [[1.0, 4.0, 9.0]], atol=1e-6)

    def test_negate_values(self):
        x = lucid.tensor([1.0, 0.0, -3.0])
        y = Negate.apply(x)
        np.testing.assert_allclose(y.numpy(), [-1.0, 0.0, 3.0], atol=1e-6)

    def test_no_grad_mode(self):
        x = lucid.tensor([2.0, 3.0])
        y = Square.apply(x)
        assert not y.requires_grad

    def test_output_requires_grad_when_input_does(self):
        x = lucid.tensor([2.0, 3.0], requires_grad=True)
        y = Square.apply(x)
        assert y.requires_grad

    def test_output_shape_preserved(self):
        x = lucid.randn(3, 4, requires_grad=True)
        y = Square.apply(x)
        assert y.shape == (3, 4)


# ── Backward / gradients ──────────────────────────────────────────────────────

class TestBackward:
    def test_square_gradient(self):
        x = lucid.tensor([[2.0, -3.0, 1.0]], requires_grad=True)
        y = Square.apply(x)
        y.sum().backward()
        np.testing.assert_allclose(x.grad.numpy(), [[4.0, -6.0, 2.0]], atol=1e-6)

    def test_negate_gradient(self):
        x = lucid.tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = Negate.apply(x)
        y.sum().backward()
        np.testing.assert_allclose(x.grad.numpy(), [-1.0, -1.0, -1.0], atol=1e-6)

    def test_scale_by_ctx_gradient(self):
        x = lucid.tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = ScaleByCtx.apply(x, 5.0)
        y.sum().backward()
        np.testing.assert_allclose(x.grad.numpy(), [5.0, 5.0, 5.0], atol=1e-6)

    def test_add_gradient_both_inputs(self):
        a = lucid.tensor([1.0, 2.0], requires_grad=True)
        b = lucid.tensor([3.0, 4.0], requires_grad=True)
        c = Add.apply(a, b)
        c.sum().backward()
        np.testing.assert_allclose(a.grad.numpy(), [1.0, 1.0], atol=1e-6)
        np.testing.assert_allclose(b.grad.numpy(), [1.0, 1.0], atol=1e-6)

    def test_saved_tensors_access(self):
        x = lucid.tensor([3.0, -1.0], requires_grad=True)
        # Verify saved_tensors can be read from ctx before backward completes
        ctx_ref: list[FunctionCtx] = []

        class SaveCapture(Function):
            @staticmethod
            def forward(ctx, x):
                ctx.save_for_backward(x)
                ctx_ref.append(ctx)
                return x * x

            @staticmethod
            def backward(ctx, grad):
                (x,) = ctx.saved_tensors
                return 2.0 * x * grad

        y = SaveCapture.apply(x)
        saved = ctx_ref[0].saved_tensors
        np.testing.assert_allclose(saved[0].numpy(), x.numpy(), atol=1e-6)

    def test_ctx_attribute_survives_backward(self):
        x = lucid.tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = ScaleByCtx.apply(x, 7.0)
        y.sum().backward()
        np.testing.assert_allclose(x.grad.numpy(), [7.0, 7.0, 7.0], atol=1e-6)


# ── needs_input_grad ──────────────────────────────────────────────────────────

class TestNeedsInputGrad:
    def test_only_first_input_requires_grad(self):
        a = lucid.tensor([1.0, 2.0], requires_grad=True)
        b = lucid.tensor([3.0, 4.0], requires_grad=False)
        c = MixedGrad.apply(a, b)
        c.sum().backward()
        # a should have gradient; b.grad should be None
        assert a.grad is not None
        assert b.grad is None

    def test_gradient_value_mixed(self):
        a = lucid.tensor([2.0, 3.0], requires_grad=True)
        b = lucid.tensor([5.0, -1.0], requires_grad=False)
        c = MixedGrad.apply(a, b)
        c.sum().backward()
        # d(a*b)/da = b
        np.testing.assert_allclose(a.grad.numpy(), b.numpy(), atol=1e-6)

    def test_neither_requires_grad(self):
        a = lucid.tensor([1.0, 2.0])
        b = lucid.tensor([3.0, 4.0])
        c = MixedGrad.apply(a, b)
        assert not c.requires_grad


# ── Chained functions ─────────────────────────────────────────────────────────

class TestChaining:
    def test_chain_square_then_negate(self):
        x = lucid.tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = Square.apply(x)   # y = x^2
        z = Negate.apply(y)   # z = -x^2
        z.sum().backward()
        # dz/dx = d(-x^2)/dx = -2x
        np.testing.assert_allclose(x.grad.numpy(), [-2.0, -4.0, -6.0], atol=1e-6)

    def test_chain_scale_then_square(self):
        x = lucid.tensor([1.0, 2.0], requires_grad=True)
        y = ScaleByCtx.apply(x, 3.0)   # y = 3x
        z = Square.apply(y)             # z = 9x^2
        z.sum().backward()
        # dz/dx = 18x
        np.testing.assert_allclose(x.grad.numpy(), [18.0, 36.0], atol=1e-6)

    def test_three_deep_chain(self):
        x = lucid.tensor([2.0], requires_grad=True)
        a = Square.apply(x)             # a = x^2 = 4
        b = ScaleByCtx.apply(a, 2.0)   # b = 2*x^2 = 8
        c = Square.apply(b)             # c = 4*x^4 = 64
        c.sum().backward()
        # dc/dx = 16*x^3, at x=2 → 128
        np.testing.assert_allclose(x.grad.numpy(), [128.0], atol=1e-4)


# ── Multi-output via tuple ────────────────────────────────────────────────────

class TestCtxAttributes:
    def test_multiple_scalars_stored(self):
        class MultiAttr(Function):
            @staticmethod
            def forward(ctx, x):
                ctx.a = 1.0
                ctx.b = 2.0
                ctx.scale = ctx.a + ctx.b  # 3.0
                return x * ctx.scale

            @staticmethod
            def backward(ctx, grad):
                return grad * ctx.scale

        x = lucid.tensor([1.0, 2.0], requires_grad=True)
        y = MultiAttr.apply(x)
        y.sum().backward()
        np.testing.assert_allclose(x.grad.numpy(), [3.0, 3.0], atol=1e-6)

    def test_attr_overwrite(self):
        class Overwrite(Function):
            @staticmethod
            def forward(ctx, x):
                ctx.val = 10.0
                ctx.val = 20.0  # overwrite
                return x

            @staticmethod
            def backward(ctx, grad):
                return grad * ctx.val

        x = lucid.tensor([1.0], requires_grad=True)
        y = Overwrite.apply(x)
        y.sum().backward()
        np.testing.assert_allclose(x.grad.numpy(), [20.0], atol=1e-6)


# ── gradcheck verification ────────────────────────────────────────────────────

class TestGradcheck:
    def test_square_gradcheck(self):
        x = lucid.tensor([[1.0, -2.0, 3.0]], requires_grad=True)
        assert gradcheck(lambda t: Square.apply(t).sum(), [x])

    def test_scale_gradcheck(self):
        x = lucid.tensor([0.5, -1.5, 2.0], requires_grad=True)
        assert gradcheck(lambda t: ScaleByCtx.apply(t, 4.0).sum(), [x])

    def test_add_gradcheck_both(self):
        a = lucid.tensor([1.0, 2.0], requires_grad=True)
        b = lucid.tensor([3.0, -1.0], requires_grad=True)
        assert gradcheck(lambda u, v: Add.apply(u, v).sum(), [a, b])

    def test_chained_gradcheck(self):
        x = lucid.tensor([1.0, 0.5, -1.0], requires_grad=True)
        assert gradcheck(lambda t: Square.apply(ScaleByCtx.apply(t, 2.0)).sum(), [x])

    def test_negate_gradcheck(self):
        x = lucid.tensor([1.0, 2.0, -3.0], requires_grad=True)
        assert gradcheck(lambda t: Negate.apply(t).sum(), [x])

    def test_linear_func_gradcheck(self):
        x = lucid.tensor([[1.0, 0.5], [-0.5, 1.0]], requires_grad=True)
        w = lucid.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
        b = lucid.tensor([0.1, -0.1], requires_grad=True)
        assert gradcheck(
            lambda xi, wi, bi: LinearFunc.apply(xi, wi, bi).sum(),
            [x, w, b],
        )


# ── Interaction with built-in ops ─────────────────────────────────────────────

class TestIntegration:
    def test_custom_then_builtin(self):
        x = lucid.tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = Square.apply(x)
        z = y.sum()  # built-in sum
        z.backward()
        np.testing.assert_allclose(x.grad.numpy(), [2.0, 4.0, 6.0], atol=1e-6)

    def test_builtin_then_custom(self):
        x = lucid.tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = x * x          # built-in multiply
        z = Negate.apply(y)
        z.sum().backward()
        np.testing.assert_allclose(x.grad.numpy(), [-2.0, -4.0, -6.0], atol=1e-6)

    def test_custom_inside_nn_forward(self):
        import lucid.nn as nn

        class SquaredLinear(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(4, 2)

            def forward(self, x):
                return Square.apply(self.linear(x))

        model = SquaredLinear()
        x = lucid.randn(3, 4)
        out = model(x)
        out.sum().backward()
        # Ensure params got gradients
        for p in model.parameters():
            assert p.grad is not None


# ── Function class API contracts ──────────────────────────────────────────────

class TestFunctionAPI:
    def test_function_has_apply_classmethod(self):
        assert callable(Square.apply)

    def test_forward_not_implemented_on_base(self):
        from lucid.autograd.function import Function as _Base
        ctx = FunctionCtx()
        with pytest.raises(NotImplementedError):
            _Base.forward(ctx)

    def test_backward_not_implemented_on_base(self):
        from lucid.autograd.function import Function as _Base
        ctx = FunctionCtx()
        with pytest.raises(NotImplementedError):
            _Base.backward(ctx)

    def test_apply_returns_tensor(self):
        x = lucid.tensor([1.0, 2.0], requires_grad=True)
        y = Square.apply(x)
        assert isinstance(y, lucid.Tensor)

    def test_function_meta_creates_apply(self):
        class NewFunc(Function):
            @staticmethod
            def forward(ctx, x):
                return x

            @staticmethod
            def backward(ctx, grad):
                return grad

        assert hasattr(NewFunc, "apply")
        x = lucid.tensor([1.0])
        assert isinstance(NewFunc.apply(x), lucid.Tensor)
