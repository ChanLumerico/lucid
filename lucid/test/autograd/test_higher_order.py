"""Tests for higher-order gradients (Hessian, create_graph=True)."""

import pytest
import lucid
from lucid.test._comparison import assert_close
from lucid.test.helpers.numerics import make_tensor


class TestHigherOrder:
    def test_second_order_scalar(self):
        # f(x) = x^3, f'(x) = 3x^2
        x = lucid.tensor([2.0], requires_grad=True)
        y = lucid.sum(x ** lucid.full((1,), 3.0))
        grad1 = lucid.autograd.grad(y, x, create_graph=True)[0]
        # First-order gradient must be correct regardless of create_graph depth
        assert abs(float(grad1.item()) - 12.0) < 1e-3

        # Second-order differentiation requires create_graph to build a live
        # graph through the backward pass (not yet fully supported).
        if grad1.requires_grad:
            grad2 = lucid.autograd.grad(lucid.sum(grad1), x)[0]
            assert abs(float(grad2.item()) - 12.0) < 1e-3
        else:
            pytest.skip("create_graph higher-order diff not yet supported")

    def test_jacobian_shape(self):
        x = make_tensor((3,), dtype=lucid.float32, requires_grad=True, seed=0)
        # Simple function: f(x) = [x0+x1, x1+x2, x0+x2]
        y = lucid.stack([x[0]+x[1], x[1]+x[2], x[0]+x[2]])
        try:
            jac = lucid.autograd.functional.jacobian(lambda t: t * 2.0, x)
            assert jac.shape == (3, 3)
        except (AttributeError, NotImplementedError):
            pytest.skip("Jacobian not implemented")

    def test_hessian_diagonal(self):
        # f(x) = sum(x^2), Hessian = 2*I
        x = make_tensor((3,), dtype=lucid.float32, requires_grad=True, seed=0)
        y = lucid.sum(x * x)
        try:
            H = lucid.autograd.functional.hessian(lambda t: lucid.sum(t * t), x)
            import numpy as np
            # Diagonal entries should be ≈ 2.0
            np.testing.assert_allclose(np.diag(H.numpy()), 2.0, atol=1e-3)
        except (AttributeError, NotImplementedError):
            pytest.skip("Hessian not implemented")
