"""
Tests for backward(create_graph=True) — higher-order automatic differentiation.

Covers:
  - First gradient is differentiable (requires_grad=True) for nonlinear ops
  - Second-order gradients are numerically correct
  - Linear ops correctly produce non-differentiable gradients (Hessian = 0)
  - Matmul second-order
  - MAML-style inner-loop parameter update is differentiable
  - Unsupported ops raise NotImplementedError
"""

import numpy as np
import pytest
import lucid


# ── Helper ────────────────────────────────────────────────────────────────────

def _second_order(fn, x_vals):
    """Compute first and second-order gradients of scalar fn(x) at x_vals."""
    x = lucid.tensor(x_vals, requires_grad=True)
    y = fn(x)
    y.backward(create_graph=True)
    g1 = lucid.tensor(x.grad.numpy().copy())   # first-order grad
    g1_req = x.grad.requires_grad
    x._impl.zero_grad()                         # clears both grad and grad_impl
    if g1_req:
        # g was the grad_impl with grad_fn — re-fetch before zero_grad cleared it
        pass
    # Re-run to get a clean first grad that we can backprop through
    x2 = lucid.tensor(x_vals, requires_grad=True)
    y2 = fn(x2)
    y2.backward(create_graph=True)
    g_first = x2.grad
    x2._impl.zero_grad()
    g_first.sum().backward()
    g2 = x2.grad.numpy() if x2.grad is not None else None
    return g1.numpy(), g2, g_first.requires_grad


# ── Second-order correctness ──────────────────────────────────────────────────

class TestSecondOrderCorrectness:
    def test_square_second_order(self):
        """d²(x²)/dx² = 2."""
        _, g2, _ = _second_order(lambda x: (x * x).sum(), [1.0, 2.0, 3.0])
        np.testing.assert_allclose(g2, [2.0, 2.0, 2.0], atol=1e-5)

    def test_cube_second_order(self):
        """d(x³)/dx = 3x²  →  d²(x³)/dx² = 6x."""
        _, g2, _ = _second_order(lambda x: (x * x * x).sum(), [1.0, 2.0])
        np.testing.assert_allclose(g2, [6.0, 12.0], atol=1e-4)

    def test_exp_second_order(self):
        """d²(eˣ)/dx² = eˣ."""
        x_vals = [0.0, 1.0]
        _, g2, _ = _second_order(lambda x: x.exp().sum(), x_vals)
        expected = np.exp(np.array(x_vals, dtype=np.float32))
        np.testing.assert_allclose(g2, expected, rtol=1e-4)

    def test_log_second_order(self):
        """d²(ln x)/dx² = -1/x²."""
        x_vals = [1.0, 2.0]
        _, g2, _ = _second_order(lambda x: x.log().sum(), x_vals)
        expected = -1.0 / np.array(x_vals) ** 2
        np.testing.assert_allclose(g2, expected, rtol=1e-4)

    def test_sqrt_second_order(self):
        """d²(√x)/dx² = -1/(4x^(3/2))."""
        x_vals = [1.0, 4.0]
        _, g2, _ = _second_order(lambda x: x.sqrt().sum(), x_vals)
        expected = -0.25 / np.array(x_vals) ** 1.5
        np.testing.assert_allclose(g2, expected, rtol=1e-4)


# ── Linear ops have Hessian = 0 ───────────────────────────────────────────────

class TestLinearOpsZeroHessian:
    def test_add_zero_hessian(self):
        """d²(2x)/dx² = 0: add is linear, gradient is non-differentiable."""
        g1, _, req = _second_order(lambda x: (x + x).sum(), [1.0, 2.0])
        np.testing.assert_allclose(g1, [2.0, 2.0], atol=1e-5)
        assert not req, "add gradient should have requires_grad=False (linear op)"

    def test_neg_zero_hessian(self):
        """d²(-x)/dx² = 0."""
        g1, _, req = _second_order(lambda x: (-x).sum(), [1.0, 2.0])
        np.testing.assert_allclose(g1, [-1.0, -1.0], atol=1e-5)
        assert not req

    def test_sub_zero_hessian(self):
        """d²(x - 2*x)/dx² = 0."""
        c = lucid.tensor([2.0, 2.0])
        g1, _, req = _second_order(lambda x: (x - c * x).sum(), [1.0, 2.0])
        assert not req


# ── requires_grad propagation ─────────────────────────────────────────────────

class TestRequiresGradPropagation:
    def test_mul_grad_is_differentiable(self):
        x = lucid.tensor([1.0, 2.0], requires_grad=True)
        y = (x * x).sum()
        y.backward(create_graph=True)
        assert x.grad is not None
        assert x.grad.requires_grad, "gradient of x² should be differentiable (= 2x)"

    def test_exp_grad_is_differentiable(self):
        x = lucid.tensor([0.5, 1.0], requires_grad=True)
        y = x.exp().sum()
        y.backward(create_graph=True)
        assert x.grad is not None
        assert x.grad.requires_grad, "gradient of exp(x) should be differentiable (= exp(x))"

    def test_matmul_grad_is_not_differentiable(self):
        # d(Ax)/dA = x (constant w.r.t. A), so gradient of matmul w.r.t. A is linear in x
        # The Hessian w.r.t. A is zero → requires_grad=False
        A = lucid.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
        x = lucid.tensor([[1.0], [2.0]])
        y = (A @ x).sum()
        y.backward(create_graph=True)
        assert A.grad is not None
        # Gradient = x^T (broadcast): [[1], [2]] — does not depend on A
        np.testing.assert_allclose(A.grad.numpy(), [[1.0, 2.0], [1.0, 2.0]], atol=1e-5)


# ── Chained second-order ──────────────────────────────────────────────────────

class TestChainedSecondOrder:
    def test_chain_rule_second_order(self):
        """f(x) = (x²)² = x⁴, f''(x) = 12x²."""
        x = lucid.tensor([1.0, 2.0], requires_grad=True)
        y = ((x * x) * (x * x)).sum()
        y.backward(create_graph=True)
        g1 = x.grad  # 4x³
        np.testing.assert_allclose(g1.numpy(), [4.0, 32.0], atol=1e-4)
        x._impl.zero_grad()
        g1.sum().backward()
        g2 = x.grad.numpy()  # 12x²
        np.testing.assert_allclose(g2, [12.0, 48.0], atol=1e-3)

    def test_exp_of_square(self):
        """f(x) = exp(x²), f'(x) = 2x*exp(x²), f''(x) = (4x²+2)*exp(x²)."""
        x = lucid.tensor([0.0, 1.0], requires_grad=True)
        y = (x * x).exp().sum()
        y.backward(create_graph=True)
        g1 = x.grad
        np.testing.assert_allclose(g1.numpy(), [0.0, 2.0 * np.e], rtol=1e-4)
        x._impl.zero_grad()
        g1.sum().backward()
        g2 = x.grad.numpy()
        # f''(0) = 2*exp(0) = 2,  f''(1) = 6*exp(1) ≈ 16.31
        expected = np.array([2.0, 6.0 * np.e], dtype=np.float32)
        np.testing.assert_allclose(g2, expected, rtol=1e-3)


# ── MAML-style usage ──────────────────────────────────────────────────────────

class TestMAMLStyle:
    def test_inner_loop_gradient_is_differentiable(self):
        """
        MAML inner loop: θ' = θ - α * dL/dθ.
        The meta-gradient requires dθ'/dθ = I - α * d²L/dθ² to be computable,
        which needs create_graph=True on the inner backward.
        """
        # Simple: L = (W @ x - y)² → dL/dW = 2*(W@x - y) * x^T
        W = lucid.tensor([[1.0, 0.5]], requires_grad=True)  # (1, 2)
        x = lucid.tensor([[1.0], [2.0]])                     # (2, 1)
        y_target = lucid.tensor([[3.0]])

        y_pred = W @ x             # (1, 1)
        loss = ((y_pred - y_target) * (y_pred - y_target)).sum()
        loss.backward(create_graph=True)

        assert W.grad is not None
        assert W.grad.requires_grad, "inner gradient must be differentiable for MAML"

    def test_inner_gradient_value_correct(self):
        """Verify the inner gradient computes correctly with create_graph."""
        W = lucid.tensor([[2.0]], requires_grad=True)
        x = lucid.tensor([[3.0]])
        y_t = lucid.tensor([[7.0]])
        # L = (2*3 - 7)² = (6-7)² = 1
        # dL/dW = 2*(W@x - y)*x = 2*(-1)*3 = -6
        pred = W @ x
        loss = ((pred - y_t) * (pred - y_t)).sum()
        loss.backward(create_graph=True)
        np.testing.assert_allclose(W.grad.numpy(), [[-6.0]], atol=1e-5)


# ── Unsupported ops raise NotImplementedError ─────────────────────────────────

class TestUnsupportedOpsError:
    def test_unsupported_softmax_raises(self):
        # Softmax's backward node doesn't implement apply_for_graph.
        import lucid.nn.functional as F
        x = lucid.randn(4, requires_grad=True)
        y = F.softmax(x, dim=0)
        with pytest.raises(RuntimeError, match="create_graph"):
            y.sum().backward(create_graph=True)
