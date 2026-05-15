"""Unit tests for Phase 19 FusionPass: fused_linear_relu/gelu + FusedLinear."""

import numpy as np
import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F


class TestFusedLinearRelu:
    """F.fused_linear_relu parity and gradient tests."""

    def test_inference_matches_unfused(self) -> None:
        np.random.seed(0)
        x = lucid.randn(4, 8)
        w = lucid.randn(16, 8)
        b = lucid.randn(16)
        with lucid.no_grad():
            y_fused = F.fused_linear_relu(x, w, b)
        y_ref = lucid.relu(F.linear(x, w, b))
        np.testing.assert_allclose(y_fused.numpy(), y_ref.numpy(), atol=1e-5)

    def test_training_backward(self) -> None:
        x = lucid.randn(3, 4, requires_grad=True)
        w = lucid.randn(8, 4, requires_grad=True)
        b = lucid.randn(8, requires_grad=True)
        F.fused_linear_relu(x, w, b).sum().backward()
        assert x.grad is not None
        assert w.grad is not None
        assert b.grad is not None

    def test_training_gradients_correct(self) -> None:
        """Training grads must match the reference (unfused) backward."""
        np.random.seed(1)
        x_np = np.random.randn(2, 4).astype(np.float32)
        w_np = np.random.randn(6, 4).astype(np.float32)
        b_np = np.random.randn(6).astype(np.float32)

        x1 = lucid.tensor(x_np.copy(), requires_grad=True)
        w1 = lucid.tensor(w_np.copy(), requires_grad=True)
        b1 = lucid.tensor(b_np.copy(), requires_grad=True)
        F.fused_linear_relu(x1, w1, b1).sum().backward()

        x2 = lucid.tensor(x_np.copy(), requires_grad=True)
        w2 = lucid.tensor(w_np.copy(), requires_grad=True)
        b2 = lucid.tensor(b_np.copy(), requires_grad=True)
        lucid.relu(F.linear(x2, w2, b2)).sum().backward()

        np.testing.assert_allclose(x1.grad.numpy(), x2.grad.numpy(), atol=1e-5)
        np.testing.assert_allclose(w1.grad.numpy(), w2.grad.numpy(), atol=1e-5)
        np.testing.assert_allclose(b1.grad.numpy(), b2.grad.numpy(), atol=1e-5)

    def test_output_shape(self) -> None:
        x = lucid.randn(2, 3, 8)
        w = lucid.randn(16, 8)
        b = lucid.randn(16)
        with lucid.no_grad():
            y = F.fused_linear_relu(x, w, b)
        assert y.shape == (2, 3, 16)

    def test_nonnegative_output(self) -> None:
        """ReLU output must be ≥ 0."""
        x = lucid.randn(10, 5)
        w = lucid.randn(8, 5)
        b = lucid.randn(8)
        with lucid.no_grad():
            y = F.fused_linear_relu(x, w, b)
        assert float(y.min().item()) >= 0.0


class TestFusedLinearGelu:
    """F.fused_linear_gelu parity and gradient tests."""

    def test_inference_matches_unfused(self) -> None:
        np.random.seed(2)
        x = lucid.randn(4, 8)
        w = lucid.randn(16, 8)
        b = lucid.randn(16)
        with lucid.no_grad():
            y_fused = F.fused_linear_gelu(x, w, b)
        y_ref = F.gelu(F.linear(x, w, b), approximate="tanh")
        np.testing.assert_allclose(y_fused.numpy(), y_ref.numpy(), atol=1e-5)

    def test_training_backward(self) -> None:
        x = lucid.randn(3, 4, requires_grad=True)
        w = lucid.randn(8, 4, requires_grad=True)
        b = lucid.randn(8, requires_grad=True)
        F.fused_linear_gelu(x, w, b).sum().backward()
        assert x.grad is not None

    def test_exact_erf_fallback(self) -> None:
        """approximate='none' falls back to unfused erf path."""
        x = lucid.randn(3, 4)
        w = lucid.randn(8, 4)
        b = lucid.randn(8)
        with lucid.no_grad():
            y = F.fused_linear_gelu(x, w, b, approximate="none")
        y_ref = F.gelu(F.linear(x, w, b), approximate="none")
        np.testing.assert_allclose(y.numpy(), y_ref.numpy(), atol=1e-5)


class TestFusedLinearModule:
    """nn.FusedLinear module tests."""

    def test_relu_forward_shape(self) -> None:
        m = nn.FusedLinear(8, 16, activation="relu")
        x = lucid.randn(4, 8)
        with lucid.no_grad():
            assert m(x).shape == (4, 16)

    def test_gelu_forward_shape(self) -> None:
        m = nn.FusedLinear(8, 16, activation="gelu")
        x = lucid.randn(4, 8)
        with lucid.no_grad():
            assert m(x).shape == (4, 16)

    def test_no_bias(self) -> None:
        m = nn.FusedLinear(8, 16, activation="relu", bias=False)
        assert m.bias is None
        x = lucid.randn(2, 8)
        with lucid.no_grad():
            assert m(x).shape == (2, 16)

    def test_backward(self) -> None:
        m = nn.FusedLinear(8, 16, activation="relu")
        x = lucid.randn(3, 8, requires_grad=True)
        m(x).sum().backward()
        assert x.grad is not None
        assert m.weight.grad is not None
        assert m.bias.grad is not None

    def test_output_matches_linear_relu(self) -> None:
        """FusedLinear(relu) output == relu(Linear(x)) with same weights."""
        from lucid._C import engine as _C_engine

        m_lin = nn.Linear(8, 16)
        m_fus = nn.FusedLinear(8, 16, activation="relu")
        m_fus.weight._impl = _C_engine.TensorImpl(
            m_lin.weight.numpy().copy(), _C_engine.Device.CPU, True
        )
        m_fus.bias._impl = _C_engine.TensorImpl(
            m_lin.bias.numpy().copy(), _C_engine.Device.CPU, True
        )
        x = lucid.randn(4, 8)
        with lucid.no_grad():
            y_lin = lucid.relu(m_lin(x))
            y_fus = m_fus(x)
        np.testing.assert_allclose(y_fus.numpy(), y_lin.numpy(), atol=1e-5)

    def test_extra_repr(self) -> None:
        m = nn.FusedLinear(4, 8, activation="gelu")
        r = repr(m)
        assert "gelu" in r
        assert "in_features=4" in r

    def test_invalid_activation_raises(self) -> None:
        with pytest.raises(ValueError, match="unsupported activation"):
            nn.FusedLinear(4, 8, activation="sigmoid")

    def test_parameters_count(self) -> None:
        m = nn.FusedLinear(4, 8)
        params = list(m.parameters())
        assert len(params) == 2  # weight + bias
