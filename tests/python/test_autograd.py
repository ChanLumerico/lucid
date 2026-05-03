"""
Tests for autograd: backward, grad, no_grad, enable_grad, Function.
"""

import pytest
import numpy as np
import lucid
import lucid.autograd as autograd
from conftest import assert_close


class TestBackward:
    def test_scalar_backward(self, seed):
        x = lucid.randn(3)
        x.requires_grad_(True)
        y = (x * x).sum()
        y.backward()
        assert_close(x.grad.numpy(), 2 * x.numpy())

    def test_backward_with_gradient(self, seed):
        x = lucid.randn(3)
        x.requires_grad_(True)
        y = x * 3.0
        g = lucid.ones(3)
        y.backward(gradient=g)
        assert_close(x.grad.numpy(), np.full(3, 3.0))

    def test_backward_nonscalar_no_gradient_raises(self):
        x = lucid.randn(3)
        x.requires_grad_(True)
        y = x * 2
        with pytest.raises(RuntimeError, match="scalar"):
            y.backward()

    def test_backward_shape_mismatch_raises(self):
        x = lucid.randn(3)
        x.requires_grad_(True)
        y = x * 2
        with pytest.raises(RuntimeError, match="shape"):
            y.backward(gradient=lucid.ones(4))


class TestGradMode:
    def test_no_grad_disables(self):
        with lucid.no_grad():
            assert not autograd.is_grad_enabled()
        assert autograd.is_grad_enabled()

    def test_no_grad_restores_nested(self):
        assert autograd.is_grad_enabled()
        with lucid.no_grad():
            assert not autograd.is_grad_enabled()
            with lucid.enable_grad():
                assert autograd.is_grad_enabled()
            assert not autograd.is_grad_enabled()
        assert autograd.is_grad_enabled()

    def test_no_grad_decorator(self):
        @lucid.no_grad()
        def fn():
            return autograd.is_grad_enabled()

        assert not fn()
        assert autograd.is_grad_enabled()

    def test_enable_grad_decorator(self):
        @lucid.enable_grad()
        def fn():
            return autograd.is_grad_enabled()

        with lucid.no_grad():
            assert fn()
            assert not autograd.is_grad_enabled()

    def test_set_grad_enabled(self):
        autograd.set_grad_enabled(False)
        assert not autograd.is_grad_enabled()
        autograd.set_grad_enabled(True)
        assert autograd.is_grad_enabled()


class TestAutogradGrad:
    def test_grad_basic(self, seed):
        x = lucid.randn(3)
        x.requires_grad_(True)
        y = (x * x).sum()
        (gx,) = autograd.grad(y, [x])
        assert gx is not None
        assert_close(gx.numpy(), 2 * x.numpy())

    def test_grad_allow_unused(self, seed):
        x = lucid.randn(3)
        x.requires_grad_(True)
        z = lucid.randn(3)
        z.requires_grad_(True)
        y = (x * x).sum()
        gx, gz = autograd.grad(y, [x, z], allow_unused=True)
        assert gx is not None
        assert gz is None

    def test_grad_unused_raises(self, seed):
        x = lucid.randn(3)
        x.requires_grad_(True)
        z = lucid.randn(3)
        z.requires_grad_(True)
        y = (x * x).sum()
        with pytest.raises(RuntimeError):
            autograd.grad(y, [x, z], allow_unused=False)

    def test_grad_with_seed(self, seed):
        x = lucid.randn(3)
        x.requires_grad_(True)
        y = x * 2
        seed_g = lucid.ones(3)
        (gx,) = autograd.grad(y, [x], grad_outputs=[seed_g])
        assert_close(gx.numpy(), np.full(3, 2.0))


class TestInferenceMode:
    def test_inference_mode(self):
        with autograd.inference_mode():
            assert not autograd.is_grad_enabled()
        assert autograd.is_grad_enabled()
