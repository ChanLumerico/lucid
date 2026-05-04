"""Tests for basic autograd: forward + backward correctness."""

import pytest
import numpy as np
import lucid
from lucid.test._comparison import assert_close
from lucid.test.helpers.numerics import make_tensor


class TestBasicGradients:
    def test_grad_of_sum(self):
        x = lucid.tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = lucid.sum(x)
        y.backward()
        assert_close(x.grad, lucid.ones(3))

    def test_grad_of_mul_by_scalar(self):
        x = lucid.tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = lucid.sum(x * 3.0)
        y.backward()
        assert_close(x.grad, lucid.full((3,), 3.0))

    def test_grad_of_add(self):
        x = lucid.tensor([1.0, 2.0], requires_grad=True)
        y = lucid.tensor([3.0, 4.0], requires_grad=True)
        z = lucid.sum(x + y)
        z.backward()
        assert_close(x.grad, lucid.ones(2))
        assert_close(y.grad, lucid.ones(2))

    def test_grad_of_sub(self):
        x = lucid.tensor([3.0, 5.0], requires_grad=True)
        y = lucid.tensor([1.0, 2.0], requires_grad=True)
        z = lucid.sum(x - y)
        z.backward()
        assert_close(x.grad, lucid.ones(2))
        assert_close(y.grad, lucid.full((2,), -1.0))

    def test_grad_of_mul(self):
        x = lucid.tensor([2.0, 3.0], requires_grad=True)
        y = lucid.tensor([4.0, 5.0], requires_grad=True)
        z = lucid.sum(x * y)
        z.backward()
        assert_close(x.grad, y.detach())
        assert_close(y.grad, x.detach())

    def test_grad_of_matmul(self):
        A = make_tensor((3, 4), requires_grad=True)
        b = make_tensor((4, 2), requires_grad=True)
        y = lucid.sum(lucid.matmul(A, b))
        y.backward()
        assert A.grad is not None
        assert A.grad.shape == (3, 4)
        assert b.grad is not None
        assert b.grad.shape == (4, 2)

    def test_chain_rule(self):
        x = lucid.tensor([2.0], requires_grad=True)
        # f(x) = (x^2 + 1)^2, f'(x) = 2*(x^2+1)*2x = 4x(x^2+1)
        y = lucid.sum((x * x + lucid.ones(1)) ** lucid.full((1,), 2.0))
        y.backward()
        expected = 4.0 * 2.0 * (4.0 + 1.0)  # at x=2: 4*2*5=40
        assert abs(float(x.grad.item()) - expected) < 1e-3

    def test_grad_accumulates_multiple_backward(self):
        x = lucid.tensor([1.0, 2.0], requires_grad=True)
        y = lucid.sum(x * 2.0)
        y.backward()
        first = x.grad.numpy().copy()
        y2 = lucid.sum(x * 3.0)
        y2.backward()
        # Should accumulate
        result = x.grad.numpy()
        np.testing.assert_allclose(result, first + np.array([3.0, 3.0]), atol=1e-5)

    def test_no_grad_context(self):
        x = lucid.tensor([1.0, 2.0], requires_grad=True)
        with lucid.no_grad():
            y = x * 2.0
        assert not y.requires_grad


class TestUnaryGradients:
    def test_grad_of_exp(self):
        x = lucid.tensor([0.0, 1.0], requires_grad=True)
        y = lucid.sum(lucid.exp(x))
        y.backward()
        # grad of exp(x) = exp(x)
        expected = lucid.exp(x.detach())
        assert_close(x.grad, expected, atol=1e-5)

    def test_grad_of_log(self):
        x = lucid.tensor([1.0, 2.0], requires_grad=True)
        y = lucid.sum(lucid.log(x))
        y.backward()
        # grad of log(x) = 1/x
        expected = lucid.reciprocal(x.detach())
        assert_close(x.grad, expected, atol=1e-5)

    def test_grad_of_sqrt(self):
        x = make_tensor((4,), low=0.5, high=2.0, requires_grad=True)
        y = lucid.sum(lucid.sqrt(x))
        y.backward()
        # grad of sqrt(x) = 1 / (2*sqrt(x))
        expected = lucid.reciprocal(lucid.sqrt(x.detach()) * lucid.full((4,), 2.0))
        assert_close(x.grad, expected, atol=1e-5)

    def test_grad_of_sigmoid(self):
        x = make_tensor((4,), low=-2.0, high=2.0, requires_grad=True)
        y = lucid.sum(lucid.sigmoid(x))
        y.backward()
        # grad of sigmoid = sigma(x) * (1 - sigma(x))
        s = lucid.sigmoid(x.detach())
        expected = s * (lucid.ones(4) - s)
        assert_close(x.grad, expected, atol=1e-5)

    def test_grad_of_relu(self):
        x = lucid.tensor([-1.0, 0.5, 2.0], requires_grad=True)
        y = lucid.sum(lucid.relu(x))
        y.backward()
        # grad of relu: 0 for x<0, 1 for x>0
        expected = lucid.tensor([0.0, 1.0, 1.0])
        assert_close(x.grad, expected)


class TestBroadcastGradients:
    def test_broadcast_sum_grad(self):
        x = make_tensor((3, 4), requires_grad=True)
        b = make_tensor((4,), requires_grad=True)
        y = lucid.sum(x + b)
        y.backward()
        assert x.grad.shape == (3, 4)
        assert b.grad.shape == (4,)
        # b's grad should be sum over batch dim
        assert_close(b.grad, lucid.sum(x.grad, dim=0))


class TestGradControl:
    def test_retain_graph(self):
        x = lucid.tensor([2.0], requires_grad=True)
        y = lucid.sum(x * x)
        y.backward(retain_graph=True)
        grad1 = float(x.grad.item())
        x.grad = None
        y.backward()
        grad2 = float(x.grad.item())
        assert abs(grad1 - grad2) < 1e-6

    def test_detach_breaks_grad_flow(self):
        x = lucid.tensor([2.0], requires_grad=True)
        y = x * 2.0
        z = y.detach() * 3.0
        s = lucid.sum(z)
        s.backward()
        assert x.grad is None
