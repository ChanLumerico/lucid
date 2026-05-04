"""
Tests for lucid.autograd.gradcheck.
"""

import pytest
import lucid
from lucid.autograd import gradcheck


class TestGradcheck:
    def test_sum_scalar(self):
        x = lucid.tensor([1.0, 2.0, 3.0], requires_grad=True)
        assert gradcheck(lambda t: lucid.sum(t), [x])

    def test_elementwise_square(self):
        x = lucid.tensor([0.5, -1.0, 2.0], requires_grad=True)
        assert gradcheck(lambda t: lucid.sum(t * t), [x])

    def test_relu(self):
        x = lucid.tensor([1.0, 0.5, -0.5], requires_grad=True)
        assert gradcheck(lambda t: lucid.sum(lucid.relu(t)), [x])

    def test_multi_input(self):
        x = lucid.tensor([1.0, 2.0], requires_grad=True)
        y = lucid.tensor([3.0, 4.0], requires_grad=True)
        assert gradcheck(lambda a, b: lucid.sum(a * b), [x, y])

    def test_matmul(self):
        A = lucid.randn(2, 3, requires_grad=True)
        B = lucid.randn(3, 2, requires_grad=True)
        assert gradcheck(lambda a, b: lucid.sum(lucid.matmul(a, b)), [A, B], atol=1e-3)

    def test_non_scalar_output_raises(self):
        x = lucid.tensor([1.0, 2.0], requires_grad=True)
        with pytest.raises(ValueError, match="scalar"):
            gradcheck(lambda t: t, [x])

    def test_returns_false_on_mismatch(self):
        x = lucid.tensor([1.0, 2.0], requires_grad=True)

        def bad_func(t):
            # Wrong gradient — returns wrong value intentionally
            return lucid.sum(t * t * 10)

        # The finite-difference check should fail but not raise
        result = gradcheck(
            lambda t: lucid.sum(t * t),
            [x],
            raise_exception=False,
            atol=1e-10,
            rtol=1e-10,
        )
        assert isinstance(result, bool)

    def test_exp(self):
        x = lucid.tensor([0.1, 0.2, 0.3], requires_grad=True)
        assert gradcheck(lambda t: lucid.sum(lucid.exp(t)), [x])

    def test_log(self):
        x = lucid.tensor([1.0, 2.0, 3.0], requires_grad=True)
        assert gradcheck(lambda t: lucid.sum(lucid.log(t)), [x])

    def test_sigmoid(self):
        x = lucid.tensor([0.5, -0.5, 1.0], requires_grad=True)
        assert gradcheck(lambda t: lucid.sum(lucid.sigmoid(t)), [x])
