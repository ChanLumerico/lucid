"""Unit tests for binary ops: add, sub, mul, div, pow, matmul, compare, broadcast."""

import pytest
import numpy as np
import lucid
from lucid.test._comparison import assert_close
from lucid.test.helpers.numerics import make_tensor


class TestArithmetic:
    def test_add_tensors(self):
        a = lucid.tensor([1.0, 2.0, 3.0])
        b = lucid.tensor([4.0, 5.0, 6.0])
        assert_close(lucid.add(a, b), lucid.tensor([5.0, 7.0, 9.0]))

    def test_sub_tensors(self):
        a = lucid.tensor([5.0, 5.0])
        b = lucid.tensor([3.0, 2.0])
        assert_close(lucid.sub(a, b), lucid.tensor([2.0, 3.0]))

    def test_mul_tensors(self):
        a = lucid.tensor([2.0, 3.0])
        b = lucid.tensor([4.0, 5.0])
        assert_close(lucid.mul(a, b), lucid.tensor([8.0, 15.0]))

    def test_div_tensors(self):
        a = lucid.tensor([6.0, 9.0])
        b = lucid.tensor([2.0, 3.0])
        assert_close(lucid.div(a, b), lucid.tensor([3.0, 3.0]))

    def test_pow_tensor_int(self):
        a = lucid.tensor([2.0, 3.0])
        b = lucid.tensor([2.0, 2.0])
        assert_close(lucid.pow(a, b), lucid.tensor([4.0, 9.0]))

    def test_add_scalar_rhs(self):
        a = lucid.tensor([1.0, 2.0])
        assert_close(a + 3.0, lucid.tensor([4.0, 5.0]))

    def test_add_scalar_lhs(self):
        a = lucid.tensor([1.0, 2.0])
        assert_close(3.0 + a, lucid.tensor([4.0, 5.0]))

    def test_sub_inplace(self):
        a = lucid.tensor([5.0, 6.0])
        a -= lucid.tensor([1.0, 2.0])
        assert_close(a, lucid.tensor([4.0, 4.0]))


class TestBroadcasting:
    def test_add_broadcast_row(self):
        a = make_tensor((3, 4))
        b = make_tensor((4,))
        r = a + b
        assert r.shape == (3, 4)

    def test_mul_broadcast_col(self):
        a = make_tensor((3, 1))
        b = make_tensor((3, 4))
        r = a * b
        assert r.shape == (3, 4)

    def test_add_broadcast_batch(self):
        a = make_tensor((2, 1, 4))
        b = make_tensor((1, 3, 4))
        r = a + b
        assert r.shape == (2, 3, 4)

    def test_broadcast_values_correct(self):
        a = lucid.tensor([[1.0, 2.0, 3.0]])  # (1, 3)
        b = lucid.tensor([[10.0], [20.0]])  # (2, 1)
        r = a + b
        expected = lucid.tensor([[11.0, 12.0, 13.0], [21.0, 22.0, 23.0]])
        assert_close(r, expected)


class TestMatmul:
    def test_matmul_2d(self):
        a = lucid.tensor([[1.0, 0.0], [0.0, 1.0]])  # identity
        b = lucid.tensor([[2.0, 3.0], [4.0, 5.0]])
        assert_close(lucid.matmul(a, b), b)

    def test_matmul_shape(self):
        a = make_tensor((3, 4))
        b = make_tensor((4, 5))
        r = lucid.matmul(a, b)
        assert r.shape == (3, 5)

    def test_matmul_batch(self):
        a = make_tensor((2, 3, 4))
        b = make_tensor((2, 4, 5))
        r = lucid.matmul(a, b)
        assert r.shape == (2, 3, 5)

    def test_dot_1d(self):
        a = lucid.tensor([1.0, 2.0, 3.0])
        b = lucid.tensor([4.0, 5.0, 6.0])
        result = lucid.dot(a, b)
        assert abs(float(result.item()) - 32.0) < 1e-4

    def test_outer_product(self):
        a = lucid.tensor([1.0, 2.0])
        b = lucid.tensor([3.0, 4.0])
        r = lucid.outer(a, b)
        assert r.shape == (2, 2)
        expected = lucid.tensor([[3.0, 4.0], [6.0, 8.0]])
        assert_close(r, expected)


class TestComparison:
    def test_equal(self):
        a = lucid.tensor([1.0, 2.0, 3.0])
        b = lucid.tensor([1.0, 0.0, 3.0])
        r = lucid.equal(a, b)
        arr = r.numpy()
        assert arr[0] and not arr[1] and arr[2]

    def test_greater(self):
        a = lucid.tensor([3.0, 1.0])
        b = lucid.tensor([1.0, 3.0])
        r = lucid.greater(a, b)
        arr = r.numpy()
        assert arr[0] and not arr[1]

    def test_less_equal(self):
        a = lucid.tensor([1.0, 2.0, 3.0])
        b = lucid.tensor([2.0, 2.0, 2.0])
        r = lucid.less_equal(a, b)
        arr = r.numpy()
        assert arr[0] and arr[1] and not arr[2]


class TestMinMax:
    def test_maximum(self):
        a = lucid.tensor([1.0, 5.0, 3.0])
        b = lucid.tensor([4.0, 2.0, 3.0])
        assert_close(lucid.maximum(a, b), lucid.tensor([4.0, 5.0, 3.0]))

    def test_minimum(self):
        a = lucid.tensor([1.0, 5.0, 3.0])
        b = lucid.tensor([4.0, 2.0, 3.0])
        assert_close(lucid.minimum(a, b), lucid.tensor([1.0, 2.0, 3.0]))
